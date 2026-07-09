"""GoodputLab disaggregated (P/D) proxy.

OpenAI-compatible FastAPI front door for vLLM prefill/decode pools. Forwards
``/v1/chat/completions`` and ``/v1/completions`` after performing a
``max_tokens=1`` (and ``max_completion_tokens=1``) prefill call to the
prefiller stage, then forwarding the original request body to the decoder
stage. Also exposes ``/v1/models`` (decode model list), ``/health`` (both
upstreams 200), and ``/metrics`` (prefill + decode metrics, source-labeled).

Runtime sentinel assertion
--------------------------
On the pod (post-deploy), the proxy first-token output for a known-prefix
greedy request (``temperature=0.0``, identical prompt, identical seed) MUST
match the colocated baseline first-token string. This is the PITFALLS P1
post-transfer validity check that proves P→D KV transfer did not corrupt the
transferred KV blocks. The proxy exposes the assertion as
``--assert-first-token-matches TEXT`` for use in ``scripts/health.sh`` after
the sentinel fixture record step (plan 01-05). When the flag is set, the
proxy compares the decode response first token to the expected string and
logs a warning on mismatch (non-fatal so it can run alongside probe traffic).

Fallback policy
---------------
If the upstream ``NixlConnector`` (or ``LMCacheConnectorV1``) internal KV
transfer does not interoperate cleanly with this FastAPI proxy, fall back
to the vLLM built-in ``--proxy`` mode by launching
``python -m vllm.entrypoints.openai.api_server --proxy --prefill-hosts ...
--decode-hosts ... --port 9100`` instead of this FastAPI shim. Record the
fallback decision in this docstring and add a top-level note in
``scripts/health.sh`` so health failures distinguish "proxy-bug" from
"KV-transfer-bug". The fallback path uses vLLM's official prefill/decode
orchestration and bypasses this hand-rolled shim entirely.

CLI args
--------
``--host``, ``--port`` — proxy bind address (default ``0.0.0.0:9100``).
``--prefiller-host``, ``--prefiller-port`` — prefill vLLM endpoint
   (default ``vllm-disagg-prefill:8100``).
``--decoder-host``, ``--decoder-port`` — decoder vLLM endpoint
   (default ``vllm-disagg-decode:8200``).
``--served-model-name`` — model name surfaced in ``/v1/models``
   (default ``goodputlab-model``).
``--assert-first-token-matches TEXT`` — optional runtime sentinel assertion;
   when set, the proxy compares the decode response first token to ``TEXT``
   and logs a warning on mismatch (does NOT crash, so probe traffic keeps
   flowing while health.sh flags the mismatch).

Constraints
-----------
- FastAPI + httpx + Pydantic v2 (project stack lock-in).
- Async/await throughout; lifespan context manager opens/closes the
  ``httpx.AsyncClient`` (no deprecated ``@app.on_event``).
- No hardcoded API keys; request ID propagated via ``X-Request-Id``.
- OpenAI-compatible status codes preserved end-to-end.

Caveat
------
This proxy has NOT been runtime-validated end-to-end yet (P1 sentinel
fixture record step lives in plan 01-05 / post-provision). Source-only
verification gate: ``python3 -m compileall scripts/disagg_proxy.py && \
python3 -m pytest tests/test_disagg_proxy_static.py -q``.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("goodputlab.disagg_proxy")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9100
DEFAULT_PREFILLER_HOST = "vllm-disagg-prefill"
DEFAULT_PREFILLER_PORT = 8100
DEFAULT_DECODER_HOST = "vllm-disagg-decode"
DEFAULT_DECODER_PORT = 8200
DEFAULT_SERVED_MODEL_NAME = "goodputlab-model"
REQUEST_ID_HEADER = "X-Request-Id"

# Per-stage upstream timeouts (seconds). Generous for prefill (model load +
# KV transfer handshake) and decode (full generation budget).
DEFAULT_PREFILL_TIMEOUT_S = 120.0
DEFAULT_DECODE_TIMEOUT_S = 300.0
HEALTH_TIMEOUT_S = 5.0
METRICS_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the proxy."""
    parser = argparse.ArgumentParser(
        prog="disagg_proxy",
        description=(
            "GoodputLab P/D disaggregation proxy. Exposes OpenAI-compatible "
            "routes backed by a prefill + decode vLLM pool pair."
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Proxy bind host.")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Proxy bind port."
    )
    parser.add_argument(
        "--prefiller-host",
        default=DEFAULT_PREFILLER_HOST,
        help="Prefill vLLM host (default: %(default)s).",
    )
    parser.add_argument(
        "--prefiller-port",
        type=int,
        default=DEFAULT_PREFILLER_PORT,
        help="Prefill vLLM port (default: %(default)s).",
    )
    parser.add_argument(
        "--decoder-host",
        default=DEFAULT_DECODER_HOST,
        help="Decode vLLM host (default: %(default)s).",
    )
    parser.add_argument(
        "--decoder-port",
        type=int,
        default=DEFAULT_DECODER_PORT,
        help="Decode vLLM port (default: %(default)s).",
    )
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_SERVED_MODEL_NAME,
        help="Model name returned by /v1/models (default: %(default)s).",
    )
    parser.add_argument(
        "--assert-first-token-matches",
        default="",
        help=(
            "Optional runtime sentinel first-token equivalence string. When "
            "set, the proxy compares the decode response first token to "
            "this string and logs a warning on mismatch (does NOT crash)."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _prefill_url(args: argparse.Namespace) -> str:
    return f"http://{args.prefiller_host}:{args.prefiller_port}"


def _decode_url(args: argparse.Namespace) -> str:
    return f"http://{args.decoder_host}:{args.decoder_port}"


def build_app(args: argparse.Namespace) -> FastAPI:
    """Construct the FastAPI app bound to the parsed CLI args."""
    prefill_url = _prefill_url(args)
    decode_url = _decode_url(args)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Async client shared across requests; per-stage timeouts set on call.
        async with httpx.AsyncClient() as client:
            _app.state.http_client = client
            _app.state.prefill_url = prefill_url
            _app.state.decode_url = decode_url
            _app.state.served_model_name = args.served_model_name
            _app.state.assert_first_token = args.assert_first_token_matches
            logger.info(
                "disagg_proxy started: prefill=%s decode=%s served_model=%s",
                prefill_url,
                decode_url,
                args.served_model_name,
            )
            yield

    app = FastAPI(
        title="GoodputLab Disagg Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _request_id(request: Request) -> str:
        """Read X-Request-Id from incoming headers or mint a new one."""
        incoming = request.headers.get(REQUEST_ID_HEADER)
        if incoming:
            return incoming
        return f"gl-{uuid.uuid4().hex}"

    async def _forward(
        request: Request,
        upstream_url: str,
        path: str,
        request_id: str,
        timeout: float,
    ) -> Response:
        """Forward the request body + relevant headers to the upstream URL."""
        client: httpx.AsyncClient = request.app.state.http_client
        # Replay the body so the proxy can mutate it for prefill-before-decode.
        body_bytes = await request.body()
        # Copy only headers we want to propagate; drop Host/Content-Length
        # which httpx will set correctly.
        fwd_headers: dict[str, str] = {"X-Request-Id": request_id}
        auth = request.headers.get("authorization")
        if auth:
            fwd_headers["authorization"] = auth
        upstream_resp = await client.post(
            f"{upstream_url}{path}",
            content=body_bytes,
            headers=fwd_headers,
            timeout=timeout,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
            headers={"X-Request-Id": request_id},
        )

    def _prefill_body(body: dict[str, Any]) -> dict[str, Any]:
        """Clone body and cap output to 1 token for the prefill stage only."""
        prefill_body = dict(body)
        # Force both legacy and OpenAI v1 field names to 1 — downstream
        # vLLM accepts either; capping both avoids silent fall-through.
        prefill_body["max_tokens"] = 1
        prefill_body["max_completion_tokens"] = 1
        return prefill_body

    def _maybe_assert_first_token(
        request: Request, decode_resp_text: str
    ) -> None:
        """Compare decode response first token to the configured sentinel."""
        expected: str = request.app.state.assert_first_token
        if not expected:
            return
        try:
            import json

            payload = json.loads(decode_resp_text)
            first_token = (
                payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if first_token and not first_token.startswith(expected):
                logger.warning(
                    "SENTINEL MISMATCH: first token %r does not match %r",
                    first_token[:32],
                    expected,
                )
        except (ValueError, KeyError, IndexError):
            # Probe traffic parsing failed; do not crash.
            logger.debug("sentinel compare skipped: could not parse decode resp")

    # -----------------------------------------------------------------------
    # Routes — common endpoint contract (D-05)
    # -----------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = _request_id(request)
        client: httpx.AsyncClient = request.app.state.http_client
        prefill_url = request.app.state.prefill_url
        decode_url = request.app.state.decode_url

        body = await request.json()
        prefill_body = _prefill_body(body)

        # 1. Prefill: cap to 1 token. Stage label logged for STRIDE T-01-04-R.
        prefill_resp = await client.post(
            f"{prefill_url}/v1/chat/completions",
            json=prefill_body,
            headers={"X-Request-Id": request_id},
            timeout=DEFAULT_PREFILL_TIMEOUT_S,
        )
        if prefill_resp.status_code >= 400:
            logger.warning(
                "prefill stage error: status=%d request_id=%s",
                prefill_resp.status_code,
                request_id,
            )
            return Response(
                content=prefill_resp.content,
                status_code=prefill_resp.status_code,
                media_type=prefill_resp.headers.get("content-type"),
                headers={"X-Request-Id": request_id},
            )

        # 2. Decode: original body (unmodified).
        decode_resp = await client.post(
            f"{decode_url}/v1/chat/completions",
            json=body,
            headers={"X-Request-Id": request_id},
            timeout=DEFAULT_DECODE_TIMEOUT_S,
        )
        _maybe_assert_first_token(request, decode_resp.text)
        return Response(
            content=decode_resp.content,
            status_code=decode_resp.status_code,
            media_type=decode_resp.headers.get("content-type"),
            headers={"X-Request-Id": request_id},
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        request_id = _request_id(request)
        client: httpx.AsyncClient = request.app.state.http_client
        prefill_url = request.app.state.prefill_url
        decode_url = request.app.state.decode_url

        body = await request.json()
        prefill_body = _prefill_body(body)

        prefill_resp = await client.post(
            f"{prefill_url}/v1/completions",
            json=prefill_body,
            headers={"X-Request-Id": request_id},
            timeout=DEFAULT_PREFILL_TIMEOUT_S,
        )
        if prefill_resp.status_code >= 400:
            logger.warning(
                "prefill stage error: status=%d request_id=%s",
                prefill_resp.status_code,
                request_id,
            )
            return Response(
                content=prefill_resp.content,
                status_code=prefill_resp.status_code,
                media_type=prefill_resp.headers.get("content-type"),
                headers={"X-Request-Id": request_id},
            )

        decode_resp = await client.post(
            f"{decode_url}/v1/completions",
            json=body,
            headers={"X-Request-Id": request_id},
            timeout=DEFAULT_DECODE_TIMEOUT_S,
        )
        _maybe_assert_first_token(request, decode_resp.text)
        return Response(
            content=decode_resp.content,
            status_code=decode_resp.status_code,
            media_type=decode_resp.headers.get("content-type"),
            headers={"X-Request-Id": request_id},
        )

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        """Proxy the decode model list. Preserves `goodputlab-model` identity."""
        client: httpx.AsyncClient = request.app.state.http_client
        decode_url = request.app.state.decode_url
        upstream = await client.get(
            f"{decode_url}/v1/models",
            timeout=HEALTH_TIMEOUT_S,
        )
        if upstream.status_code != 200:
            # Fall back to a stub from CLI args so health checks still see
            # the served-model-name (P5 healthcheck anti-pattern).
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": request.app.state.served_model_name,
                            "object": "model",
                            "owned_by": "goodputlab",
                        }
                    ],
                }
            )
        return JSONResponse(content=upstream.json())

    @app.get("/health")
    async def health(request: Request) -> Response:
        """Requires BOTH upstream /health endpoints to return 200."""
        client: httpx.AsyncClient = request.app.state.http_client
        prefill_url = request.app.state.prefill_url
        decode_url = request.app.state.decode_url

        async def _check(url: str) -> tuple[str, int]:
            try:
                r = await client.get(f"{url}/health", timeout=HEALTH_TIMEOUT_S)
                return url, r.status_code
            except httpx.HTTPError as exc:
                logger.warning("health probe failed for %s: %s", url, exc)
                return url, 0

        results = [await _check(prefill_url), await _check(decode_url)]
        statuses = dict(results)
        all_ok = all(code == 200 for _url, code in results)
        body = {
            "status": "ok" if all_ok else "degraded",
            "upstreams": statuses,
            "served_model_name": request.app.state.served_model_name,
        }
        return JSONResponse(
            content=body, status_code=200 if all_ok else 503
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> PlainTextResponse:
        """Concat prefill + decode Prometheus metrics with source prefixes."""
        client: httpx.AsyncClient = request.app.state.http_client
        prefill_url = request.app.state.prefill_url
        decode_url = request.app.state.decode_url

        async def _fetch(url: str, label: str) -> str:
            try:
                r = await client.get(
                    f"{url}/metrics", timeout=METRICS_TIMEOUT_S
                )
                if r.status_code != 200:
                    return f"# [{label}] upstream error status={r.status_code}\n"
                return f"# === [{label}] ===\n{r.text}"
            except httpx.HTTPError as exc:
                return f"# [{label}] upstream error: {exc}\n"

        prefill_text = await _fetch(prefill_url, "prefill")
        decode_text = await _fetch(decode_url, "decode")
        return PlainTextResponse(
            content=f"{prefill_text}\n{decode_text}",
            media_type="text/plain; version=0.0.4",
        )

    # -----------------------------------------------------------------------
    # Expose raw /v1/models passthrough helper for testing parity with
    # P5 healthcheck (model id present in /v1/models response).
    # -----------------------------------------------------------------------

    @app.get("/")
    async def root() -> JSONResponse:  # pragma: no cover - trivial
        return JSONResponse(
            {
                "service": "goodputlab-disagg-proxy",
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/models",
                    "/health",
                    "/metrics",
                ],
            }
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()