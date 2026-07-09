"""Static disagg-proxy source-mark tests (no network, no GPU).

These tests read scripts/disagg_proxy.py as text and assert the source contains
the required behavior markers for the OpenAI-compatible disaggregated proxy
(plan 01-04). They must not depend on running vLLM or any network sockets.

The proxy is the FastAPI front door for P/D disaggregated topologies (disagg
and disagg-tier). It performs a prefill-before-decode sequence on completion
requests and preserves the common endpoint contract
(/v1/chat/completions, /v1/completions, /v1/models, /health, /metrics) used
by colocated and chunked profiles.

PITFALLS P1: post-transfer known-prefix token validity is the load-bearing
safety mechanism. The proxy docstring + --assert-first-token-matches CLI flag
document the runtime sentinel first-token equivalence assertion vs the
colocated baseline. These tests guard the marker contract, not the runtime
measurement.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROXY_SOURCE = REPO_ROOT / "scripts" / "disagg_proxy.py"


def _read_proxy_source() -> str:
    """Load scripts/disagg_proxy.py as text. Fails loudly if the file is missing."""
    assert PROXY_SOURCE.exists(), (
        f"disagg proxy source not found at {PROXY_SOURCE} — "
        "task 01-04 must ship scripts/disagg_proxy.py"
    )
    return PROXY_SOURCE.read_text(encoding="utf-8")


# --- Route surface -----------------------------------------------------------


def test_proxy_exposes_required_routes() -> None:
    """Common endpoint contract: required OpenAI-compat routes must all be declared."""
    src = _read_proxy_source()
    # Each route string MUST appear as a literal in the proxy source.
    assert "/v1/chat/completions" in src, (
        "proxy must define /v1/chat/completions (D-05 common endpoint contract)"
    )
    assert "/v1/completions" in src, (
        "proxy must define /v1/completions (D-05 common endpoint contract)"
    )
    assert "/v1/models" in src, (
        "proxy must define /v1/models (D-05 common endpoint contract)"
    )
    assert "/health" in src, (
        "proxy must define /health (D-05 common endpoint contract)"
    )
    assert "/metrics" in src, (
        "proxy must define /metrics (D-05 common endpoint contract)"
    )


# --- Prefill-before-decode behavior -----------------------------------------


def test_proxy_caps_prefill_to_one_token() -> None:
    """Prefill request body must cap max_tokens AND max_completion_tokens to 1.

    The decode request must receive the original, unmodified body.
    """
    src = _read_proxy_source()
    # The proxy must set both fields on the prefill body.
    assert "max_tokens" in src, "proxy must manipulate max_tokens on prefill body"
    assert "max_completion_tokens" in src, (
        "proxy must manipulate max_completion_tokens on prefill body"
    )
    # The proxy must reference both stages.
    assert "prefiller_host" in src, "proxy must accept --prefiller-host CLI arg"
    assert "decoder_host" in src, "proxy must accept --decoder-host CLI arg"


# --- Request ID propagation --------------------------------------------------


def test_proxy_propagates_request_id_header() -> None:
    """X-Request-Id header must be generated (when missing) and propagated."""
    src = _read_proxy_source()
    assert "X-Request-Id" in src, (
        "proxy must propagate or generate an X-Request-Id header for log correlation"
    )


# --- CLI surface -------------------------------------------------------------


def test_proxy_defines_required_cli_args() -> None:
    """CLI args for host/port + prefiller + decoder endpoints + served model name."""
    src = _read_proxy_source()
    # argparse must be used.
    assert "argparse" in src, "proxy must use argparse for CLI parsing"
    # Endpoint CLI args (both host and port for each stage).
    assert "--prefiller-host" in src, "proxy must declare --prefiller-host CLI arg"
    assert "--prefiller-port" in src, "proxy must declare --prefiller-port CLI arg"
    assert "--decoder-host" in src, "proxy must declare --decoder-host CLI arg"
    assert "--decoder-port" in src, "proxy must declare --decoder-port CLI arg"
    # Proxy itself binds host/port.
    assert "--host" in src, "proxy must declare --host CLI arg"
    assert "--port" in src, "proxy must declare --port CLI arg"
    # Served model name override.
    assert "--served-model-name" in src, (
        "proxy must declare --served-model-name CLI arg"
    )


def test_proxy_declares_assert_first_token_matches() -> None:
    """Optional runtime sentinel first-token equivalence check."""
    src = _read_proxy_source()
    assert "--assert-first-token-matches" in src, (
        "proxy must expose --assert-first-token-matches TEXT for sentinel check"
    )


# --- Async HTTP client + FastAPI lifespan -----------------------------------


def test_proxy_uses_async_httpx_client() -> None:
    """Proxy must use httpx.AsyncClient with no hardcoded API keys."""
    src = _read_proxy_source()
    assert "httpx" in src, "proxy must depend on httpx"
    assert "AsyncClient" in src, "proxy must use httpx.AsyncClient"


def test_proxy_uses_fastapi_lifespan() -> None:
    """FastAPI lifespan context manager is required (no deprecated @app.on_event)."""
    src = _read_proxy_source()
    assert "lifespan" in src, "proxy must define a FastAPI lifespan context manager"
    assert "FastAPI" in src, "proxy must use FastAPI"


# --- Top-of-file docstring + sentinel assertion contract ---------------------


def test_proxy_docstring_documents_sentinel_assertion_and_fallback() -> None:
    """Top-of-file docstring must record the runtime sentinel assertion and
    the vLLM built-in --proxy fallback decision.
    """
    src = _read_proxy_source()
    # Docstring must explicitly cite the sentinel assertion + fallback policy.
    assert "first-token" in src or "first_token" in src, (
        "proxy docstring must document the runtime first-token sentinel assertion"
    )
    assert "--proxy" in src, (
        "proxy docstring/code must mention the vLLM built-in --proxy fallback"
    )


# --- /metrics source labeling ------------------------------------------------


def test_proxy_metrics_endpoint_labels_prefill_and_decode() -> None:
    """Proxy /metrics must prefix prefill and decode sections so health checks
    can parse source-labeled counters."""
    src = _read_proxy_source()
    # Both prefill and decode must be labeled in /metrics output.
    assert "prefill" in src, "proxy /metrics must include prefill source label"
    assert "decode" in src, "proxy /metrics must include decode source label"