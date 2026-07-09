"""Standalone sentinel CLI for P/D KV-transfer post-transfer validity check.

PITFALLS P1: NIXL LIBFABRIC can silently produce garbage output. Logs and
counter metrics stay clean; only a known-prefix token comparison detects
corruption. This module:

1. Sends a fixed known-prefix prompt (greedy, temperature=0.0) to an
   OpenAI-compatible vLLM endpoint.
2. In `record` mode: writes a fixture file containing the resolved model id,
   the vLLM version when reported, a prompt SHA-256 prefix for cross-version
   pinning, and the first-N generated tokens + first few logprobs.
3. In `check` mode: reloads that fixture and exits non-zero on any token
   mismatch or any logprob drift above `--logprob-epsilon`.

Fixtures are NEVER fabricated in source control. Only `record` mode against a
trusted colocated topology produces a fixture. Drift = NIXL/tier corruption
between P and D.

CLI:
    python3 tests/sentinel.py --mode record --base-url http://localhost:18000/v1
    python3 tests/sentinel.py --mode check  --base-url http://localhost:19100/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Deterministic prompt — long enough to span multiple KV blocks (256 tokens).
# We repeat the pangram ~50 times to force the prompt past several blocks.
# The prefix MUST be deterministic; never sample or randomize it.
# ---------------------------------------------------------------------------
KNOWN_PREFIX: str = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
) * 50

DEFAULT_SERVED_MODEL_NAME = "goodputlab-model"
DEFAULT_FIXTURE_DIR = "tests/_fixtures"
DEFAULT_MAX_TOKENS = 50
DEFAULT_LOGPROB_EPSILON = 1e-3
DEFAULT_TIMEOUT_SECONDS = 60.0
LOGPROB_SAMPLE_SIZE = 5  # how many leading logprobs to record/check


# ---------------------------------------------------------------------------
# Fixture schema
# ---------------------------------------------------------------------------

def _prompt_sha256(prompt: str) -> str:
    """Truncated SHA-256 (16 hex chars) used to pin fixtures to a prompt hash."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _safe_filename_fragment(s: str) -> str:
    """Reduce an arbitrary string to [A-Za-z0-9_.-] for use in fixture filenames."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


@dataclass(frozen=True)
class SentinelFixture:
    """In-memory representation of a sentinel fixture JSON file."""

    model_id: str
    served_model_name: str
    vllm_version: str
    prompt_sha256: str
    prompt_chars: int
    recorded_at: str
    max_tokens: int
    tokens: list[str]
    logprobs_first_n: list[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "served_model_name": self.served_model_name,
            "vllm_version": self.vllm_version,
            "prompt_sha256": self.prompt_sha256,
            "prompt_chars": self.prompt_chars,
            "recorded_at": self.recorded_at,
            "max_tokens": self.max_tokens,
            "tokens": self.tokens,
            "logprobs_first_n": self.logprobs_first_n,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SentinelFixture:
        # Strict field check — any missing field is a tamper signal.
        required = {
            "model_id",
            "served_model_name",
            "vllm_version",
            "prompt_sha256",
            "prompt_chars",
            "recorded_at",
            "max_tokens",
            "tokens",
            "logprobs_first_n",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"fixture missing required fields: {sorted(missing)}")
        return cls(
            model_id=str(data["model_id"]),
            served_model_name=str(data["served_model_name"]),
            vllm_version=str(data["vllm_version"]),
            prompt_sha256=str(data["prompt_sha256"]),
            prompt_chars=int(data["prompt_chars"]),
            recorded_at=str(data["recorded_at"]),
            max_tokens=int(data["max_tokens"]),
            tokens=[str(t) for t in data["tokens"]],
            logprobs_first_n=[float(x) for x in data["logprobs_first_n"]],
        )


# ---------------------------------------------------------------------------
# Live HTTP call against an OpenAI-compatible vLLM endpoint
# ---------------------------------------------------------------------------

def _post_completion(
    base_url: str,
    served_model_name: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """POST /v1/completions with greedy, logprob-bearing settings."""
    url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": served_model_name,
        "prompt": KNOWN_PREFIX,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "logprobs": LOGPROB_SAMPLE_SIZE,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict) or "choices" not in body or not body["choices"]:
        keys = list(body) if isinstance(body, dict) else type(body).__name__
        raise ValueError(f"malformed completion response: keys={keys}")
    return body


def _extract_tokens_and_logprobs(
    body: dict[str, Any], max_tokens: int
) -> tuple[list[str], list[float]]:
    """Pull the token string list and leading logprobs out of a completions body."""
    choice = body["choices"][0]
    logprobs_obj = choice.get("logprobs") or {}
    tokens = logprobs_obj.get("tokens") or []
    token_logprobs = logprobs_obj.get("token_logprobs") or []
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list):
        raise ValueError("logprobs.tokens/token_logprobs must be lists")
    tokens = [str(t) for t in tokens][:max_tokens]
    # First N logprobs; the very first logprob is usually None (no prior token).
    leading: list[float] = []
    for v in token_logprobs[: LOGPROB_SAMPLE_SIZE + 1]:
        if v is None:
            leading.append(0.0)  # canonical placeholder
        else:
            leading.append(float(v))
    leading = leading[:LOGPROB_SAMPLE_SIZE]
    return tokens, leading


def _resolve_vllm_version(base_url: str, timeout: float) -> str:
    """Best-effort probe of vLLM's reported version. Returns 'unknown' on failure."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/version", timeout=min(timeout, 5.0))
        if resp.ok:
            data = resp.json()
            version = data.get("version") if isinstance(data, dict) else None
            if isinstance(version, str) and version:
                return version
    except requests.RequestException:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Record / check
# ---------------------------------------------------------------------------

def record_mode(
    base_url: str,
    served_model_name: str,
    fixture_dir: Path,
    max_tokens: int,
    timeout: float,
) -> int:
    """Send the sentinel prompt and write a fixture JSON. Returns 0 on success."""
    body = _post_completion(base_url, served_model_name, max_tokens, timeout)
    tokens, leading = _extract_tokens_and_logprobs(body, max_tokens)
    vllm_version = _resolve_vllm_version(base_url, timeout)
    fixture = SentinelFixture(
        model_id=served_model_name,
        served_model_name=served_model_name,
        vllm_version=vllm_version,
        prompt_sha256=_prompt_sha256(KNOWN_PREFIX),
        prompt_chars=len(KNOWN_PREFIX),
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        max_tokens=max_tokens,
        tokens=tokens,
        logprobs_first_n=leading,
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"sentinel_{_safe_filename_fragment(fixture.served_model_name)}"
        f"_{_safe_filename_fragment(fixture.vllm_version)}"
        f"_{fixture.prompt_sha256}.json"
    )
    out_path = fixture_dir / fname
    out_path.write_text(json.dumps(fixture.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"SENTINEL RECORD: wrote {out_path} ({len(tokens)} tokens)", file=sys.stderr)
    return 0


def check_mode(
    base_url: str,
    served_model_name: str,
    fixture_dir: Path,
    max_tokens: int,
    logprob_epsilon: float,
    timeout: float,
) -> int:
    """Send the sentinel prompt and compare against every fixture for this model."""
    if not fixture_dir.exists():
        print(f"SENTINEL FAIL: fixture dir missing: {fixture_dir}", file=sys.stderr)
        return 1
    # Find a fixture matching this served model name (any version hash).
    glob_pat = f"sentinel_{_safe_filename_fragment(served_model_name)}_*.json"
    matches = sorted(fixture_dir.glob(glob_pat))
    if not matches:
        print(
            f"SENTINEL FAIL: no fixture for served-model-name="
            f"{served_model_name!r} in {fixture_dir}",
            file=sys.stderr,
        )
        return 1
    fixture_path = matches[0]
    try:
        fixture = SentinelFixture.from_json(json.loads(fixture_path.read_text(encoding="utf-8")))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"SENTINEL FAIL: malformed fixture {fixture_path}: {exc}", file=sys.stderr)
        return 1
    # Prompt hash must match (catches a drift in the KNOWN_PREFIX itself).
    if fixture.prompt_sha256 != _prompt_sha256(KNOWN_PREFIX):
        print(
            f"SENTINEL FAIL: prompt hash mismatch (fixture={fixture.prompt_sha256}, current="
            f"{_prompt_sha256(KNOWN_PREFIX)})",
            file=sys.stderr,
        )
        return 1
    # Issue the live call.
    try:
        body = _post_completion(base_url, served_model_name, max_tokens, timeout)
    except requests.RequestException as exc:
        print(f"SENTINEL FAIL: HTTP error against {base_url}: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"SENTINEL FAIL: {exc}", file=sys.stderr)
        return 1
    tokens, leading = _extract_tokens_and_logprobs(body, max_tokens)
    # Token exact-match.
    expected_tokens = fixture.tokens[: len(tokens)]
    if tokens != expected_tokens:
        print(
            f"SENTINEL FAIL: token mismatch (got={tokens[:5]} expected={expected_tokens[:5]})",
            file=sys.stderr,
        )
        return 1
    # Logprob drift check (epsilon-bounded L_inf).
    n = min(len(leading), len(fixture.logprobs_first_n))
    for i in range(n):
        if abs(leading[i] - fixture.logprobs_first_n[i]) > logprob_epsilon:
            print(
                f"SENTINEL FAIL: logprob drift at index {i}: "
                f"got={leading[i]:.6f} expected={fixture.logprobs_first_n[i]:.6f} "
                f"epsilon={logprob_epsilon}",
                file=sys.stderr,
            )
            return 1
    print(f"SENTINEL PASS: {len(tokens)} tokens matched {fixture_path.name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Known-prefix P/D KV-transfer validity probe (PITFALLS P1).",
    )
    parser.add_argument(
        "--mode",
        choices=["record", "check"],
        default="check",
        help="record writes a fixture; check compares against the existing fixture",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SENTINEL_BASE_URL", "http://localhost:19100/v1"),
        help="OpenAI-compatible base URL (default: env SENTINEL_BASE_URL or http://localhost:19100/v1)",
    )
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_SERVED_MODEL_NAME,
        help=f"served model name as registered with vLLM (default: {DEFAULT_SERVED_MODEL_NAME})",
    )
    parser.add_argument(
        "--fixture-dir",
        default=DEFAULT_FIXTURE_DIR,
        help=f"fixture directory (default: {DEFAULT_FIXTURE_DIR})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"greedy max_tokens to record/check (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--logprob-epsilon",
        type=float,
        default=DEFAULT_LOGPROB_EPSILON,
        help=f"absolute logprob drift tolerance (default: {DEFAULT_LOGPROB_EPSILON})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP request timeout seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fixture_dir = Path(args.fixture_dir)
    if args.mode == "record":
        return record_mode(
            base_url=args.base_url,
            served_model_name=args.served_model_name,
            fixture_dir=fixture_dir,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    return check_mode(
        base_url=args.base_url,
        served_model_name=args.served_model_name,
        fixture_dir=fixture_dir,
        max_tokens=args.max_tokens,
        logprob_epsilon=args.logprob_epsilon,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())