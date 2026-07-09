"""Runtime smoke tests for the four Phase 1 topology profiles.

These tests exercise the OpenAI-compatible `/v1/chat/completions` endpoint
that every topology profile must expose (D-05: common endpoint contract).

By default the tests SKIP because the RunPod pod `t3son251d5gcvg` is
usually stopped during local development. To force execution, set:

    GOODPUTLAB_RUN_LIVE=1 pytest tests/test_topos.py

When live, the tests assume the corresponding `make up-*` target is
already running on the developer machine or the pod.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests

# Served model name is fixed by D-05 (common endpoint contract).
SERVED_MODEL_NAME = "goodputlab-model"

# Topology name → base URL. Ports come straight from docker-compose.yml.
TOPOLOGY_BASE_URLS: dict[str, str] = {
    "colocated": "http://localhost:18000/v1",
    "chunked": "http://localhost:18001/v1",
    "disagg": "http://localhost:19100/v1",
    "disagg-tier": "http://localhost:19200/v1",
}

TOPOLOGY_NAMES = sorted(TOPOLOGY_BASE_URLS.keys())

# Live gate: default (unset / != '1') → skip. '1' → run.
LIVE_GATE_ENV = "GOODPUTLAB_RUN_LIVE"


def _live_mode() -> bool:
    """True iff GOODPUTLAB_RUN_LIVE is explicitly set to '1'."""
    return os.environ.get(LIVE_GATE_ENV) == "1"


# Skip reason shown when the gate is not enabled. Names the make target.
_SKIP_REASON = (
    "set GOODPUTLAB_RUN_LIVE=1 to run live topology smoke tests "
    "(requires `make up-colocated` / `make up-chunked` / "
    "`make up-disagg` / `make up-disagg-tier`)"
)


def _make_skipper() -> pytest.MarkDecorator:
    """Return a skipif marker that activates when live mode is disabled."""
    return pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)


def _chat_payload() -> dict[str, Any]:
    """Deterministic short chat request used by the smoke tests."""
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Say the word ok."},
        ],
        "temperature": 0.0,
        "max_tokens": 8,
    }


def _extract_content(response_json: dict[str, Any]) -> str:
    """Pull non-empty text out of an OpenAI-style chat response."""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    # Some OpenAI-compatible servers emit `text` instead of `message.content`.
    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


@pytest.mark.parametrize("topology", TOPOLOGY_NAMES)
@pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)
def test_topology_chat_completion(topology: str) -> None:
    """Every topology profile must serve `/v1/chat/completions` with model
    `goodputlab-model` and return at least one non-empty choice.
    """
    base_url = TOPOLOGY_BASE_URLS[topology]
    url = f"{base_url}/chat/completions"

    response = requests.post(url, json=_chat_payload(), timeout=30)
    assert response.status_code == 200, (
        f"{topology}: expected HTTP 200 from {url}, got {response.status_code} "
        f"({response.text[:200]!r})"
    )

    payload = response.json()
    assert "choices" in payload, (
        f"{topology}: response missing 'choices' field: {payload!r}"
    )
    assert payload["choices"], (
        f"{topology}: 'choices' list was empty: {payload!r}"
    )

    content = _extract_content(payload)
    assert content.strip(), (
        f"{topology}: generated content was empty (payload={payload!r})"
    )


@pytest.mark.parametrize("topology", TOPOLOGY_NAMES)
@pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)
def test_topology_served_model_name(topology: str) -> None:
    """Sanity-check the served model id via `/v1/models`.

    This guards against a wrong `--served-model-name` slipping into the
    compose file (D-05).
    """
    base_url = TOPOLOGY_BASE_URLS[topology]
    response = requests.get(f"{base_url}/models", timeout=10)
    assert response.status_code == 200, (
        f"{topology}: /v1/models returned {response.status_code}"
    )
    body = response.json()
    data = body.get("data")
    assert isinstance(data, list), (
        f"{topology}: /v1/models did not return `data` list: {body!r}"
    )
    ids = [entry.get("id") for entry in data if isinstance(entry, dict)]
    assert SERVED_MODEL_NAME in ids, (
        f"{topology}: expected served model {SERVED_MODEL_NAME!r} in "
        f"/v1/models response, got {ids!r}"
    )
