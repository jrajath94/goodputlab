"""Tests for scripts/real_bench.py — the live-vLLM bench runner.

We can't start vLLM in CI (no GPU), so we test the *contract*:
- trace determinism
- router registration per topology
- CLI argument surface
- base-url timeout behavior
- summary.json shape
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest  # noqa: F401  # kept for consistency with other test files

# Make scripts/ importable.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from real_bench import _router, _trace, _wait_for_vllm  # noqa: E402

from control.pool import Pool  # noqa: E402


def test_trace_has_deterministic_seeded_arrivals() -> None:
    a = _trace(n=5, rate=4.0, prompt_tokens=32)
    b = _trace(n=5, rate=4.0, prompt_tokens=32)
    a_ts = list(a.requests)
    b_ts = list(b.requests)
    assert [r.request_id for r in a_ts] == [r.request_id for r in b_ts]
    assert len(a_ts) == 5


def test_router_registers_all_pools_for_every_topology() -> None:
    from bench.orchestrator import Topology

    for topo in Topology:
        r = _router(topo)
        states = r.pool_states()
        pools = {ps.pool for ps in states.values()}
        assert Pool.PREFILL in pools, topo
        assert Pool.DECODE in pools, topo


def test_cli_help_exits_zero() -> None:
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.real_bench", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert out.returncode == 0
    assert "--base-url" in out.stdout
    assert "--model" in out.stdout
    assert "--out" in out.stdout


def test_wait_for_vllm_returns_false_on_unreachable() -> None:
    """Negative test: bogus URL must time out cleanly."""
    ok = asyncio.run(_wait_for_vllm("http://127.0.0.1:1/v1", timeout_s=4))
    assert ok is False


def test_summary_json_shape(tmp_path: Path) -> None:
    """Verify the summary.json the runner would write matches schema."""
    expected_keys = {
        "n_topologies",
        "base_url",
        "model",
        "all_reconcile",
        "topologies",
    }
    sample = {
        "n_topologies": 4,
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "all_reconcile": True,
        "topologies": ["colocated", "chunked", "disagg", "disagg_tier"],
    }
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(sample, indent=2))
    loaded = json.loads(out.read_text())
    assert expected_keys.issubset(loaded.keys())
    assert loaded["n_topologies"] == 4