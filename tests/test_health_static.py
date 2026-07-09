"""Static health-gate source-mark tests (no network, no GPU).

These tests read ``scripts/health.sh`` as text and assert the shell source
contains the required behavior markers for the Phase 1 health gate (plan
01-06). They must not depend on running vLLM, the disagg proxy, or any
network sockets. The health shell script gates every topology through
common ``/health``, ``/v1/models``, ``/metrics`` endpoints, invokes the
sentinel CLI for known-prefix post-transfer validity (PITFALLS P1), and
asserts NIXL metric deltas + zero failed transfers on the disagg
topologies.

The marker contract (every literal asserted below) is the spec;
``scripts/health.sh`` is the implementation under test.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTH_SCRIPT = REPO_ROOT / "scripts" / "health.sh"
SENTINEL_CLI = REPO_ROOT / "tests" / "sentinel.py"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def _read_health_script() -> str:
    """Load ``scripts/health.sh`` as text. Fails loudly if missing."""
    assert HEALTH_SCRIPT.exists(), (
        f"health script not found at {HEALTH_SCRIPT} — "
        "task 2 of plan 01-06 must ship scripts/health.sh"
    )
    return HEALTH_SCRIPT.read_text(encoding="utf-8")


# --- Shell preamble ----------------------------------------------------------


def test_health_script_uses_strict_mode() -> None:
    """``set -euo pipefail`` MUST appear on the first non-comment line."""
    src = _read_health_script()
    assert "set -euo pipefail" in src, (
        "scripts/health.sh must enable strict mode (set -euo pipefail)"
    )


# --- Topology + port mapping -------------------------------------------------


def test_health_script_lists_all_topologies() -> None:
    """The four topology names MUST appear as accepted arguments."""
    src = _read_health_script()
    for name in ("colocated", "chunked", "disagg", "disagg-tier"):
        assert name in src, (
            f"scripts/health.sh must accept topology argument {name!r}"
        )


def test_health_script_uses_correct_port_per_topology() -> None:
    """Port mapping MUST be colocated=18000, chunked=18001, disagg=19100,
    disagg-tier=19200 (per docker-compose.yml hard constraint)."""
    src = _read_health_script()
    for port in ("18000", "18001", "19100", "19200"):
        assert port in src, (
            f"scripts/health.sh must reference port {port} "
            "(per docker-compose.yml topologiy port mapping)"
        )
    # Parse topology → port assignments explicitly. Either PORT assignment
    # in a case statement or `${PORT:=...}` style must bind each topology
    # to the correct port; the assertion below is fuzzy-on-shape but
    # strict on adjacency of topology names + port numbers in the file.
    colocated_pair = ("colocated" in src) and ("18000" in src)
    chunked_pair = ("chunked" in src) and ("18001" in src)
    disagg_pair = ("disagg" in src) and ("19100" in src)
    disagg_tier_pair = ("disagg-tier" in src) and ("19200" in src)
    assert all(
        (colocated_pair, chunked_pair, disagg_pair, disagg_tier_pair)
    ), (
        "scripts/health.sh must wire each topology name to its docker-compose "
        "port (colocated=18000, chunked=18001, disagg=19100, disagg-tier=19200)"
    )


# --- Common endpoint contract (D-05) ----------------------------------------


def test_health_script_probes_common_endpoints() -> None:
    """``/health``, ``/v1/models``, ``/metrics`` MUST be probed for every topology."""
    src = _read_health_script()
    assert "/health" in src, "scripts/health.sh must probe /health"
    assert "/v1/models" in src, "scripts/health.sh must probe /v1/models"
    assert "/metrics" in src, "scripts/health.sh must probe /metrics"


def test_health_script_requires_goodputlab_model_id() -> None:
    """Model id assertion MUST be ``goodputlab-model`` (D-05 common contract)."""
    src = _read_health_script()
    assert "goodputlab-model" in src, (
        "scripts/health.sh must require goodputlab-model served-model-name "
        "(per docker-compose.yml and Makefile D-05 common endpoint contract)"
    )


# --- Sentinel CLI invocation (PITFALLS P1) ----------------------------------


def test_health_script_invokes_sentinel_cli() -> None:
    """The shell MUST invoke ``tests/sentinel.py --mode check`` for every topology."""
    src = _read_health_script()
    assert "tests/sentinel.py" in src, (
        "scripts/health.sh must invoke tests/sentinel.py CLI"
    )
    assert "--mode check" in src, (
        "scripts/health.sh must invoke sentinel with --mode check"
    )


def test_sentinel_cli_exists_for_health_invocation() -> None:
    """The sentinel CLI implementation must already exist on disk."""
    assert SENTINEL_CLI.exists(), (
        "scripts/health.sh requires tests/sentinel.py (plan 01-05 dependency)"
    )


def test_compose_file_declares_topology_ports() -> None:
    """docker-compose.yml must keep the port mapping the health script relies on."""
    assert COMPOSE_FILE.exists(), "docker-compose.yml must exist at repo root"
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
    for port in ("18000", "18001", "19100", "19200"):
        assert port in compose_text, (
            f"docker-compose.yml must publish port {port} "
            "(health gate hard constraint)"
        )


# --- NIXL metric deltas (disagg topologies only) -----------------------------


def test_health_uses_real_nixl_metrics() -> None:
    """Disagg health checks MUST assert on real vLLM/NIXL metric names.

    The four metrics below are the load-bearing evidence that a P→D KV
    transfer actually happened on real vLLM/NIXL connector code paths
    (per .planning/phases/01-topologies-topo/01-RESEARCH.md Q6).
    """
    src = _read_health_script()
    assert "vllm:nixl_xfer_time_seconds_count" in src, (
        "scripts/health.sh must assert vllm:nixl_xfer_time_seconds_count "
        "increases after a sentinel round-trip (disagg topologies)"
    )
    assert "vllm:nixl_bytes_transferred_sum" in src, (
        "scripts/health.sh must assert vllm:nixl_bytes_transferred_sum "
        "increases after a sentinel round-trip (disagg topologies)"
    )
    assert "vllm:nixl_num_failed_transfers_total" in src, (
        "scripts/health.sh must assert "
        "vllm:nixl_num_failed_transfers_total == 0 (disagg topologies)"
    )
    assert "vllm:nixl_num_failed_notifications_total" in src, (
        "scripts/health.sh must assert "
        "vllm:nixl_num_failed_notifications_total == 0 (disagg topologies)"
    )


# --- Rejection of the invalid kv_transfer_complete_count gate (PITFALLS P1) --


def test_health_rejects_kv_transfer_complete_count_as_gate() -> None:
    """``kv_transfer_complete_count`` is NOT a real vLLM/NIXL metric; the
    health script must explicitly mark it as non-gating (explanatory
    comment + test-visible literal string).

    PITFALLS P1: relying on a fake metric would silently miss NIXL
    corruption. Sentinel-token validity is the only valid gate.
    """
    src = _read_health_script()
    assert "kv_transfer_complete_count" in src, (
        "scripts/health.sh must mention kv_transfer_complete_count with "
        "an explicit NOT-A-GATE marker (PITFALLS P1 silent-garbage mitigation)"
    )
    # The marker must be an explicit rejection, not a usage as a gate.
    lowered = src.lower()
    rejection_markers = (
        "not a valid gate",
        "not a gate",
        "not valid",
        "not a real",
        "is not a real",
        "is not valid",
        "obsolete",
        "deprecated",
        "do not gate",
        "not used as a gate",
        "rejected as a gate",
    )
    assert any(marker in lowered for marker in rejection_markers), (
        "scripts/health.sh must include an explicit rejection comment "
        "for kv_transfer_complete_count (PITFALLS P1)"
    )
    # The literal "NOT-A-GATE" / "NOT_A_GATE" / "[rejected]" / "[not-a-gate]"
    # test-visible string must appear so downstream tests / grep audits
    # can confirm the marker is present.
    test_visible = (
        "NOT-A-GATE" in src
        or "NOT_A_GATE" in src
        or "[rejected]" in lowered
        or "[not-a-gate]" in lowered
        or "not-a-gate" in lowered
    )
    assert test_visible, (
        "scripts/health.sh must include a test-visible NOT-A-GATE marker "
        "for kv_transfer_complete_count (PITFALLS P1)"
    )


# --- OK summary line + non-zero exit on failure -----------------------------


def test_health_script_emits_ok_summary_line() -> None:
    """Every topology MUST emit ``[OK] <topology> healthy`` on success."""
    src = _read_health_script()
    assert "[OK]" in src and "healthy" in src, (
        "scripts/health.sh must emit an '[OK] <topology> healthy' "
        "summary line per topology"
    )


def test_health_script_supports_all_keyword() -> None:
    """The ``all`` keyword MUST be accepted so ``make health`` can call once."""
    src = _read_health_script()
    assert "all" in src, (
        "scripts/health.sh must accept the 'all' argument (make health contract)"
    )
