#!/usr/bin/env bash
# GoodputLab Phase 1 topology health gate (plan 01-06).
#
# Verifies that all four Phase 1 topologies serve the common endpoint
# contract (D-05) and that disaggregated topologies actually transfer
# KV blocks via real vLLM/NIXL metrics. The sentinel-token CLI run
# closes PITFALLS P1 (NIXL silent garbage output cannot be detected by
# counter increments alone; only a known-prefix first-token comparison
# on the decode side catches corruption).
#
# Usage:
#   scripts/health.sh colocated
#   scripts/health.sh chunked
#   scripts/health.sh disagg
#   scripts/health.sh disagg-tier
#   scripts/health.sh all
#
# Exit codes:
#   0  every check passed for the requested topology(ies)
#   1  any probe, sentinel run, or NIXL metric gate failed
#   64 usage error (bad topology argument)
#
# Port mapping is locked to docker-compose.yml:
#   colocated    -> 18000 (single-process vLLM, no KV transfer)
#   chunked      -> 18001 (single-process vLLM, chunked-prefill on)
#   disagg       -> 19100 (P=18000-pool + D=18000-pool via FastAPI proxy)
#   disagg-tier  -> 19200 (P/D + LMCache tiering via FastAPI proxy)

set -euo pipefail

# ---------------------------------------------------------------------------
# Topology → port mapping (port assignment is the health gate's hard
# contract with docker-compose.yml; changing this requires updating both).
# ---------------------------------------------------------------------------

PORT_COLOCATED=18000
PORT_CHUNKED=18001
PORT_DISAGG=19100
PORT_DISAGG_TIER=19200

SERVED_MODEL_NAME="goodputlab-model"
SENTINEL_CLI="tests/sentinel.py"
CURL_TIMEOUT_S=10
METRICS_TIMEOUT_S=15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_port_for() {
    case "$1" in
        colocated) echo "$PORT_COLOCATED" ;;
        chunked) echo "$PORT_CHUNKED" ;;
        disagg) echo "$PORT_DISAGG" ;;
        disagg-tier) echo "$PORT_DISAGG_TIER" ;;
        *) echo "" ;;
    esac
}

_is_disagg_topology() {
    case "$1" in
        disagg|disagg-tier) return 0 ;;
        *) return 1 ;;
    esac
}

_fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# Fetch a single Prometheus metric value (sum or count) from a /metrics
# endpoint. Returns "0" if the metric is absent. Empty/missing metrics
# on a non-disagg topology are skipped; on a disagg topology they fail.
_metric_value() {
    local url="$1"
    local metric_name="$2"
    local body
    body="$(curl -fs --max-time "$METRICS_TIMEOUT_S" "$url/metrics" || true)"
    if [[ -z "$body" ]]; then
        echo ""
        return
    fi
    # Match Prometheus text format: optional HELP/TYPE comment lines are
    # ignored; metric lines look like:
    #   metric_name{labels...} value
    # We extract the bare-name series only (no labels) to avoid double
    # counting histogram labels. If labelled series exist, take their
    # sum.
    awk -v target="$metric_name" '
        BEGIN { sum = 0; seen = 0 }
        /^#/ { next }
        {
            # split first field at the first {
            n = index($1, "{")
            if (n == 0) n = length($1) + 1
            name = substr($1, 1, n - 1)
            if (name == target) {
                seen = 1
                sum += $NF + 0
            }
        }
        END { if (seen) printf("%g", sum); else print "" }
    ' <<<"$body"
}

_diff_positive() {
    local before="$1"
    local after="$2"
    awk -v b="$before" -v a="$after" 'BEGIN { exit (a > b ? 0 : 1) }'
}

# ---------------------------------------------------------------------------
# Per-topology probes
# ---------------------------------------------------------------------------

_probe_common() {
    local topology="$1"
    local port="$2"
    local base="http://localhost:${port}"

    echo "[1/4] /health on ${topology} (port ${port})"
    if ! curl -fs --max-time "$CURL_TIMEOUT_S" "${base}/health" >/dev/null; then
        _fail "${topology}: /health did not return 2xx"
    fi

    echo "[2/4] /v1/models on ${topology} requires ${SERVED_MODEL_NAME}"
    local models_body
    if ! models_body="$(curl -fs --max-time "$CURL_TIMEOUT_S" "${base}/v1/models")"; then
        _fail "${topology}: /v1/models did not return 2xx"
    fi
    if ! python3 - "$models_body" <<'PY' >/dev/null 2>&1
import json, sys
body = json.loads(sys.argv[1])
data = body.get("data") or []
ids = {m.get("id") for m in data if isinstance(m, dict)}
assert "goodputlab-model" in ids, f"missing {SERVED_MODEL_NAME}; got {sorted(ids)}"
PY
    then
        _fail "${topology}: /v1/models missing required id ${SERVED_MODEL_NAME}"
    fi

    echo "[3/4] /metrics on ${topology} (scrape returns Prometheus text)"
    local metrics_body
    if ! metrics_body="$(curl -fs --max-time "$METRICS_TIMEOUT_S" "${base}/metrics")"; then
        _fail "${topology}: /metrics did not return 2xx"
    fi
    if [[ -z "$metrics_body" ]] || ! grep -q '^# HELP\|^# TYPE' <<<"$metrics_body"; then
        _fail "${topology}: /metrics response was empty or not Prometheus text"
    fi
}

_run_sentinel() {
    local topology="$1"
    local port="$2"
    local base="http://localhost:${port}/v1"
    echo "[4/4] sentinel-token validity on ${topology}"
    # `--mode check` reads tests/_fixtures/sentinel_*.json; absence = FAIL.
    # The sentinel CLI returns 0 on pass, non-zero on mismatch/P1 corruption.
    if ! python3 "$SENTINEL_CLI" --mode check \
            --base-url "$base" \
            --served-model-name "$SERVED_MODEL_NAME"; then
        _fail "${topology}: sentinel-token validity check failed (PITFALLS P1)"
    fi
}

# ---------------------------------------------------------------------------
# Disagg-only NIXL metric delta + zero-failure gates
# ---------------------------------------------------------------------------
#
# PITFALLS P1 mitigation: ``kv_transfer_complete_count`` is NOT a real
# vLLM/NIXL metric (pre-NIXL-connector proxy logs only). Health gate
# explicitly marks it as NOT-A-GATE here and asserts on the real
# vllm:nixl_* family below. Sentinel-token validity above is the
# load-bearing check; NIXL metric deltas are corroborating evidence.

check_nixl_disagg() {
    local topology="$1"
    local port="$2"
    local base="http://localhost:${port}"
    echo "[+] NIXL metric deltas on ${topology} (disagg NIXL/UCX transfer)"

    # --- counters BEFORE
    local before_xfers before_bytes before_fail_xfers before_fail_notes
    before_xfers="$(_metric_value "$base" vllm:nixl_xfer_time_seconds_count)"
    before_bytes="$(_metric_value "$base" vllm:nixl_bytes_transferred_sum)"
    before_fail_xfers="$(_metric_value "$base" vllm:nixl_num_failed_transfers_total)"
    before_fail_notes="$(_metric_value "$base" vllm:nixl_num_failed_notifications_total)"
    # On a fresh topology, these may be absent or 0. Treat absent as 0
    # for the "must increase" assertions, but require presence for the
    # failed-counter assertion below.
    before_xfers="${before_xfers:-0}"
    before_bytes="${before_bytes:-0}"
    before_fail_xfers="${before_fail_xfers:-0}"
    before_fail_notes="${before_fail_notes:-0}"

    # --- force one KV transfer via an explicit short completion
    # We hit the public /v1/completions endpoint, not the sentinel CLI,
    # because the sentinel CLI may short-circuit on warm-cache replay.
    # Greedy + 1 token forces prefill+decode through disagg.
    local probe_payload
    probe_payload='{"model":"goodputlab-model","prompt":"hello","max_tokens":1,"temperature":0.0,"stream":false}'
    if ! curl -fs --max-time 30 \
            -H 'content-type: application/json' \
            -d "$probe_payload" \
            "${base}/v1/completions" >/dev/null; then
        _fail "${topology}: P/D probe completion failed (KV transfer not exercised)"
    fi

    # --- counters AFTER
    local after_xfers after_bytes
    if ! after_xfers="$(_metric_value "$base" vllm:nixl_xfer_time_seconds_count)"; then
        _fail "${topology}: missing vllm:nixl_xfer_time_seconds_count metric"
    fi
    if ! after_bytes="$(_metric_value "$base" vllm:nixl_bytes_transferred_sum)"; then
        _fail "${topology}: missing vllm:nixl_bytes_transferred_sum metric"
    fi
    after_xfers="${after_xfers:-0}"
    after_bytes="${after_bytes:-0}"

    # --- delta assertions (positive change required)
    if ! _diff_positive "$before_xfers" "$after_xfers"; then
        _fail "${topology}: vllm:nixl_xfer_time_seconds_count did not increase (${before_xfers} -> ${after_xfers})"
    fi
    if ! _diff_positive "$before_bytes" "$after_bytes"; then
        _fail "${topology}: vllm:nixl_bytes_transferred_sum did not increase (${before_bytes} -> ${after_bytes})"
    fi

    # --- failure counters (must equal zero; presence required on disagg)
    local now_fail_xfers now_fail_notes
    now_fail_xfers="$(_metric_value "$base" vllm:nixl_num_failed_transfers_total)"
    now_fail_notes="$(_metric_value "$base" vllm:nixl_num_failed_notifications_total)"
    if [[ -z "$now_fail_xfers" ]]; then
        _fail "${topology}: vllm:nixl_num_failed_transfers_total metric missing on /metrics"
    fi
    if [[ -z "$now_fail_notes" ]]; then
        _fail "${topology}: vllm:nixl_num_failed_notifications_total metric missing on /metrics"
    fi
    if ! awk -v v="$now_fail_xfers" 'BEGIN { exit (v + 0 == 0 ? 0 : 1) }'; then
        _fail "${topology}: vllm:nixl_num_failed_transfers_total = ${now_fail_xfers} (must be 0)"
    fi
    if ! awk -v v="$now_fail_notes" 'BEGIN { exit (v + 0 == 0 ? 0 : 1) }'; then
        _fail "${topology}: vllm:nixl_num_failed_notifications_total = ${now_fail_notes} (must be 0)"
    fi
}

# ---------------------------------------------------------------------------
# Per-topology driver
# ---------------------------------------------------------------------------

run_one() {
    local topology="$1"
    local port
    port="$(_port_for "$topology")"
    if [[ -z "$port" ]]; then
        _fail "unknown topology: ${topology}"
    fi

    echo "=== ${topology} (port ${port}) ==="
    _probe_common "$topology" "$port"
    if _is_disagg_topology "$topology"; then
        check_nixl_disagg "$topology" "$port"
    fi
    _run_sentinel "$topology" "$port"
    echo "[OK] ${topology} healthy"
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

main() {
    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 <colocated|chunked|disagg|disagg-tier|all>" >&2
        exit 64
    fi
    local target="$1"
    case "$target" in
        all)
            for topo in colocated chunked disagg disagg-tier; do
                run_one "$topo"
            done
            echo "[OK] all topologies healthy"
            ;;
        colocated|chunked|disagg|disagg-tier)
            run_one "$target"
            ;;
        *)
            echo "Usage: $0 <colocated|chunked|disagg|disagg-tier|all>" >&2
            exit 64
            ;;
    esac
}

main "$@"
