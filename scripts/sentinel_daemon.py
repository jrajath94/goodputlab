"""Continuous sentinel drift probe — emits Prometheus gauge `sentinel_drift`.

CONTEXT D-03 Layer 3: every 60 seconds, run the standalone sentinel check
against the configured OpenAI-compatible base URL and reflect the result as a
Prometheus gauge `sentinel_drift` (0 = pass, 1 = drift/failure/timeout).

We invoke `tests/sentinel.py --mode check` as a subprocess so the daemon
shares the exact check semantics with the standalone CLI and the
health-check integration in scripts/health.sh (single source of truth for
the comparison logic).

The daemon NEVER logs prompts, generated tokens, or logprobs. Only the
timestamp, base URL, served model name, interval, and pass/fail status go to
stderr. This avoids leaking any input that could hint at model state.

CLI:
    python3 -m scripts.sentinel_daemon \\
        --base-url http://localhost:19100/v1 \\
        --served-model-name goodputlab-model \\
        --interval-seconds 60 \\
        --metrics-port 9108
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SENTINEL_CLI = REPO_ROOT / "tests" / "sentinel.py"
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_METRICS_PORT = 9108
DEFAULT_TIMEOUT_SECONDS = 120.0

# Exactly one gauge, named to match CONTEXT D-03 / plan acceptance criteria.
SENTINEL_DRIFT_GAUGE = Gauge(
    "sentinel_drift",
    "1 when the most recent sentinel check failed (token mismatch, logprob drift, "
    "HTTP error, timeout, or missing fixture); 0 when it passed.",
)


def _run_sentinel_check(
    base_url: str,
    served_model_name: str,
    fixture_dir: str,
    timeout: float,
) -> int:
    """Invoke tests/sentinel.py --mode check. Returns the process exit code."""
    cmd = [
        sys.executable,
        str(DEFAULT_SENTINEL_CLI),
        "--mode",
        "check",
        "--base-url",
        base_url,
        "--served-model-name",
        served_model_name,
        "--fixture-dir",
        fixture_dir,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[sentinel_daemon] timeout after {timeout:.0f}s waiting for sentinel check",
            file=sys.stderr,
        )
        return 124  # conventional `timeout` exit code
    if completed.returncode != 0:
        # Sentinel writes a one-line FAIL marker to stderr; we summarize only.
        tail = (completed.stderr or "").strip().splitlines()
        first = tail[0] if tail else "(no stderr)"
        msg = f"[sentinel_daemon] check failed (rc={completed.returncode}): {first}"
        print(msg, file=sys.stderr)
    return completed.returncode


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentinel_daemon",
        description="Continuous sentinel drift probe (Prometheus gauge sentinel_drift).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:19100/v1",
        help="OpenAI-compatible base URL to probe",
    )
    parser.add_argument(
        "--served-model-name",
        default="goodputlab-model",
        help="served model name as registered with vLLM (default: goodputlab-model)",
    )
    parser.add_argument(
        "--fixture-dir",
        default="tests/_fixtures",
        help="fixture directory passed to tests/sentinel.py (default: tests/_fixtures)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"interval between probes in seconds (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=DEFAULT_METRICS_PORT,
        help=f"Prometheus /metrics port (default: {DEFAULT_METRICS_PORT})",
    )
    parser.add_argument(
        "--check-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-probe subprocess timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval_seconds < 1:
        print("ERROR: --interval-seconds must be >= 1", file=sys.stderr)
        return 2

    start_http_server(args.metrics_port)
    print(
        f"[sentinel_daemon] started base_url={args.base_url} model={args.served_model_name} "
        f"interval={args.interval_seconds}s metrics_port={args.metrics_port}",
        file=sys.stderr,
    )

    # Graceful shutdown — SIGTERM (systemd / k8s) and SIGINT (Ctrl-C).
    stop_requested = {"flag": False}

    def _request_stop(signum: int, _frame: object) -> None:
        print(f"[sentinel_daemon] received signal {signum}, shutting down", file=sys.stderr)
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    while not stop_requested["flag"]:
        rc = _run_sentinel_check(
            base_url=args.base_url,
            served_model_name=args.served_model_name,
            fixture_dir=args.fixture_dir,
            timeout=args.check_timeout,
        )
        SENTINEL_DRIFT_GAUGE.set(1 if rc != 0 else 0)
        status = "PASS" if rc == 0 else "FAIL"
        print(
            f"[sentinel_daemon] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"base_url={args.base_url} model={args.served_model_name} {status}",
            file=sys.stderr,
        )
        # Sleep in 1-second slices so SIGTERM is honored promptly.
        for _ in range(args.interval_seconds):
            if stop_requested["flag"]:
                break
            time.sleep(1)

    print("[sentinel_daemon] stopped cleanly", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())