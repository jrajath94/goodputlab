# Autoscaler live workload-shift — 2026-07-17 — partial evidence

`scripts/autoscaler_live.py` drove `PoolAutoscaler` from a **live** vLLM
queue signal (`vllm:num_requests_waiting` / `num_requests_running`)
through a prompt-heavy → decode-heavy shift on the v1.2 pod
(1× H100 of `u1n8efij0owar2`, Qwen2.5-7B, 20480 ctx).

## What was measured (run.jsonl, 43 ticks × 2 s)

| Check | Result |
|---|---|
| AUTO-05 drain invariant (no scale-down with in-flight > 0), asserted every tick against live counters | **0 violations** |
| Thrash (flips within 240 s window) | **0 flips** |
| Dwell gate | never triggered (no flips to suppress) |
| Max `num_requests_waiting` observed | 0 |
| Max `num_requests_running` observed | 3 |

## What was NOT measured (honest)

The **scale-up path never engaged live** because the waiting queue never
built:

1. First attempt: all requests shared one prompt — vLLM's automatic
   prefix caching absorbed the entire prefill. Fixed (unique prefix per
   request; pinned in the harness docstring).
2. Second attempt (cache-busted, 24 rps × ~8K-token prompts): running
   never exceeded 3 — consistent with requests being throttled upstream
   of the engine (RunPod HTTP proxy), not with H100 absorption. The
   harness now records per-status response counts to make this failure
   mode visible.
3. The on-pod localhost rerun (which removes the proxy from the path)
   was interrupted: the pod was stopped externally before the run
   completed.

So: PID/drain/dwell logic is validated against **live queue telemetry**
(not simulated), and the drain invariant held under real traffic — but
"queue builds ⇒ controller scales up" exists only as unit-test evidence
(`tests/test_autoscaler.py`), not as a live trace. Closing that needs
one on-pod localhost run of this harness (~5 pod-minutes, <$0.30);
the command is in the module docstring.

Replica actuation is virtual (single server); the module docstring
states this. Real multi-replica actuation is v1.1+ scope.
