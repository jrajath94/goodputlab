# P:D autoscaler tuning

This document describes how to tune `control/autoscaler.py` for the
GoodputLab pool fleet. It is the v0.1 reference; live-cluster validation
is deferred to v1.1 per `CHANGELOG.md` §0.1 (GPU budget cap).

The autoscaler is a discrete-time PID per pool with integer-replica
adjustments and a drain protocol. The math, the anti-windup rules, and
the current "what is enforced vs planned" boundary are all below.

## Topology

```text
              ┌──────────────────────────┐
   queue_depth│     PoolAutoscaler       │decisions (delta, reason)
   in_flight  │   tick(every 1-5 s)      │─────────────────────► orchestrator
              └──────────────────────────┘
                       │
                       ▼
              per-pool PidController
              (Kp · e + Ki · ∫e + Kd · ė)
              with anti-windup + clamp
```

One controller per `Pool` enum value (`PREFILL`, `DECODE`, `COLOCATED`,
`TIER`). Each tick produces one `AutoscalerDecision` per pool.

## Tick loop

```text
for each pool in topology:
    error    = queue_depth - target_queue_depth
    raw      = pid.update(error, dt=1.0)             # clamped to [out_min, out_max]
    delta    = sign-of-PID × step_size               # integer in [-step, +step]
    delta    = clamp(replicas + delta, min, max) - replicas

    if delta < 0 and in_flight[pool] > 0:
        emit (delta=0, reason="drain_wait")           # never scale down with live work
        continue

    emit (delta, reason="queue_high"|"queue_low"|"stable", drained=(delta<0))
```

## Enforced (v0.1)

The following are unit-tested in `tests/test_autoscaler.py` and are
guaranteed by the code as written.

### Drain protocol

A scale-down decision is **only emitted when `in_flight == 0`** for that
pool. While `in_flight > 0`, the autoscaler returns
`delta=0, reason="drain_wait"` regardless of how empty the queue is.

This is the load-bearing guarantee: a worker that still owns in-flight
requests is never flipped to a new role. The property test
`test_drain_blocks_scale_down_under_sustained_inflight` runs 50 random
ticks with `in_flight ∈ [1, 8]` and asserts `delta >= 0` on every one.

### Output clamping + anti-windup

When the raw PID output saturates at `output_min` or `output_max`, the
integrator stops accumulating. Without anti-windup, a long-running
positive error with a low output clamp would lock the controller into
saturation forever ("integrator windup"). See `control/pid.py:update`.

### Floor / ceiling on replicas

`min_replicas` and `max_replicas` bound the integer output. The
`step_size` parameter (default 1) bounds the per-tick delta so a single
PID excursion cannot double the fleet.

### Stable reason when at target

When `queue_depth == target_queue_depth` and `in_flight == 0`, the
controller returns `delta=0, reason="stable"`.

## Planned, not yet enforced (v1.1)

The README and the CHANGELOG describe a **120 s minimum dwell between
flips** and a **1-5 s tick interval**. Neither is currently enforced by
the controller — the README text is forward-looking.

| Item                          | Status     | Where it should land                  |
|-------------------------------|------------|---------------------------------------|
| 120 s minimum dwell           | not yet    | `PoolAutoscaler._last_flip_ts[pool]`  |
| 1-5 s tick interval           | convention | orchestrator-side scheduler           |
| Multi-pool cross-coupling     | not yet    | prefill/decode imbalance detection    |
| Workload-shift scenario bench | not yet    | live GPU; v1.1                        |

Without minimum dwell, the controller can theoretically flip a worker
on consecutive ticks if the PID output stays saturated. In practice the
PID integral decay and the replica floor/ceiling bound this, but the
formal guarantee does not exist yet. Live tuning in v1.1 will measure
oscillation frequency and pick the dwell.

## Default gains

```python
PidGains(kp=1.0, ki=0.0, kd=0.0)
output_min=-10.0
output_max=+10.0
step_size=1
```

These are conservative defaults:

- **`ki=0`** — no integral action. The drain protocol already prevents
  the "queue stays high, scale-up races forever" failure mode that
  integral action normally guards against. Pure proportional keeps the
  controller interpretable: `error → scale-up by ⌊error / target⌋` over
  the next few ticks.
- **`kd=0`** — derivative amplifies measurement noise in `queue_depth`,
  which is a coarse scrape of `vllm:num_requests_running`. Add `kd`
  only after wiring a low-pass filter on the input.
- **`kp=1.0`** — with `output_max=10`, this gives a 1:1 error-to-output
  mapping up to 10. A queue depth 30 above target on a 4-replica pool
  yields `delta = +1` per tick until the floor/ceiling is hit.

## Tuning procedure (v1.1)

1. Pick a representative workload mix from
   `bench/results/real/<topology>.json` (chat / RAG / agentic).
2. Drive that mix into a 2×P + 2×D cluster at 2× sustained rate for
   10 minutes.
3. Scrape `goodputlab_queue_depth`, `goodputlab_in_flight`,
   `goodputlab_replicas` per second into a CSV.
4. Sweep `kp ∈ {0.5, 1.0, 2.0, 4.0}` with `ki=0`, `kd=0`.
5. For each `kp`, measure:
   - **Oscillation frequency** (replicas-direction flips per minute).
   - **Steady-state error** (mean |depth - target|).
   - **Settling time** after a 50% rate spike.
6. Pick `kp` that minimises a weighted sum of the three, with
   oscillation weighted heavily. Lock gains in
   `configs/autoscaler.json` and emit a per-pool `goodput_attained`
   panel.

## Property tests already shipped

- `test_drain_blocks_scale_down_under_sustained_inflight` — 50-tick
  random walk, no scale-down with in_flight>0.
- `test_drain_fires_immediately_when_inflight_drops_to_zero` — drain
  fires the first tick in_flight hits 0.
- `test_autoscaler_caps_at_max_replicas` — clamp upper.
- `test_autoscaler_floor_at_min_replicas` — clamp lower.

These together assert the v0.1 invariants. v1.1 will add
`test_min_dwell_between_flips` once the dwell timestamp is wired in.

## Failure modes the autoscaler must NOT introduce

- **Flip a worker mid-request.** Covered by drain protocol.
- **Double the fleet in one tick.** Covered by `step_size`.
- **Flip a worker back and forth each tick.** Will be covered by
  minimum dwell (v1.1).
- **Integrator windup after a long saturation.** Covered by
  anti-windup in `control/pid.py`.
- **Scale down a healthy pool to zero.** Covered by `min_replicas >= 1`
  default and the floor test.

## References

- `control/autoscaler.py` — implementation, 122 lines.
- `control/pid.py` — discrete-time PID with anti-windup.
- `control/pool.py` — `Pool` enum and `PoolState` model.
- `tests/test_autoscaler.py` — 15 tests, including 2 drain property
  tests.
- `CHANGELOG.md` §0.1 — release-scope policy and the v1.1 deferral.