"""Pool autoscaler — per-pool PID + integer-replica adjustments + drain + min-dwell.

Per tick, for each pool:

1. Read ``queue_depth`` and compute error = depth - target.
2. Run PID; map the output to a signed integer delta in [-step_size, +step_size].
3. Clamp to ``[min_replicas, max_replicas]``.
4. If scale-down would apply but ``in_flight > 0``, return ``delta=0``
   with reason="drain_wait" instead.  Once ``in_flight == 0``, scale-down
   fires and ``drained=True`` is recorded.
5. If a flip (delta != 0) is being emitted but the pool has flipped within
   ``min_dwell_s`` seconds, suppress it and return ``delta=0`` with
   reason="dwell_wait" instead.  This prevents ping-pong oscillation when
   the PID error alternates sign across ticks.

The ``min_dwell_s=0`` default disables the feature (back-compat for the
existing test suite).  Set to 120.0 (or tune via TUNING.md) for the live
controller.  The check fires *after* the drain check, so drain is always
the stricter constraint.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from control.pid import PidController
from control.pool import Pool


class PoolTopology(BaseModel):
    """Per-pool replica count + target queue depth."""

    model_config = ConfigDict(extra="forbid")

    pool: Pool
    replicas: int = Field(ge=0)
    target_queue_depth: int = Field(default=16, ge=0)


class AutoscalerDecision(BaseModel):
    """One autoscaler step for one pool."""

    model_config = ConfigDict(extra="forbid")

    pool: Pool
    delta: int
    reason: str
    drained: bool = False


# Default clock: monotonic.  Tests inject a fake.


class PoolAutoscaler:
    """One PID per pool; integer step size; bounded [min, max] replicas.

    ``min_dwell_s`` enforces a minimum wall-clock interval between flips on
    the same pool.  ``clock`` is a zero-arg callable returning a float
    timestamp; defaults to ``time.monotonic``.  Both are tunable for
    deterministic tests.
    """

    def __init__(
        self,
        controllers: dict[Pool, PidController],
        min_replicas: int = 1,
        max_replicas: int = 8,
        step_size: int = 1,
        min_dwell_s: float = 0.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if min_replicas < 0:
            raise ValueError("min_replicas must be >= 0")
        if max_replicas < min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if min_dwell_s < 0:
            raise ValueError("min_dwell_s must be >= 0")
        self._controllers = controllers
        self._min = min_replicas
        self._max = max_replicas
        self._step = step_size
        self._min_dwell_s = min_dwell_s
        self._clock = clock if clock is not None else time.monotonic
        # Per-pool last flip timestamp (None = never flipped).
        self._last_flip_ts: dict[Pool, float] = {}

    def tick(
        self,
        topology: dict[Pool, PoolTopology],
        queue_depths: dict[Pool, int],
        in_flight: dict[Pool, int],
        dt: float = 1.0,
    ) -> list[AutoscalerDecision]:
        """One tick: returns one decision per pool."""
        out: list[AutoscalerDecision] = []
        now = self._clock()
        for pool, topo in topology.items():
            controller = self._controllers.get(pool)
            if controller is None:
                out.append(AutoscalerDecision(pool=pool, delta=0, reason="no_controller"))
                continue

            depth = queue_depths.get(pool, 0)
            error = float(depth - topo.target_queue_depth)
            raw = controller.update(error, dt=dt)
            # Map raw PID output to a signed int in [-step, +step].
            # Output range comes from controller's clamp.
            span = controller.output_max - controller.output_min
            if span <= 0:
                delta = 0
            else:
                # Normalize to [-1, +1] then scale by step.
                normalized = (raw - controller.output_min) / span * 2.0 - 1.0
                delta = int(round(normalized * self._step))

            # Floor / ceiling.
            would_be = topo.replicas + delta
            if would_be > self._max:
                delta = self._max - topo.replicas
            if would_be < self._min:
                delta = self._min - topo.replicas

            # Drain protocol: scale-down only when in_flight == 0.
            if delta < 0 and in_flight.get(pool, 0) > 0:
                out.append(AutoscalerDecision(pool=pool, delta=0, reason="drain_wait"))
                continue

            # Min-dwell: any non-zero flip within min_dwell_s of the previous
            # flip is suppressed.  Stable ticks (delta==0) do NOT start a
            # dwell window, so a long steady state can flip freely later.
            if delta != 0 and self._min_dwell_s > 0:
                last_ts = self._last_flip_ts.get(pool)
                if last_ts is not None and (now - last_ts) < self._min_dwell_s:
                    out.append(AutoscalerDecision(pool=pool, delta=0, reason="dwell_wait"))
                    continue

            if delta > 0:
                reason = "queue_high"
            elif delta < 0:
                reason = "queue_low"
            else:
                reason = "stable"

            if delta != 0:
                self._last_flip_ts[pool] = now

            out.append(
                AutoscalerDecision(
                    pool=pool,
                    delta=delta,
                    reason=reason,
                    drained=(delta < 0),
                )
            )
        return out


__all__ = ["AutoscalerDecision", "PoolAutoscaler", "PoolTopology"]
