"""Open-loop arrival processes for GoodputLab load generation.

LOAD-04: Poisson (exponential inter-arrival) and ON/OFF (two-state Markov)
processes that yield timestamps instead of blocking on a workload.
LOAD-07: every random draw routes through ``random.Random(seed)`` so
``PoissonArrival(seed=42).sample(1000)`` is bit-identical across runs
on the same cPython version.

We deliberately avoid ``numpy.random`` because its default bit generator
(PCG64) has changed between numpy minor versions in the past; the
load generator's replay contract must hold across deployments.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator

from core.trace import RequestSpec, Trace


class PoissonArrival:
    """Exponential inter-arrival process; deterministic from (rate, seed, n)."""

    def __init__(self, rate_per_sec: float, seed: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0, got {rate_per_sec}")
        self._rate = rate_per_sec
        self._rng = random.Random(seed)

    @property
    def rate(self) -> float:
        return self._rate

    def sample(self, n: int) -> list[float]:
        """Return ``n`` arrival offsets in seconds from t=0, strictly increasing.

        Uses ``random.Random.expovariate`` which is documented stable in
        cPython — no numpy dependency, no version drift on the bit pattern.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return []
        offsets: list[float] = []
        t = 0.0
        for _ in range(n):
            t += self._rng.expovariate(self._rate)
            offsets.append(t)
        return offsets


class OnOffArrival:
    """Two-state Markov arrival process.

    During the ON phase, arrivals are drawn from a Poisson(rate=on_rate)
    process.  During the OFF phase, no arrivals are emitted.  Cycle
    lengths are deterministic: ON for ``on_dur`` seconds, OFF for
    ``off_dur`` seconds, repeat until ``duration_s`` is exhausted.
    """

    def __init__(
        self,
        on_rate: float,
        on_dur: float,
        off_dur: float,
        seed: int,
    ) -> None:
        if on_rate <= 0:
            raise ValueError(f"on_rate must be > 0, got {on_rate}")
        if on_dur <= 0 or off_dur <= 0:
            raise ValueError(f"on_dur and off_dur must be > 0, got {on_dur}/{off_dur}")
        self._on_rate = on_rate
        self._on_dur = on_dur
        self._off_dur = off_dur
        self._rng = random.Random(seed)

    def sample(self, duration_s: float) -> list[float]:
        """Return arrival offsets in [0, duration_s], strictly increasing.

        Advances the RNG in the same order on every call, so two calls
        with the same (on_rate, on_dur, off_dur, seed, duration_s)
        produce bit-identical float lists — LOAD-07 replay contract.
        """
        if duration_s <= 0:
            return []
        cycle = self._on_dur + self._off_dur
        n_cycles = int(duration_s // cycle) + 1
        offsets: list[float] = []
        t = 0.0
        cycle_idx = 0
        while t < duration_s:
            cycle_start = cycle_idx * cycle
            on_end = cycle_start + self._on_dur
            if on_end > duration_s:
                on_end = duration_s
            # Poisson draws within this ON phase.
            phase_t = cycle_start
            while phase_t < on_end:
                phase_t += self._rng.expovariate(self._on_rate)
                if phase_t < on_end:
                    offsets.append(phase_t)
            t = cycle_start + cycle
            cycle_idx += 1
            if cycle_idx > n_cycles + 1:  # safety stop
                break
        return offsets


class OpenLoopScheduler:
    """Yields ``(RequestSpec, arrival_ts_ns)`` pairs in time order.

    Maps arrival offsets from the trace's arrival process onto the
    generator's request specs.  The mapping is index-based: the i-th
    arrival in the schedule pairs with ``trace.requests[i]`` (extra
    arrivals are clamped; missing arrivals mean fewer requests were
    scheduled than the trace defined).

    This is open-loop: it never blocks waiting for downstream capacity.
    If a downstream call is in flight, the next arrival simply queues
    at the caller — that's the whole point of open-loop load testing.
    """

    def __init__(
        self,
        trace: Trace,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._trace = trace
        self._clock_ns: Callable[[], int] = clock_ns or time.perf_counter_ns
        if trace.arrival.process == "poisson":
            proc: PoissonArrival | OnOffArrival = PoissonArrival(
                rate_per_sec=trace.arrival.rate_per_sec,
                seed=trace.arrival.seed,
            )
            self._offsets = proc.sample(trace.n_requests())
        else:
            assert trace.arrival.on_duration_s is not None
            assert trace.arrival.off_duration_s is not None
            proc = OnOffArrival(
                on_rate=trace.arrival.rate_per_sec,
                on_dur=trace.arrival.on_duration_s,
                off_dur=trace.arrival.off_duration_s,
                seed=trace.arrival.seed,
            )
            # ON/OFF: sample the whole window, then assign to requests in order.
            self._offsets = proc.sample(trace.duration_s)
        self._t0_ns: int | None = None  # set on first __next__

    def __iter__(self) -> Iterator[tuple[RequestSpec, int]]:
        self._t0_ns = self._clock_ns()
        for i, offset_s in enumerate(self._offsets):
            if i >= len(self._trace.requests):
                break
            arrival_ns = self._t0_ns + int(offset_s * 1_000_000_000)
            yield self._trace.requests[i], arrival_ns

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def n_scheduled(self) -> int:
        return min(len(self._offsets), len(self._trace.requests))


__all__ = ["OnOffArrival", "OpenLoopScheduler", "PoissonArrival"]
