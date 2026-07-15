"""Typed Prometheus metrics for the GoodputLab control plane (OBS-01).

Each ``MetricsRegistry`` owns its own ``CollectorRegistry`` so multiple
benchmarks or instances can coexist without state bleed.  Recording
methods are typed (no string metric names at call sites) so a typo in
a caller shows up as an AttributeError, not a silently-absent series.

Metric inventory:

| Metric                              | Type      | Labels         |
|-------------------------------------|-----------|----------------|
| goodputlab_ttft_ms                  | Histogram | slo_class      |
| goodputlab_itl_ms                   | Histogram | slo_class      |
| goodputlab_queue_depth              | Gauge     | pool           |
| goodputlab_kv_tier_hit_total        | Counter   | tier           |
| goodputlab_kv_tier_miss_total       | Counter   | tier           |
| goodputlab_spec_acceptance_total    | Counter   | —              |
| goodputlab_spec_proposed_total      | Counter   | —              |
| goodputlab_role_flip_total          | Counter   | from, to       |
| goodputlab_request_total            | Counter   | status_code    |
| goodputlab_request_error_total      | Counter   | error_class    |
| goodputlab_reconcile_gate_passes_total | Counter | —              |
| goodputlab_cache_no_history_total       | Counter | —              |
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

# Default latency buckets cover 1ms..10s; both TTFT (typically 50-800ms)
# and ITL (typically 5-50ms) live comfortably inside these ranges.
_LATENCY_BUCKETS_MS = (
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
)


class MetricsRegistry:
    """Owns one ``CollectorRegistry`` + typed metric handles.

    Construct fresh per benchmark run.  ``scrape()`` returns the
    Prometheus text-format body.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        # Histograms
        self.ttft_ms = Histogram(
            "goodputlab_ttft_ms",
            "Time to first token in milliseconds (per-request, client-measured)",
            labelnames=("slo_class",),
            buckets=_LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        self.itl_ms = Histogram(
            "goodputlab_itl_ms",
            "Inter-token latency in milliseconds (per-token, client-measured)",
            labelnames=("slo_class",),
            buckets=_LATENCY_BUCKETS_MS,
            registry=self.registry,
        )

        # Gauges
        self.queue_depth = Gauge(
            "goodputlab_queue_depth",
            "Current in-flight requests per pool (router snapshot)",
            labelnames=("pool",),
            registry=self.registry,
        )

        # Counters
        self.kv_tier_hit = Counter(
            "goodputlab_kv_tier_hit_total",
            "KV cache tier hits (prefix reused from LMCache)",
            labelnames=("tier",),
            registry=self.registry,
        )
        self.kv_tier_miss = Counter(
            "goodputlab_kv_tier_miss_total",
            "KV cache tier misses (prefix not found in LMCache)",
            labelnames=("tier",),
            registry=self.registry,
        )
        self.spec_acceptance = Counter(
            "goodputlab_spec_acceptance_total",
            "Speculative decoding tokens accepted by verifier (EAGLE-3)",
            registry=self.registry,
        )
        self.spec_proposed = Counter(
            "goodputlab_spec_proposed_total",
            "Speculative decoding tokens proposed by draft model",
            registry=self.registry,
        )
        self.role_flip = Counter(
            "goodputlab_role_flip_total",
            "Pool role transitions (router changed a node's role at runtime)",
            labelnames=("from", "to"),
            registry=self.registry,
        )
        self.request_total = Counter(
            "goodputlab_request_total",
            "Requests dispatched by the loadgen, by HTTP status code",
            labelnames=("status_code",),
            registry=self.registry,
        )
        self.request_error = Counter(
            "goodputlab_request_error_total",
            "Failed requests (non-2xx) by error class",
            labelnames=("error_class",),
            registry=self.registry,
        )
        self.reconcile_gate_passes = Counter(
            "goodputlab_reconcile_gate_passes_total",
            "Bench runs whose client/server reconciliation passed",
            registry=self.registry,
        )
        self.no_history = Counter(
            "goodputlab_cache_no_history_total",
            "Router lookups that found no prior prefix entry (cold-cache regime)",
            registry=self.registry,
        )

    # ---------- Recording helpers (typed wrappers) ----------

    def observe_ttft(self, slo_class: str, ttft_ms: float) -> None:
        self.ttft_ms.labels(slo_class=slo_class).observe(ttft_ms)

    def observe_itl(self, slo_class: str, itl_ms: float) -> None:
        self.itl_ms.labels(slo_class=slo_class).observe(itl_ms)

    def set_queue_depth(self, pool: str, depth: int) -> None:
        self.queue_depth.labels(pool=pool).set(depth)

    def inc_kv_hit(self, tier: str, n: int = 1) -> None:
        self.kv_tier_hit.labels(tier=tier).inc(n)

    def inc_kv_miss(self, tier: str, n: int = 1) -> None:
        self.kv_tier_miss.labels(tier=tier).inc(n)

    def inc_spec_accepted(self, n: int) -> None:
        self.spec_acceptance.inc(n)

    def inc_spec_proposed(self, n: int) -> None:
        self.spec_proposed.inc(n)

    def inc_role_flip(self, from_role: str, to_role: str) -> None:
        self.role_flip.labels(**{"from": from_role, "to": to_role}).inc()

    def inc_request(self, status_code: int) -> None:
        self.request_total.labels(status_code=str(status_code)).inc()

    def inc_request_error(self, error_class: str) -> None:
        self.request_error.labels(error_class=error_class).inc()

    def inc_gate_pass(self) -> None:
        self.reconcile_gate_passes.inc()

    def inc_no_history(self) -> None:
        self.no_history.inc()


__all__ = ["MetricsRegistry"]