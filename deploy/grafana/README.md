# GoodputLab Grafana dashboard

This directory holds the **OBS-02** deliverable: a Grafana dashboard
JSON that imports cleanly and references every metric declared by
`obs/registry.py` (the OBS-01 source of truth).

## Status: PLACEHOLDER (v0.2.x)

The dashboard renders **zero values** against a fresh Prometheus. It
will populate when a real sweep attaches a recording source. Per
`docs/GAP_REPORT.md` §Gap 6, shipping a placeholder is honest only if
the JSON self-describes as one — see the top-level `"description"`
field for the explicit gap.

Honest framing:

- All panel targets are valid PromQL against the OBS-01 metric names.
- The 8 panels cover the ROADMAP Phase 8 panel-token list: goodput,
  TTFT p95, ITL p95, queue depth, KV-tier hit rate, spec acceptance,
  role-flip count + thrash + zero-drop, plus prefix-index size and
  reconcile-gate passes.
- No fabricated numbers. The dashboard ships in the same shape as
  the simulators in `spec/eagle.py` and `kv/lmcache_client.py`:
  the interface is real, the data-binding is the v1.1 sweep.

## Import

1. Stand up Grafana (10.x or 11.x; the JSON uses `schemaVersion: 39`).
2. Add a Prometheus datasource that scrapes `obs/server.py` (default
   `:9100/metrics`).
3. **Dashboards → Import → Upload JSON file** → select
   `goodputlab.json`.
4. Pick the Prometheus datasource when prompted (the templating
   variable `${DS_PROMETHEUS}` resolves via the import dialog).

## Tests

`tests/test_grafana_dashboard.py` (5 tests) pins:

- The JSON parses and has the modern Grafana top-level shape.
- `schemaVersion >= 36` (Grafana 9+ — refuse to ship a v7-era file).
- Every metric declared by `MetricsRegistry` is referenced by some
  panel's PromQL target.
- Every ROADMAP Phase 8 panel token has a panel title.
- The dashboard's `description` field self-describes as a placeholder
  or explicitly references v1.1 / benchmark-sweep status.

Add a new metric to `obs/registry.py` and the test will fail until
the dashboard adds a panel target for it. That is the contract.

## Updating the schemaVersion

When bumping to a new Grafana major:

1. Export a real dashboard from that Grafana to learn the new
   `schemaVersion`.
2. Update the JSON.
3. The test's `>= 36` check accepts higher; tighten to a range if a
   regression slips in.