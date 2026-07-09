# GoodputLab

Control plane for disaggregated LLM serving: separated prefill and decode workers with cache-aware routing, admission control, and autoscaling.

Goodput here means completed requests per second that meet their SLO, as opposed to raw throughput.

## Status

Early stage. Phase 1 (serving topologies: disaggregated, colocated, chunked, tiered) is in progress. CI and tests are not set up yet.

## License

MIT
