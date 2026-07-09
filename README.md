# GoodputLab

SLO-aware disaggregated LLM serving control plane.

## Overview

GoodputLab measures when disaggregated prefill/decode (P/D) serving beats monolithic serving under SLO constraints. It implements a routing layer that classifies requests by SLO class, performs cache-aware routing, and adapts the P:D worker ratio to maintain target latencies.

The key metric: **goodput** — throughput while maintaining p95/p99 TTFT and ITL targets.

## Architecture

```
Requests (SLO-tagged)
  ↓
Router (cache-aware prefix routing, admission control)
  ├─ Prefill Pool (2× vLLM, chunked prefill or continuous)
  └─ Decode Pool (2× vLLM, continuous batching + EAGLE-3 spec)
  ↓
KV Tier (LMCache: HBM → DRAM → disk eviction)
  ↓
Autoscaler (PID loop on queue depth imbalance)
```

## Key Decisions

**Goodput Over Throughput**: SLOs are real constraints in production. Raw throughput at 95% latency miss is useless. We optimize throughput *while maintaining* SLO attainment.

**Cache-Aware Routing**: Prefill outputs shared KV blocks (by rolling hash). Routing prefers decode workers that have seen the same prefix, reducing KV transfer cost. Tiebreaker: load (queued tokens).

**Queued Tokens as Load Signal**: Request count is misleading (short prompt = low cost). Tokens in queue = true scheduling cost.

**Honest Findings**: Disaggregation is not universally faster. For short-context prompts with no prefix reuse, monolithic serving (or chunked prefill) may win. We report this.

## Installation

```bash
pip install goodputlab
pip install "goodputlab[dev]"  # pytest, mypy, ruff
```

Requires Docker and docker-compose for topology deployment.

## Deployment

```bash
# Deploy all 4 topologies (colocated, chunked-prefill, disagg, disagg+tiering)
docker-compose -f deployments/full-stack.yml up -d

# Health checks
./scripts/health_check.sh all
```

Serves Qwen3-32B (FP8). Cold-to-serving: <20 minutes.

## Benchmarking

```bash
make bench
```

Runs the full matrix (4 topologies × 3 workloads × 6 load levels × 3 seeds). Outputs:
- **goodput_curves.csv**: throughput at ≥99% SLO attainment per topology
- **ttft_cdf.png**: TTFT distributions (p50/p95/p99 marked)
- **cost_per_1m_tokens.csv**: infrastructure cost per topology
- **failure_analysis.txt**: drill results (decode node failure, KV stall, pathological mix)

## Workloads

- **Chat**: Multi-turn, shared system prompt, 0.5-2K output tokens
- **RAG**: Long context (8-32K), short completion (50-200 tokens)
- **Agentic**: Bursty arrivals, high prefix overlap, mixed lengths

All traces reproducible from seed.

## Testing

```bash
pytest tests/ -v --cov=core --cov=control
```

Tests cover router logic, admission control, and metrics reconciliation (vs vLLM ground truth).

## Design Notes

**P:D Autoscaler with 120s Dwell**: Why not scale per-request? Thrashing. Why 120s? Allows transient queue imbalance to settle. Empirically tuned; document your choice if changing.

**Cache Affinity First, Load Second**: Two competing objectives: use cached KV vs balance queue depth. Ordering matters. We prefer hits, load-balance on ties.

**Break-Even for KV Tiering**: HBM KV cache is limited. Spilling to DRAM/disk adds latency. We plot the curve: benefit (prefix reuse rate) vs overhead (disk I/O). Expect 3-5% overhead in unpressured regime.

**Speculative Decoding × Disaggregation**: EAGLE-3 spec on the decode pool interacts with P/D separation. The draft head must be available on all decode workers. We document this interaction and potential stalls.

## Limitations

- Assumes shared NVMe or network storage for KV tiering. Direct NVMe limits tiering scope.
- Router uses rolling-hash prefix matching (not semantic similarity). Prefix reuse is conservative.
- Autoscaler runs every 5 seconds; sub-second bursty workload dynamics are smoothed.
- Failure drills are manual (kill a worker, observe recovery). No automated chaos injection yet.

## References

- vLLM: https://docs.vllm.ai
- SGLang: https://github.com/hpcaitech/sglang
- LMCache: https://github.com/lm-sys/lm-cache
- Prometheus: https://prometheus.io

## License

MIT
