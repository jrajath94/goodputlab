# GoodputLab

SLO-aware benchmark and control plane for vLLM serving. Measures where
colocated, chunked-prefill, disaggregated, and tiered topologies actually win, and
where they do not, on real GPU hardware.

**Verified 2026-07-18:** 415 tests pass, 25 hardware-gated tests skip, and
measured source modules report 93% line coverage locally. Every number below
names its hardware, date, and commit.

## What it is

A benchmark rig plus a small control plane for vLLM. It exercises four
topologies on the same workload:

- **Colocated** - one vLLM process serving prefill and decode together, with
  vLLM's continuous batching and paged attention doing the scheduling and
  KV-cache management.
- **Chunked prefill** - prefill work interleaved with decode in chunks.
- **Disaggregated** - separate prefill and decode processes with real NIXL
  KV-cache transfer between them.
- **Disaggregated + LMCache tier** - NIXL disaggregation with an LMCache
  prefix-cache tier between the pools.

It drives these with open-loop load (Poisson arrivals, three workload shapes:
multi-turn chat, RAG with 80% prefix overlap, bursty agentic traffic) and
records mean and p95 TTFT (time to first token) plus mean ITL (inter-token
latency) per cell. The metric it optimizes is goodput: completed requests per
second under an SLO attainment target. Raw throughput without an attainment
guarantee is not a usable signal.

A reconciliation gate compares load-generator telemetry against vLLM's
`/metrics` and rejects runs with more than 2% drift.

## Headline results

**Run 1 (2026-07-09, RunPod 1x H100 SXM 80 GB, Qwen2.5-7B-Instruct, commit
`c57ee66`).** Single-process topology emulation: the router made the pool
decision, so the disagg rows carry no real prefill-to-decode transfer cost.

| Topology (emulated routing) | mean TTFT | p95 TTFT | mean ITL | success |
| --- | --- | --- | --- | --- |
| colocated | 76.5 ms | 127.3 ms | 6.38 ms | 100% |
| chunked-prefill | 79.6 ms | 137.4 ms | 6.33 ms | 100% |
| disagg (routing only, no P/D transfer) | 77.2 ms | 126.5 ms | 6.32 ms | 100% |
| disagg_tier (prefix cache path) | 69.6 ms | 111.6 ms | 6.21 ms | 100% |

Honest reading: the prefix-cache path wins TTFT on this trace (9% mean, 12%
p95 vs colocation) because deterministic replay hits the same prefixes.
The disagg-routing row is statistically indistinguishable from colocation,
which is what you expect when no real KV transfer is in the path: the router
layer itself adds ~nothing.

**True P/D disaggregation (2026-07-17, same hardware, vLLM 0.11.2, real NIXL
KV transfer: 30 transfers, 2.11 GB, 0 failed, 16.9 ms mean).** Two vLLM
processes sharing one H100.

| Cell (chat, qwen2.5-7b) | mean TTFT | mean ITL | reconciles |
| --- | --- | --- | --- |
| colocated @ 4 rps | 735 ms | 8.4 ms | yes |
| true disagg @ 4 rps | 919 ms | 7.8 ms | yes |
| colocated @ 16 rps | 775 ms | 7.2 ms | yes |
| true disagg @ 16 rps | 8,979 ms | 330 ms | no (kept as failure exhibit) |

On shared hardware, disaggregation is pure overhead at low load (+25% TTFT)
and collapses under load (two processes time-slice one GPU's SMs).
Negative results are data.

**Dedicated hardware (2026-07-17, prefill on GPU0 + decode on GPU1, 54/54
cells reconciled: 617 NIXL transfers, 30.9 GB, 0 failed).**

| Topology | Cells | mean TTFT | mean ITL |
| --- | --- | --- | --- |
| colocated | 18/18 | 900 ms | 8.8 ms |
| chunked-prefill | 18/18 | 974 ms | 8.7 ms |
| disagg (true, 2 GPU) | 18/18 | 1,034 ms | 8.3 ms |

Dedicated hardware fixes the collapse (1.00 success through 32 rps) and shows
disagg's interference-isolation benefit in ITL. But disagg still pays
+134 ms mean TTFT and 2x the hardware at every measured load. A single H100
serving a 7B model is not saturated enough for stage separation to win. That
boundary is the study's central result.

Every cell above is reproducible from `bash scripts/health.sh all` plus the
per-topology JSON in `bench/results/real/`.

## What the rig ships

1. Four vLLM topologies behind one command: colocated, chunked, NIXL
   disaggregated, NIXL + LMCache tiered.
2. A load generator with three workload shapes at Poisson and ON/OFF
   open-loop arrivals.
3. A cache-aware router with per-pool salt (a mitigation for CVE-2025-25183)
   and admission control that holds interactive SLO attainment over batch by
   queueing rather than dropping.
4. A PID prefill-to-decode autoscaler with anti-windup and a drain protocol
   that refuses to flip a role while requests are in flight.
5. An EAGLE-3 speculative-decoding simulator with auto-disable at the
   acceptance-rate crossover and a topology gate that refuses to engage on
   pure disagg (the draft-verify round trip hurts more than it saves).
6. The reconciliation gate described above.
7. A sentinel-token validator: it sends a deterministic prompt, compares the
   first-token output to a recorded fixture, and refuses to claim the P-to-D
   flow is healthy otherwise. Counter metrics alone do not detect silent
   corruption on a disagg hop.

## What is not claimed

- This is a benchmark rig, not a production fleet. All four topologies ran on
  one or two H100s, not multi-node. Multi-node P-to-D is out of scope.
- Continuous batching, paged attention, and KV-cache transfer are vLLM's
  machinery, which this harness measures. This repo does not reimplement them.
- Chunked prefill was not faster than colocation here. That matches what the
  literature predicts for small models at low batch.
- The autoscaler is unit-tested but has never been operated against a live
  GPU pool. That scenario is tracked, not claimed.

## The 216-cell matrix sweep

The full sweep is 4 topologies x 3 models x 6 rates x 3 mixes = 216 cells.
A 2-cell pilot exercised the pipeline on real GPU ($1.26). A 72-cell reduced
sweep ran on qwen2.5-7b ($1.30, 24/72 reconciled). Status of the remaining
cells and the v1.1 follow-up list are tracked in the repo; cells are marked
reconciled only when their JSON exists in `bench/results/`.

## Known limitations

- **Single-node.** UCX `cuda_ipc` transport is GPU-direct within one box.
- **Single-tenant.** No auth, no per-tenant rate limiting. It is a benchmark
  rig.
- **Ollama local baseline has a measurement hole.** The streaming-timestamp
  parser captures HTTP success but loses per-token timestamps on short
  prompts against reasoning models. Documented and tracked; the vLLM Run 1
  above remains the canonical TTFT/ITL evidence.
- Remaining GPU-only work is itemized in `docs/GPU_EXECUTION_PLAN.md` with
  cost estimates per rung.

## Quickstart

```
# no-GPU path: Ollama local baseline
make install-dev
ollama serve                      # in another shell
ollama pull qwen3:8b
GOODPUTLAB_RUN_OLLAMA=1 python3 -m pytest tests/test_ollama_smoke.py -v
python3 -m bench.ollama_smoke --model qwen3:8b --n 8
pytest -q                           # full unit suite, ~60s on a laptop

# GPU path: on a RunPod H100 pod
make provision                      # budget gate, image + model + sentinel fixture
make up-colocated                   # one topology; `make down` before switching
make health                         # /health + /v1/models + sentinel + NIXL deltas
python3 -m scripts.real_bench --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct --out bench/results/real
```

## Layout

```
control/   pool, router, pid, autoscaler (control plane, no GPU)
core/      trace schema, metrics parsing, reconciliation
loadgen/   open-loop arrivals + workload traces + http client
kv/        LMCache client + tier admission policy
spec/      EAGLE-3 draft-verify simulator + auto-disable + topology gate
obs/       Prometheus registry + /metrics HTTP exporter
bench/     mock vLLM, orchestrator, router A/B harness, real bench, ollama smoke
scripts/   health gate, disagg proxy, sentinel daemon, real bench, pull model
tests/     47 pytest files + sentinel CLI
configs/   NIXL UCX + LMCache JSON / YAML
deploy/    provisioning primitives
```

## License

MIT
