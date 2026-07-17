# Paired chat probe — 2026-07-17 — first real colocated-vs-chunked pair

Prior sweeps served "colocated" and "chunked" from one server config,
so the labels compared nothing. This probe restarts vLLM between
passes — `--no-enable-chunked-prefill` for colocated cells,
`--enable-chunked-prefill` for chunked cells — using the runner's
`--topologies` filter, so each label maps to a genuinely different
server configuration.

## Results

| Cell | Success | Mean TTFT | p95 TTFT | Mean ITL | Reconciles |
|---|---|---|---|---|---|
| colocated @ 4 rps | 1.00 | 721 ms | 1,271 ms | 8.5 ms | yes |
| chunked @ 4 rps | 1.00 | 663 ms | 954 ms | 8.2 ms | yes |
| colocated @ 16 rps | 1.00 | 511 ms | 626 ms | 9.2 ms | yes |
| chunked @ 16 rps | 1.00 | 648 ms | 963 ms | 8.8 ms | yes |

At these light loads and short chat prompts the two configs are within
noise of each other (n=12 measured requests per cell) — consistent
with Run 1's "chunked is not automatically faster" finding. Chunked
prefill's payoff regime (long-prompt interference at load) is what the
RAG mix at higher rates would probe.

## Run log

| Field | Value |
|---|---|
| Date | 2026-07-17 |
| Pod | RunPod secure, 1× H100 SXM 80GB (`ewzxmo3mcttjm8`), $2.99/hr |
| vLLM | 0.11.2, `--max-model-len 20480` |
| Model | Qwen/Qwen2.5-7B-Instruct |
