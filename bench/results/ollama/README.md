# Ollama local baseline — measurement notes

## What this directory holds

Per-model CampaignReport JSON snapshots from `bench/ollama_smoke.py` run
against a local Ollama server (M1 Max unified memory, macOS Darwin).

Each file `qwen3_<tag>.json` is one measurement of the loadgen →
vLLMHttpClient → local Ollama pipeline, gated by
`GOODPUTLAB_RUN_OLLAMA=1` and the `tests/test_ollama_smoke.py`
contract.

## Run it yourself

```bash
# local Ollama must be running on :11434
ollama serve &
ollama pull qwen3:8b

# from repo root:
GOODPUTLAB_RUN_OLLAMA=1 python3 -m pytest tests/test_ollama_smoke.py -v
python3 -m bench.ollama_smoke --model qwen3:8b --n 8 --output-tokens 96
```

## Honest measurement hole

The captured `mean_ttft_ms` and `mean_itl_ms` may report `0.0` even
when `success_rate == 1.0`. This is a known issue in the streaming
SSE timestamp parser (`loadgen/client.py` → `_stream_tokens`) when
the upstream returns OpenAI-style `chat.completion.chunk` lines
where the first chunk has no visible `content` (Ollama's qwen3 family
emits reasoning tokens before any visible content, so the per-token
timestamp list ends up empty for short prompts).

A direct probe confirms streaming works:

```
status= 200
t_first_delta=743.0ms (first data: line received)
```

The probe-vs-bench delta is exactly the parse-stripped-time bug. The
fix belongs in `loadgen/client.py`; tracked in the issues list. Until
fixed, treat Ollama results as a smoke test for the HTTP plumbing,
not as TTFT/ITL evidence.

**Re-measured 2026-07-16** (`--model qwen3:8b --n 8`): the smoke run
captured non-zero telemetry (success 100 %, mean TTFT 1145.6 ms, mean
ITL 22.9 ms, `qwen3_8b.json`), so the parser returned per-token
timestamps on this run. The hole above is therefore intermittent
(prompt-length/reasoning-token dependent), not permanent. The policy is
unchanged either way: Ollama numbers validate request shape, streaming
parse, and result plumbing only — they are never vLLM or P/D evidence.

Run 1 on vLLM (commit `c57ee66`) is the canonical TTFT/ITL evidence.
