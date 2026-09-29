# KV-cache scaling - Qwen/Qwen2.5-0.5B-Instruct

Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`

## Campaign

| Field | Value |
|-------|-------|
| Campaign | kv-scaling-20260929l-qwen2p5-0p5b-rtx4090 |
| Measured |  |
| Context | Qwen/Qwen2.5-0.5B-Instruct (rev 7ae557604adf67be50417f59c2c2f167def9a775) on NVIDIA GeForce RTX 4090, vLLM 0.11.2, n=120, method=vllm-block-accounting; ignore_eos=True; decode_window=30; transcribed from bootstrap.log exhibit (kv-scaling-20260929l-qwen2p5-0p5b-rtx4090) |
| Cells | 24 |
| GPU RAM | 80 GiB (assumed, not in lookup table) |

## Regression

Slope (measured bytes/token): 12,295.8

Intercept: 314,053.6 bytes; R^2 = 1.0000.

The parity plot (theory_vs_measured.png) draws +/-10% bands around y=x: points inside the bands agree with theory within measurement noise. The parity plot only renders when the measured-vs-theory cross-check passed; a failed check raises instead of plotting.

## Knee (tokens where KV pressure dominates)

| Batch | Knee total tokens | KV at knee (GiB) | Share of GPU RAM |
|-------|-------------------|------------------|------------------|
| 8 | 131,241 | 1.50 | 1.9% |

## Cells

| seq_len | batch | n | p50 alloc (GiB) | p95-p50 (MiB) | measured B/tok | theory B/tok | ratio |
|---------|-------|---|---------------|----------------|----------------|--------------|-------|
| 1,024 | 1 | 5 | 0.012 | 0.00 | 12,509.7 | 12,288.0 | 1.018 |
| 2,048 | 1 | 5 | 0.024 | 0.00 | 12,400.4 | 12,288.0 | 1.009 |
| 4,096 | 1 | 5 | 0.047 | 0.00 | 12,344.6 | 12,288.0 | 1.005 |
| 8,192 | 1 | 5 | 0.094 | 0.00 | 12,316.4 | 12,288.0 | 1.002 |
| 16,384 | 1 | 5 | 0.188 | 0.00 | 12,302.2 | 12,288.0 | 1.001 |
| 1,024 | 2 | 5 | 0.024 | 0.00 | 12,416.4 | 12,288.0 | 1.010 |
| 2,048 | 2 | 5 | 0.048 | 0.00 | 12,353.1 | 12,288.0 | 1.005 |
| 4,096 | 2 | 5 | 0.095 | 0.00 | 12,320.8 | 12,288.0 | 1.003 |
| 8,192 | 2 | 5 | 0.188 | 0.00 | 12,305.9 | 12,288.0 | 1.001 |
| 16,384 | 2 | 5 | 0.376 | 0.00 | 12,297.4 | 12,288.0 | 1.001 |
| 1,024 | 4 | 5 | 0.049 | 0.00 | 12,369.7 | 12,288.0 | 1.007 |
| 2,048 | 4 | 5 | 0.095 | 0.00 | 12,329.4 | 12,288.0 | 1.003 |
| 4,096 | 4 | 5 | 0.189 | 0.00 | 12,311.1 | 12,288.0 | 1.002 |
| 8,192 | 4 | 5 | 0.377 | 0.00 | 12,301.8 | 12,288.0 | 1.001 |
| 16,384 | 4 | 5 | 0.752 | 0.00 | 12,296.1 | 12,288.0 | 1.001 |
| 1,024 | 8 | 5 | 0.097 | 0.00 | 12,346.3 | 12,288.0 | 1.005 |
| 2,048 | 8 | 5 | 0.191 | 0.00 | 12,321.3 | 12,288.0 | 1.003 |
| 4,096 | 8 | 5 | 0.378 | 0.00 | 12,308.5 | 12,288.0 | 1.002 |
| 8,192 | 8 | 5 | 0.753 | 0.00 | 12,302.0 | 12,288.0 | 1.001 |
| 16,384 | 8 | 5 | 1.503 | 0.00 | 12,294.6 | 12,288.0 | 1.001 |
| 1,024 | 16 | 5 | 0.194 | 0.00 | 12,341.3 | 12,288.0 | 1.004 |
| 2,048 | 16 | 5 | 0.381 | 0.00 | 12,321.7 | 12,288.0 | 1.003 |
| 4,096 | 16 | 5 | 0.756 | 0.00 | 12,311.7 | 12,288.0 | 1.002 |
| 8,192 | 16 | 5 | 1.505 | 0.00 | 12,299.1 | 12,288.0 | 1.001 |

## Method

Each cell records GPU allocated bytes (p50 and p95 over n samples) for a fixed (model, seq_len, batch) point, minus a baseline allocation measured with the same engine and no KV cache. Per-cell incremental measured bytes/token = (p50_alloc - baseline) / total_tokens. The campaign regression fits allocated bytes against total context tokens across all cells; its slope is the headline measured bytes/token and is plotted against the theoretical 2 x layers x kv_heads x head_dim x bytes_per_element value in theory_vs_measured.png.

## Limits

These numbers describe this campaign only. They are not generalized to other models, GPUs, or vLLM versions: KV layout, block size, and allocator behavior are all engine-version dependent. Cells whose measured_bytes_per_token is empty contribute to the regression through theory only. GPU RAM totals come from a lookup table and are marked assumed where the GPU was not recognized.

## Plots

- kv_bytes_per_token_vs_seqlen.png
- kv_fraction_of_gpu_ram.png
- theory_vs_measured.png
