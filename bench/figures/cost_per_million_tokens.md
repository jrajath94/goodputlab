# Cost per 1M output tokens (Run 1, H100 SXM spot)

Assumptions: H100 SXM spot $1.99/hr (RunPod 2026-07), 120 output tok/s per H100, 256 output tokens per request median.

| Topology | Replicas | Mean TTFT (ms) | Mean ITL (ms) | $/1M output tok |
|----------|----------|----------------|---------------|-----------------|
| colocated | 1 | 76.53 | 6.38 | $4.61 |
| chunked | 1 | 79.58 | 6.33 | $4.61 |
| disagg | 2 | 77.24 | 6.32 | $9.21 |
| disagg_tier | 2 | 69.62 | 6.21 | $9.21 |

Linear in replica count; tier sidecar cost modelled as zero (negligible GPU; dominated by KV storage in production).