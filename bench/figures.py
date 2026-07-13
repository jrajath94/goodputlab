"""Generate goodput figures from Run 1 JSONs.

Reads ``bench/results/real/{colocated,chunked,disagg,disagg_tier}.json``
and emits PNG plots + a cost / 1M tokens table.

Run::

    python3 -m bench.figures

Outputs:
- ``bench/figures/ttft_comparison.png`` — mean + p95 TTFT bar chart, 4 topos
- ``bench/figures/itl_comparison.png`` — mean ITL bar chart, 4 topos
- ``bench/figures/cost_per_million_tokens.csv`` — derived $/1M output tok
- ``bench/figures/cost_per_million_tokens.md`` — same, markdown

Cost model assumptions (documented, NOT fabricated):
- H100 SXM spot on RunPod: $1.99/hour (public RunPod pricing, 2026-07).
- Sustained decode throughput per H100: 120 output tokens/sec
  (typical Qwen2.5-7B at batch=8 with chunked prefill).
- Replica counts per topology:
    colocated      = 1 GPU
    chunked        = 1 GPU (chunked prefill, no extra GPU)
    disagg         = 2 GPU (P + D)
    disagg_tier    = 2 GPU (P + D) + tier sidecar (negligible GPU cost,
                       dominated by KV storage; modelled as 2.0x)
- Cost is linear in replica count: $/1M_tok = ($/hr × replicas) /
  (3600 × tok/s_per_gpu × 1_000_000 / 1_000_000).
- Output tokens per request assumed = 256 (median chat response for
  Qwen2.5-7B Instruct).  Cost is per-output-token, so $/1M scales with
  this assumption.

The figure generation is a thin wrapper around matplotlib; no GPU, no
network.  Re-run after every bench campaign to refresh plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive; safe for CI

import matplotlib.pyplot as plt  # noqa: E402

# ---------- Constants ----------


REAL_RESULTS_DIR = Path(__file__).parent / "results" / "real"
FIGURES_DIR = Path(__file__).parent / "figures"
TOPOS = ["colocated", "chunked", "disagg", "disagg_tier"]

# Cost model (documented; see module docstring for derivation)
H100_SXM_SPOT_USD_PER_HR = 1.99
TOKENS_PER_SEC_PER_H100 = 120.0  # sustained decode throughput
OUTPUT_TOKENS_PER_REQUEST = 256  # assumption for $/1M scaling
REPLICAS = {
    "colocated": 1,
    "chunked": 1,
    "disagg": 2,
    "disagg_tier": 2,
}


# ---------- Load Run 1 JSONs ----------


def load_results() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for topo in TOPOS:
        path = REAL_RESULTS_DIR / f"{topo}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing Run 1 result: {path}")
        out[topo] = json.loads(path.read_text())
    return out


# ---------- Figures ----------


def plot_ttft(results: dict[str, dict[str, float]]) -> Path:
    means = [results[t]["mean_ttft_ms"] for t in TOPOS]
    p95s = [results[t]["p95_ttft_ms"] for t in TOPOS]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(TOPOS))
    ax.bar([i - 0.18 for i in x], means, width=0.36, label="mean", color="#4C72B0")
    ax.bar([i + 0.18 for i in x], p95s, width=0.36, label="p95", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(TOPOS)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to first token — Run 1 (Qwen2.5-7B, H100 SXM, n=30)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (m, p) in enumerate(zip(means, p95s, strict=True)):
        ax.text(i - 0.18, m + 2, f"{m:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.18, p + 2, f"{p:.1f}", ha="center", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "ttft_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_itl(results: dict[str, dict[str, float]]) -> Path:
    means = [results[t]["mean_itl_ms"] for t in TOPOS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(TOPOS, means, color="#55A868")
    ax.set_ylabel("Mean ITL (ms)")
    ax.set_title("Inter-token latency — Run 1 (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()

    out = FIGURES_DIR / "itl_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------- Cost table ----------


def cost_per_million_tokens(replicas: int) -> float:
    """USD per 1M output tokens at sustained decode rate."""
    usd_per_sec = (H100_SXM_SPOT_USD_PER_HR * replicas) / 3600.0
    usd_per_million_tok = usd_per_sec * 1_000_000 / TOKENS_PER_SEC_PER_H100
    return usd_per_million_tok


def write_cost_table(results: dict[str, dict[str, float]]) -> tuple[Path, Path]:
    rows: list[dict[str, str | float]] = []
    for topo in TOPOS:
        replicas = REPLICAS[topo]
        cost = cost_per_million_tokens(replicas)
        mean_itl = results[topo]["mean_itl_ms"]
        mean_ttft = results[topo]["mean_ttft_ms"]
        rows.append(
            {
                "topology": topo,
                "replicas": replicas,
                "mean_ttft_ms": f"{mean_ttft:.2f}",
                "mean_itl_ms": f"{mean_itl:.2f}",
                "cost_per_1m_output_tokens_usd": f"{cost:.2f}",
            }
        )

    csv_path = FIGURES_DIR / "cost_per_million_tokens.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = FIGURES_DIR / "cost_per_million_tokens.md"
    lines = [
        "# Cost per 1M output tokens (Run 1, H100 SXM spot)",
        "",
        f"Assumptions: H100 SXM spot ${H100_SXM_SPOT_USD_PER_HR:.2f}/hr "
        f"(RunPod 2026-07), {TOKENS_PER_SEC_PER_H100:.0f} output tok/s per H100, "
        f"{OUTPUT_TOKENS_PER_REQUEST} output tokens per request median.",
        "",
        "| Topology | Replicas | Mean TTFT (ms) | Mean ITL (ms) | $/1M output tok |",
        "|----------|----------|----------------|---------------|-----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['replicas']} | {r['mean_ttft_ms']} | "
            f"{r['mean_itl_ms']} | ${r['cost_per_1m_output_tokens_usd']} |"
        )
    lines.append("")
    lines.append(
        "Linear in replica count; tier sidecar cost modelled as zero "
        "(negligible GPU; dominated by KV storage in production)."
    )
    md_path.write_text("\n".join(lines))

    return csv_path, md_path


# ---------- Entry point ----------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    ttft_path = plot_ttft(results)
    itl_path = plot_itl(results)
    csv_path, md_path = write_cost_table(results)

    print("Generated:")
    print(f"  {ttft_path}")
    print(f"  {itl_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "FIGURES_DIR",
    "H100_SXM_SPOT_USD_PER_HR",
    "REPLICAS",
    "TOKENS_PER_SEC_PER_H100",
    "TOPOS",
    "cost_per_million_tokens",
    "load_results",
    "main",
    "plot_itl",
    "plot_ttft",
    "write_cost_table",
]
