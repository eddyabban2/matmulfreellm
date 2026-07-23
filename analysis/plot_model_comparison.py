"""Compare packed wide-sweep results for two MMFreeLM models."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt
from analysis.plot_wide_sweep import load_rows

def draw(rows, field, ylabel, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    seqs = sorted({r["sequence_length"] for r in rows})
    colors = dict(zip(seqs, plt.colormaps["viridis"].resampled(len(seqs)).colors))
    for ax, model in zip(axes, sorted({r["model"] for r in rows})):
        groups = defaultdict(list)
        for r in rows:
            if r["model"] == model:
                groups[(r["sequence_length"], r["batch_size"])].append(r[field])
        for seq in seqs:
            points = sorted((b, median(v)) for (s,b),v in groups.items() if s == seq)
            if points: ax.plot(*zip(*points), marker="o", markersize=2.5, color=colors[seq], label=f"seq {seq}")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(model); ax.set_xlabel("Batch size"); ax.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel(ylabel); axes[1].legend(title="Prompt length", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("first", type=Path); p.add_argument("second", type=Path); p.add_argument("--output-dir", type=Path, default=Path("outputs/images/model_comparison")); a=p.parse_args()
    rows=[r for r in load_rows(a.first)+load_rows(a.second) if r["packed"]]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    draw(rows,"tokens_per_second","Tokens / second",a.output_dir/"throughput.png")
    draw(rows,"run_time_seconds","Seconds / generation",a.output_dir/"runtime.png")
    draw(rows,"prefill_seconds","Seconds to first token",a.output_dir/"ttft.png")
if __name__ == "__main__": main()
