"""Plot runtime and time-to-first-token from a wide sweep log."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt
from analysis.plot_wide_sweep import load_rows

def draw(rows, field, ylabel, title, output):
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence_length"], row["batch_size"])].append(row[field])
    colors = plt.colormaps["viridis"].resampled(len({r["sequence_length"] for r in rows}))
    for color, seq in zip(colors.colors, sorted({r["sequence_length"] for r in rows})):
        points = sorted((batch, median(values)) for (s, batch), values in groups.items() if s == seq)
        if points:
            ax.plot(*zip(*points), marker="o", markersize=2.5, color=color, label=f"seq {seq}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Batch size"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=.25); ax.legend(title="Prompt length", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("log", type=Path); parser.add_argument("--output-dir", type=Path, default=Path("outputs/images")); args=parser.parse_args()
    rows=load_rows(args.log)
    if not rows: raise SystemExit("no completed rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model=str(rows[0]["model"])
    draw(rows, "run_time_seconds", "Seconds / generation", f"{model} generation runtime by batch", args.output_dir / "wide_sweep_runtime_by_batch.png")
    draw(rows, "prefill_seconds", "Seconds to first token", f"{model} time to first token by batch", args.output_dir / "wide_sweep_ttft_by_batch.png")
if __name__ == "__main__": main()
