"""Plot an effective roofline using the MMFreeLM-370M wide-sweep results.

Arithmetic intensity is estimated as batch size FLOP/byte by assuming two
operations per parameter per generated token and one FP16-equivalent model
weight read per batch. This makes the chart a workload-level comparison, not
a per-kernel hardware-counter roofline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_wide_sweep import load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/images/wide_sweep_roofline.png"))
    parser.add_argument("--parameters", type=float, default=370_000_000)
    parser.add_argument("--comparison-log", type=Path)
    parser.add_argument("--comparison-parameters", type=float, default=2_702_357_504)
    parser.add_argument("--memory-bandwidth", type=float, default=102.0,
                        help="GB/s; Jetson Orin assumption used by plot_roofline.py")
    parser.add_argument("--compute-peak", type=float, default=17_000.0,
                        help="GFLOP/s; Jetson Orin assumption used by plot_roofline.py")
    args = parser.parse_args()

    rows = [row for row in load_rows(args.log) if row["sequence_length"] == 1]
    if not rows:
        raise SystemExit("no completed sequence-length-1 rows found")
    datasets = [(str(rows[0]["model"]), rows, args.parameters, "o")]
    if args.comparison_log:
        comparison = [row for row in load_rows(args.comparison_log) if row["sequence_length"] == 1]
        if comparison:
            datasets.append((str(comparison[0]["model"]), comparison, args.comparison_parameters, "s"))

    figure, axis = plt.subplots(figsize=(9, 6))
    intensities = np.logspace(-1, 5, 300)
    memory_roof = args.memory_bandwidth * intensities
    roofline = np.minimum(memory_roof, args.compute_peak)
    axis.loglog(intensities, memory_roof, "--", color="0.45", label=f"Memory roof ({args.memory_bandwidth:g} GB/s)")
    axis.axhline(args.compute_peak, linestyle="--", color="0.2",
                 label=f"Compute roof ({args.compute_peak:,.0f} GFLOP/s)")
    axis.loglog(intensities, roofline, color="black", linewidth=1.7, label="Roofline")

    colors = {True: "#1f77b4", False: "#ff7f0e"}
    for model_name, dataset_rows, parameters, marker in datasets:
        measurements = defaultdict(list)
        for row in dataset_rows:
            measurements[(row["packed"], row["batch_size"])].append(row["tokens_per_second"])
        for packed, label in ((True, "packed"), (False, "unpacked")):
            points = sorted(
                (batch, median(values) * 2 * parameters / 1e9)
                for (is_packed, batch), values in measurements.items()
                if is_packed == packed
            )
            if points:
                axis.scatter(*zip(*points), marker=marker, s=28, color=colors[packed], alpha=0.8,
                             label=f"{model_name} ({label})", zorder=3)

    axis.set_xlabel("Effective arithmetic intensity (FLOP/byte)")
    axis.set_ylabel("Effective throughput (GFLOP/s)")
    axis.set_title("MMFreeLM effective roofline (sequence length 1)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="upper left")
    axis.text(0.02, 0.02,
              "Estimate: 2 FLOP × 370M parameters per generated token;\nFP16-equivalent weight read per batch.",
              transform=axis.transAxes, fontsize=8, va="bottom")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
