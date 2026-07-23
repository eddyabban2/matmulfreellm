"""Create summary plots from a wide MMFreeLM benchmark sweep."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"run=(\d+) sequence_length=(\d+)")
CSV_RE = re.compile(r"Data written to (outputs/csvs/benchmark_results-[^\s]+\.csv)")


def load_rows(log_path: Path) -> list[dict[str, object]]:
    current_run: tuple[int, int] | None = None
    rows: list[dict[str, object]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        run_match = RUN_RE.search(line)
        if run_match:
            current_run = (int(run_match.group(1)), int(run_match.group(2)))
            continue
        csv_match = CSV_RE.search(line)
        if not csv_match or current_run is None:
            continue
        csv_path = log_path.parents[2] / csv_match.group(1)
        with csv_path.open(newline="") as file:
            for row in csv.DictReader(file):
                rows.append(
                    {
                        "run": current_run[0],
                        "sequence_length": current_run[1],
                        "model": row["model"],
                        "packed": row["Weight Packing"] == "True",
                        "batch_size": int(row["batch size"]),
                        "tokens_per_second": float(row["tokens_per_second"]),
                        "run_time_seconds": float(row["run_time_seconds"]),
                        "prefill_seconds": float(row["Avg Prefill Time (s)"]),
                    }
                )
    return rows


def grouped(rows: list[dict[str, object]], fields: tuple[str, ...]):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def plot_throughput_curves(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    curves = grouped(rows, ("packed", "sequence_length", "batch_size"))
    sequence_lengths = sorted({row["sequence_length"] for row in rows})
    color_map = plt.colormaps["viridis"].resampled(len(sequence_lengths))
    colors = dict(zip(sequence_lengths, color_map.colors))
    for packed, axis in zip((True, False), axes):
        for sequence_length in sequence_lengths:
            points = []
            for (is_packed, seq, batch), values in curves.items():
                if is_packed == packed and seq == sequence_length:
                    points.append((batch, median(value["tokens_per_second"] for value in values)))
            if points:
                points.sort()
                axis.plot(*zip(*points), marker="o", markersize=2.5, linewidth=1,
                          label=f"seq {sequence_length}", color=colors[sequence_length])
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Batch size")
        axis.set_title("Packed weights" if packed else "Unpacked weights")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Tokens / second")
    axes[1].legend(title="Prompt length", fontsize=8, ncol=2)
    model_name = str(rows[0]["model"])
    figure.suptitle(f"{model_name} throughput across the wide sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_summary(rows: list[dict[str, object]], output: Path, metric: str, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    per_run = grouped(rows, ("packed", "sequence_length", "run"))
    summary = defaultdict(list)
    for (packed, sequence_length, _), values in per_run.items():
        if metric == "peak_tps":
            value = max(row["tokens_per_second"] for row in values)
        else:
            value = max(row["batch_size"] for row in values)
        summary[(packed, sequence_length)].append(value)
    for packed, marker, label in ((True, "o", "Packed"), (False, "s", "Unpacked")):
        points = sorted(
            (sequence_length, median(values))
            for (is_packed, sequence_length), values in summary.items()
            if is_packed == packed
        )
        if points:
            axis.plot(*zip(*points), marker=marker, linewidth=1.5, label=label)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Prompt length")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/images"))
    args = parser.parse_args()
    rows = load_rows(args.log)
    if not rows:
        raise SystemExit("no completed benchmark rows found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_throughput_curves(rows, args.output_dir / "wide_sweep_throughput_by_batch.png")
    plot_summary(rows, args.output_dir / "wide_sweep_peak_throughput.png", "peak_tps",
                 "Peak throughput by prompt length", "Peak tokens / second")
    plot_summary(rows, args.output_dir / "wide_sweep_max_batch.png", "max_batch",
                 "Maximum fitting batch by prompt length", "Maximum batch size")


if __name__ == "__main__":
    main()
