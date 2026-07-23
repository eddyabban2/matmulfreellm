"""Plot throughput and runtime scaling from one MMFreeLM benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> dict[bool, list[dict[str, float]]]:
    groups: dict[bool, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            groups[row["Weight Packing"] == "True"].append(
                {
                    "batch_size": float(row["batch size"]),
                    "tokens_per_second": float(row["tokens_per_second"]),
                    "run_time_seconds": float(row["run_time_seconds"]),
                }
            )
    return groups


def plot(groups: dict[bool, list[dict[str, float]]], field: str, ylabel: str, title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for packed, marker, color, label in (
        (True, "o", "#1f77b4", "Packed weights"),
        (False, "s", "#ff7f0e", "Unpacked weights"),
    ):
        points = sorted((row["batch_size"], row[field]) for row in groups[packed])
        if points:
            axis.plot(*zip(*points), marker=marker, color=color, label=label)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Batch size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/images"))
    parser.add_argument("--prefix", default="throughput")
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()
    groups = load_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {args.title_suffix}" if args.title_suffix else ""
    plot(groups, "tokens_per_second", "Tokens / second",
         f"MMFreeLM throughput by batch{suffix}", args.output_dir / f"{args.prefix}_throughput.png")
    plot(groups, "run_time_seconds", "Seconds / generation",
         f"MMFreeLM generation time by batch{suffix}", args.output_dir / f"{args.prefix}_runtime.png")


if __name__ == "__main__":
    main()
