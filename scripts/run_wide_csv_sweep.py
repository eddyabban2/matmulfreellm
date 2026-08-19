"""Run a time-bounded, wide MMFreeLM CSV benchmark sweep.

Each child benchmark writes its own CSV under ``outputs/csvs``.  The runner
cycles through exponentially spaced prompt lengths until the time budget
expires, so an interrupted run still leaves all completed measurements intact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

DEFAULT_SEQUENCE_LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def parse_sequence_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item)
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive integers")
    return lengths


def build_command(args: argparse.Namespace, sequence_length: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "analysis.generate_csv",
        "--model",
        args.model,
        "--sequence_length",
        str(sequence_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--iterations",
        str(args.iterations),
        "--batch_sampling",
        "exponential",
        "--min_batch_size",
        str(args.min_batch_size),
        "--max_batch_size",
        str(args.max_batch_size),
        "--batch_samples",
        str(args.batch_samples),
    ]
    if args.collect_power_data:
        command.append("--collect_power_data")
    if args.print_csv:
        command.append("--print_csv")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0, help="total runtime budget")
    parser.add_argument("--model", default="ridger/MMfreeLM-370M")
    parser.add_argument("--sequence-lengths", type=parse_sequence_lengths,
                        default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=65536)
    parser.add_argument("--batch-samples", type=int, default=50)
    parser.add_argument("--collect-power-data", action="store_true",
                        help="record Zeus GPU power/energy columns in each CSV")
    parser.add_argument("--print-csv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.hours <= 0:
        parser.error("--hours must be positive")
    deadline = time.monotonic() + args.hours * 3600
    run_number = 0

    while time.monotonic() < deadline:
        for sequence_length in args.sequence_lengths:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            command = build_command(args, sequence_length)
            print(f"run={run_number} sequence_length={sequence_length} remaining={remaining:.0f}s", flush=True)
            print(" ".join(command), flush=True)
            if args.dry_run:
                run_number += 1
                continue
            try:
                subprocess.run(command, check=True, timeout=remaining)
            except subprocess.TimeoutExpired:
                print("Time budget reached; stopping the sweep.", flush=True)
                return 0
            except subprocess.CalledProcessError as error:
                print(f"Benchmark failed for sequence_length={sequence_length}: {error}", flush=True)
            run_number += 1

        if args.dry_run:
            break

    print("Wide sweep complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
