import subprocess
import argparse

parser = argparse.ArgumentParser(
    description="profiles layer norm"
)

parser.add_argument(
    "-p", 
    "--profile",
    default="detailed",
    help="sets the metric profile"
)

parser.add_argument(
    "-b",
    "--batch_size",
    default="128",
    help="sets the batch size"
)

parser.add_argument(
    "-s",
    "--sequence_length",
    default="10",
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "--max_new_tokens",
    default="5",
    help="sets the number of new tokens to be generated"
)

parser.add_argument(
    "-i",
    "--iterations",
    default="1",
    help="Determines the number of iterations to benchmark for"
)

args = parser.parse_args()
metric_profile = args.profile
num_iterations = args.iterations
batch_size = args.batch_size
seq_len = args.sequence_length
max_new_tokens = args.max_new_tokens

ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
print(f"Extracted ncu path: {ncu_path}")

command = [
    ncu_path, 
    "--nvtx", 
    "--nvtx-include", "workload/",
    "--nvtx-exclude", "warmup/",
    "--config-file", "off",
    "--export", "/home/eabban/matmulfreellm/ncu_runs/generate",
    "--force-overwrite",
    "--replay-mode", "application",
    "--app-replay-match", "name",
    "--target-processes", "application-only",
    "--set", metric_profile,
    "python", "/home/eabban/matmulfreellm/quiet_run.py",
    "-b", batch_size,
    "-s", seq_len,
    "-n", max_new_tokens,
    "-i", num_iterations
]

print(command)
print(f"running command {' '.join(command)}")
subprocess.run(command, check=True)