import subprocess
import argparse

parser = argparse.ArgumentParser(
    description="profiles layer norm"
)

parser.add_argument(
    "-p", 
    "--profile",
    default="full",
    help="sets the sequence length of input tokens"
)

args = parser.parse_args()
metric_profile = args.profile

ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
print(f"Extracted ncu path: {ncu_path}")

command = [
    ncu_path, 
    "--nvtx", 
    "--nvtx-include", "workload/",
    "--config-file", "off",
    "--export", "/home/eabban/matmulfreellm/ncu_runs/batch10Iter10",
    "--force-overwrite",
    "--target-processes", "application-only",
    "--set", metric_profile,
    "/home/eabban/matmulfreellm/layer_norm_only.py",
    "-b", "10",
    "-i", "1"
]

print(f"running command {' '.join(command)}")
subprocess.run(command, check=True)