import subprocess
import argparse
import sys
sys.path.append('..')

parser = argparse.ArgumentParser(
    description="runs Nvidia Nsight Systems on a generate"
)

parser.add_argument(
    "-p", 
    "--profile",
    default="full",
    help="sets the sequence length of input tokens"
)

args = parser.parse_args()
metric_profile = args.profile

nsys_path = subprocess.check_output(["which", "nsys"]).decode('ascii').strip()
print(f"Extracted nsys path: {nsys_path}")

output_file = "/home/eabban/matmulfreellm/nsys_runs/generate"

# nsys_command="nsys profile --output=$output_file --trace=cuda,nvtx --stats=true -p generate \
#   $python_command /home/eabban/matmulfreellm/quiet_run.py -b 5 -i 10 -s 32 --max_new_tokens 32 "


command = [
    nsys_path, "profile",
    "--force-overwrite", "true",
    "--output=" + output_file, 
   "--trace=cuda,nvtx", 
   "--stats=true",
   "--capture-range=nvtx", 
   "-p", "workload", 
   # "--nvtx-domain-include", "workload,warmup", 
   "python", "/home/eabban/matmulfreellm/quiet_run.py", 
    "-b", "1",
    "-i", "1", 
    "-s", "1", 
    "--max_new_tokens", "2"
]

print(f"running command {' '.join(command)}")
exit_code = subprocess.run(command, check=False)
print(f"exit code: {exit_code}")
