import subprocess
import argparse
import sys
import os
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
os.chdir('../')
print(f"Extracted nsys path: {nsys_path}")

def create_report_name(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    model_name = model_name.replace("/", "-")
    return  os.getcwd() + "/outputs/nsys_runs/pipelined" + model_name + "batch" + str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)

batch_size = 5
seq_len = 5 
new_tokens = 5 
gpu_count = 2 

report_name = create_report_name(batch_size, new_tokens, seq_len)
print(report_name)
command = [
    nsys_path, "profile",
    "--force-overwrite", "true",
    "--output=" + report_name, 
   "--trace=cuda,nvtx", 
   "--stats=true",
   "--stop-on-exit", "false",
   "--capture-range=nvtx", 
   "-p", "workload", 
   "--trace-fork-before-exec=true",
   # "--kill=none",
   "torchrun", 
    f"--nproc_per_node={gpu_count}", 
    os.getcwd() + "/pipelined_quiet_run.py", 
    "-b", str(batch_size),
    "-i", "1", 
    "-s", str(seq_len), 
    "-n", str(new_tokens)
]

print(f"running command {' '.join(command)}")
exit_code = subprocess.run(command, check=False)
print(f"exit code: {exit_code}")
