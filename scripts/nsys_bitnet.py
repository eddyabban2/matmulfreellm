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

parser.add_argument(
    "--prefill_decode",
    action='store_true',
    default=False,
    help="sets whether we mark prefill and decode sections of the workload"
)

args = parser.parse_args()
metric_profile = args.profile
prefill_decode = args.prefill_decode

nsys_path = subprocess.check_output(["which", "nsys"]).decode('ascii').strip()
os.chdir('../')
print(f"Extracted nsys path: {nsys_path}")

def create_report_name(bs, new_tokens, seq_len, model_name='bitnet'):
    model_name = model_name.replace("/", "-")
    return  os.getcwd() + "/outputs/nsys_runs/nsys_profiler" + model_name + "batch" + str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len) + "prefillAndDecode" + str(prefill_decode)

batch_size = 500
seq_len = 10 
new_tokens = 100

report_name = create_report_name(batch_size, new_tokens, seq_len)
print(report_name)
command = [
    nsys_path, "profile",
    "--force-overwrite", "true",
    "--output=" + report_name, 
   "--trace=cuda,nvtx", 
   "--stats=true",
   "--capture-range=nvtx", 
   "-p", "workload", 
   # "--nvtx-domain-include", "workload,warmup", 
   "python", os.getcwd() + "/quiet_run.py", 
    "-b", str(batch_size),
    "-i", "3", 
    "-s", str(seq_len), 
    "-n", str(new_tokens), 
    "--model_name", "microsoft/bitnet-b1.58-2B-4T", 
    "--prefill_decode"
]
if prefill_decode: 
    command.append("--prefill_decode")

print(f"running command {' '.join(command)}")
exit_code = subprocess.run(command, check=False)
print(f"exit code: {exit_code}")
