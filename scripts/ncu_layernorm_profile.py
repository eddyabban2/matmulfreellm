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

double_precision_metrics = [ "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", 
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum", 
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum" ]
single_precision_metrics = ["sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"]
half_precision_metrics = ["sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum"]
tensor_core_metrics = ["sm__ops_path_tensor_op_hmma_pred_on.sum",
        "sm__ops_path_tensor_op_imma_pred_on.sum"]


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
    "--metrics", ",".join(["dram__bytes.sum"] + 
            double_precision_metrics + 
            single_precision_metrics + 
            half_precision_metrics + 
            tensor_core_metrics),
    "/home/eabban/matmulfreellm/layer_norm_only.py",
    "-b", "10",
    "-i", "1"
]

print(f"running command {' '.join(command)}")
subprocess.run(command, check=True)