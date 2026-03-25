import subprocess

ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
print(f"Extracted ncu path: {ncu_path}")

command = [
    ncu_path, 
    "--nvtx", "--nvtx-include", "workload/",
    "--config-file", "off",
    "--export", "/home/eabban/matmulfreellm/ncu_runs/batch10Iter10",
    "--force-overwrite",
    "--target-processes", "application-only",
    "--set", "detailed",
    "/home/eabban/matmulfreellm/layer_norm_only.py",
    "-b", "10",
    "-i", "1"
]

print(command)
print(f"running command {' '.join(command)}")
# subprocess.run(command, check=True)