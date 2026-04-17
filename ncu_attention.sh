#!/bin/bash

ncu_path="/opt/nvidia/nsight-compute/2025.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu"

command="$ncu_path \
    --nvtx --nvtx-include workload/ --config-file off --export /home/eabban/matmulfreellm/ncu_runs/batch10Iter10 \
    --force-overwrite --target-processes application-only --set detailed \
    /home/eabban/matmulfreellm/attention_only.py -b 10 -i 1"       

echo "running command $command" 
$command


