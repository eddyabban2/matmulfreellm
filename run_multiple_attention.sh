#!/bin/bash

for batch_power in {11..14}; 
do
    batch_size=$((2**batch_power))
    echo "running for batch size: $batch_size"

    /opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/eabban/atten_ncu2/batchsize$batch_size --force-overwrite --set detailed /home/eabban/eddy_matmulfreellm/attention_only.py -b $batch_size
done

