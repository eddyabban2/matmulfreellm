#!/bin/bash

date=$(date '+%Y-%m-%d_%H:%M:%S')
python_command='/home/eabban/matmulfreellm/venv/bin/python'
output_file="nsys_runs/matmul_profile_$date"

nsys_command="nsys profile --output=$output_file --trace=cuda,nvtx --stats=true -p generate --python-backtrace=cuda \
  $python_command /home/eabban/matmulfreellm/quiet_run.py -b 5 -i 10 -s 32 --max_new_tokens 32 "

ncu_command="ncu --config-file off --export output_file --force-overwrite --set detailed $python_command quiet_run.py -b 5 -i 10 -s 32 --max_new_tokens 32"
echo $ncu_command
$ncu_command
