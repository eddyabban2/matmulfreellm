#!/bin/bash

date=$(date '+%Y-%m-%d_%H:%M:%S')
python_command='/home/eabban/matmulfreellm/venv/bin/python'
output_file="nsys_runs/matmul_profile_$date"


nsys_command="nsys profile --output=$output_file --trace=cuda,nvtx --stats=true --python-backtrace=cuda \
  $python_command /home/eabban/matmulfreellm/attention_only.py -b 5 -i 10"

$nsys_command
