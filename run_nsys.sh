#!/bin/bash

nsys profile --output=matmul_profile --trace=cuda,nvtx --stats=true \
  python /home/eabban/matmulfreellm/quiet_run.py -b 5 -i 10 -s 32 --max_new_tokens 32
