#!/bin/bash
echo "********************************"
echo "             Benchmarking        "
echo "********************************"


echo "********************************"
echo "        Getting Metrics "
echo "********************************"

batch_size=1024
iter=5
/home/eabban/venv/bin/python3 batched_generate.py -b $batch_size -s 10 -i $iter -m -p > batched_generate.batch"$batch_size".iter"$iter".stdout

# echo "********************************"S
# echo "    Running Batched Benchmark"
# echo "********************************"
# CUDA_LAUNCH_BLOCKING=1  xpu-perf  /home/eabban/venv/bin/python3 batched_generate.py -b 128 -s 10 -i 1 > /dev/null
# /home/eabban/xpu-perf/profiler/flamegraph.pl merged_trace.folded > batched_output.svg > /dev/null

# echo "batched benchmark is finished now running unbatched"

# echo "********************************"
# echo "   Running Unbatched Benchmark"
# echo "********************************"
# CUDA_LAUNCH_BLOCKING=1  xpu-perf  /home/eabban/venv/bin/python3 unbatched_generate.py > /dev/null
# /home/eabban/xpu-perf/profiler/flamegraph.pl merged_trace.folded > unbatched_output.svg > /dev/null

echo "benchmark is finished you can find results in unbatched_output.svg, batched_output.svg and batched_generate.batch1024.stdout"
