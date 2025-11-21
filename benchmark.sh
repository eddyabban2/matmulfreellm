#!/bin/sh
echo "********************************"
echo "             Benchmarking        "
echo "********************************"

echo "********************************"
echo "    Running Batched Benchmark"
echo "********************************"
CUDA_LAUNCH_BLOCKING=1  xpu-perf  /home/eabban/venv/bin/python3 batched_generate.py -b 128 -s 10 -i 1 > /dev/null
/home/eabban/xpu-perf/profiler/flamegraph.pl merged_trace.folded > batched_output.svg > /dev/null

echo "batched benchmark is finished now running unbatched"

echo "********************************"
echo "   Running Unbatched Benchmark"
echo "********************************"
CUDA_LAUNCH_BLOCKING=1  xpu-perf  /home/eabban/venv/bin/python3 unbatched_generate.py > /dev/null
/home/eabban/xpu-perf/profiler/flamegraph.pl merged_trace.folded > unbatched_output.svg > /dev/null

echo "benchmark is finished you can find results in unbatched_output.svg and batched_output.svg"
