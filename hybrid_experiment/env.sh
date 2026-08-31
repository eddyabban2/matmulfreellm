# Alveo U250 helpers for gpu01.  Source this file:
#     source /au250_xrt/env.sh
#

container_name="eddy_u250"


_AU250_BDF=0000:64:00.0
# Refuse to launch at/above this FPGA die temp (C). Passive "PQ" card: unstable when hot.
_AU250_TEMP_LIMIT=${AU250_TEMP_LIMIT:-85}

_au250_fpga_temp() { cat /sys/bus/pci/devices/${_AU250_BDF}/xmc.m.*/xmc_fpga_temp 2>/dev/null | head -1; }
_au250_mgmt()      { ls /dev/xclmgmt* 2>/dev/null | head -1; }
# Expose every DRM render node (the Xilinx one's index isn't stable on this GPU box) + mgmt.
_au250_devflags()  { local f n; for n in /dev/dri/renderD*; do f="$f --device $n"; done; echo "$f --device $(_au250_mgmt)"; }

au250-temp() { echo "AU250 FPGA die: $(_au250_fpga_temp) C   (warn 88, shutdown 97)"; }

au250-status() {   # what shell/partitions are running on the card
  docker run --rm --privileged --device "$(_au250_mgmt)" -v /sys:/sys \
    -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro xbmgmt215 \
    xbmgmt examine -d "$_AU250_BDF" -r platform
}

au250-reload() {   # (re)load the xdma shell — normally automatic at boot via systemd
  sudo systemctl restart au250-shell.service && echo "au250: shell reload requested"
}

au250-run() {      # au250-run <cmd...>   run in the matched container; your CWD is mounted at /work
  if [ "$#" -eq 0 ]; then
    echo "usage: au250-run <command...>   e.g.  au250-run python3 pynq_add_example.py foo.xclbin"
    return 2
  fi
  local t; t=$(_au250_fpga_temp)
  if [ -n "$t" ] && [ "$t" -ge "$_AU250_TEMP_LIMIT" ]; then
    echo "au250: REFUSING to run — FPGA at ${t}C (>= ${_AU250_TEMP_LIMIT}C). Let it cool / check airflow." >&2
    echo "       This passive card is unstable near its 97C shutdown. Override: AU250_TEMP_LIMIT=NN au250-run ..." >&2
    return 1
  fi
  docker run --rm --privileged $(_au250_devflags) \
    -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro -e HF_TOKEN --gpus all --network host \
    -v "$PWD/..":/app -w /work $container_name \
    bash -c 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; exec "$@"' _ "$@"
}

build_docker() { 
  docker build -t $container_name -f Dockerfile ..
}

run_test() { 
  au250-run python3 /au250_xrt/example/pynq_add_example.py /au250_xrt/example/MaxCores_370M.xclbin
  local t; t=$(_au250_fpga_temp)
  if [ -n "$t" ] && [ "$t" -ge "$_AU250_TEMP_LIMIT" ]; then
    echo "au250: REFUSING to run — FPGA at ${t}C (>= ${_AU250_TEMP_LIMIT}C). Let it cool / check airflow." >&2
    echo "       This passive card is unstable near its 97C shutdown. Override: AU250_TEMP_LIMIT=NN au250-run ..." >&2
    return 1
  fi

  docker run --rm --privileged $(_au250_devflags) \
      -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
      -v /au250_xrt:/au250_xrt:ro -e HF_TOKEN  --gpus all --network host \
      -v "$PWD/..":/work -w /work $container_name \
      bash -c 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; exec python3 quiet_run.py -b 5 -s 10 -n 10 -i 1 --model_name microsoft/bitnet-b1.58-2B-4T --prefill_decode'
}

enter_docker() { 
  docker run --rm -it --privileged $(_au250_devflags) \
    -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro -e HF_TOKEN --gpus all --network host \
    -v "$PWD/..":/work $container_name \
    bash
}

run_layer_test() { 
  local t; t=$(_au250_fpga_temp)
  if [ -n "$t" ] && [ "$t" -ge "$_AU250_TEMP_LIMIT" ]; then
    echo "au250: REFUSING to run — FPGA at ${t}C (>= ${_AU250_TEMP_LIMIT}C). Let it cool / check airflow." >&2
    echo "       This passive card is unstable near its 97C shutdown. Override: AU250_TEMP_LIMIT=NN au250-run ..." >&2
    return 1
  fi

  docker run --rm --privileged $(_au250_devflags) \
      -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
      -v /au250_xrt:/au250_xrt:ro -e HF_TOKEN  --gpus all --network host \
      -v "$PWD/..":/work -w /work/hybrid_experiment $container_name \
      bash -c 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; exec python3 U250_Bitlinear.py'
}

echo "AU250 ready.  au250-run <cmd...> | au250-status | au250-temp | au250-reload   (temp guard ${_AU250_TEMP_LIMIT}C)"
