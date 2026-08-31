# FROM ubuntu:22.04
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install base utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git sudo ca-certificates && rm -rf /var/lib/apt/lists/*

# 2. Clone XRT
RUN git clone --depth 1 --branch 202310.2.15.225 https://github.com/Xilinx/XRT.git /XRT
WORKDIR /XRT

# 3. Install XRT dependencies and build
RUN apt-get update && (yes | ./src/runtime_src/tools/scripts/xrtdeps.sh -docker)
WORKDIR /XRT/build
RUN ./build.sh -opt -noctest -j 5

# 4. Set XRT Environment Variables
ENV XILINX_XRT=/XRT/build/Release/opt/xilinx/xrt
ENV PATH=/XRT/build/Release/opt/xilinx/xrt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/XRT/build/Release/opt/xilinx/xrt/lib

# 5. Install Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

    
WORKDIR /work
ADD ../.. /work

RUN pip3 install --no-cache-dir 'numpy<2' 'pycparser<3' 'pynq==3.0.1'

RUN pip3 install --no-cache-dir -r bitnet_requirements.txt
RUN pip3 install --upgrade Pillow
    
WORKDIR /work/hybrid_experiment
CMD ["/bin/bash"]