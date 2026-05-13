#!/usr/bin/env python3
"""
Build a TensorRT serialized engine from an ONNX file without importing PyTorch.
On Jetson, PyTorch + TensorRT in one process competes for unified memory during
Myelin autotuning; run this in a fresh interpreter after CPU ONNX export.

Example:
  MMFREELM_TRT_WORKSPACE_MB=128 MMFREELM_TRT_FP32=1 \\
    ./scripts/build_trt_engine_only.py \\
    --onnx /tmp/mmfreelm_xxx.onnx --engine /tmp/mmfreelm_xxx.engine \\
    --max-batch 1 --max-seq 80
"""
from __future__ import annotations

import argparse
import gc
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to .onnx (external weights in same dir)")
    p.add_argument("--engine", required=True, help="Output .engine path")
    p.add_argument("--max-batch", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=96)
    args = p.parse_args()

    import tensorrt as trt

    log = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(log, "")

    onnx_abs = os.path.abspath(args.onnx)
    onnx_dir = os.path.dirname(onnx_abs) or "."
    onnx_file = os.path.basename(onnx_abs)

    builder = trt.Builder(log)
    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(net, log)

    prev = os.getcwd()
    try:
        os.chdir(onnx_dir)
        with open(onnx_file, "rb") as f:
            blob = f.read()
        ok = parser.parse(blob)
    finally:
        os.chdir(prev)

    del blob
    gc.collect()

    if not ok:
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        return 1

    cfg = builder.create_builder_config()
    ws_mb = int(os.environ.get("MMFREELM_TRT_WORKSPACE_MB", "128"))
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mb << 20)

    if os.environ.get("MMFREELM_TRT_MINIMAL_TACTICS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        cfg.set_tactic_sources(
            int(trt.TacticSource.CUBLAS) | int(trt.TacticSource.CUBLAS_LT)
        )
        print("[TRT] Minimal tactic sources: CUBLAS | CUBLAS_LT.", file=sys.stderr)

    use_fp16 = os.environ.get("MMFREELM_TRT_FP32", "").lower() not in (
        "1",
        "true",
        "yes",
    )
    if use_fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 enabled.", file=sys.stderr)
    else:
        print("[TRT] FP32 engine.", file=sys.stderr)

    mb, ms = args.max_batch, args.max_seq
    prof = builder.create_optimization_profile()
    prof.set_shape(
        "input_ids",
        min=(1, 1),
        opt=(max(1, mb // 2), max(1, ms // 2)),
        max=(mb, ms),
    )
    cfg.add_optimization_profile(prof)

    gc.collect()
    data = builder.build_serialized_network(net, cfg)
    if data is None:
        print("TRT build returned None (OOM or unsupported graph).", file=sys.stderr)
        return 2

    raw = bytes(memoryview(data))
    eng_abs = os.path.abspath(args.engine)
    os.makedirs(os.path.dirname(eng_abs) or ".", exist_ok=True)
    with open(eng_abs, "wb") as f:
        f.write(raw)
    print(f"Wrote {eng_abs} ({len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
