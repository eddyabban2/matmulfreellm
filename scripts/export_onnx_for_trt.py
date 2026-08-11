#!/usr/bin/env python3
"""Export MMfreeLM ONNX for TensorRT engine build on a remote GPU.

The Jetson can export ONNX (~2 GB for 370M FP16) but engine build is slow and
memory-heavy. Use this script to produce an ONNX file, copy it to a workstation,
then build with trtexec or TensorRT Python API.

Example:
  mmfreelm-export-onnx \\
      --model ridger/MMfreeLM-370M \\
      --out /tmp/mmfreelm_370m.onnx

On the build machine (TensorRT 8.6+ / 10+):
  trtexec --onnx=/tmp/mmfreelm_370m.onnx \\
      --saveEngine=/tmp/mmfreelm_370m.engine \\
      --fp16 \\
      --minShapes=input_ids:1x1 \\
      --optShapes=input_ids:1x32 \\
      --maxShapes=input_ids:1x128
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import mmfreelm  # noqa: F401
from transformers import AutoModelForCausalLM

from mmfreelm.tensorrt import ModelForwardWrapper, patch_all_triton_ops


def export_onnx(model_name: str, out_path: str, fp32: bool = False) -> None:
    import mmfreelm.layers.hgrn_bit as hgrn_bit_mod
    import mmfreelm.models.hgrn_bit.modeling_hgrn_bit as modeling_mod
    import mmfreelm.ops.hgrn.recurrent_fuse as recurrent_fuse_mod
    from mmfreelm.ops.hgrn.naive import onnx_recurrent_hgrn

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    if not fp32:
        model = model.half()
    model = patch_all_triton_ops(model).eval()
    fwd = ModelForwardWrapper(model)

    orig_rf = recurrent_fuse_mod.fused_recurrent_hgrn
    orig_hb = hgrn_bit_mod.fused_recurrent_hgrn
    orig_hb_swiglu = hgrn_bit_mod.swiglu
    orig_model_swiglu = modeling_mod.swiglu

    def export_swiglu(x, y):
        return (x * torch.sigmoid(x)) * y

    recurrent_fuse_mod.fused_recurrent_hgrn = onnx_recurrent_hgrn
    hgrn_bit_mod.fused_recurrent_hgrn = onnx_recurrent_hgrn
    hgrn_bit_mod.swiglu = export_swiglu
    modeling_mod.swiglu = export_swiglu

    dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    print(f"[export] Writing {out_path} (trace seq=1, dynamic batch/seq axes)")
    try:
        with torch.no_grad():
            torch.onnx.export(
                fwd,
                (dummy,),
                out_path,
                opset_version=17,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                do_constant_folding=False,
                dynamo=False,
            )
    finally:
        recurrent_fuse_mod.fused_recurrent_hgrn = orig_rf
        hgrn_bit_mod.fused_recurrent_hgrn = orig_hb
        hgrn_bit_mod.swiglu = orig_hb_swiglu
        modeling_mod.swiglu = orig_model_swiglu

    from mmfreelm.onnx_export import patch_ternary_matmul_domain

    n = patch_ternary_matmul_domain(out_path)
    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"[export] Done ({size_gb:.2f} GiB, {n} TernaryMatMul nodes)")


def main():
    p = argparse.ArgumentParser(description="Export ONNX for remote TensorRT build")
    p.add_argument("--model", default="ridger/MMfreeLM-370M")
    p.add_argument("--out", required=True, help="Output .onnx path")
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required for export", file=sys.stderr)
        sys.exit(1)
    export_onnx(args.model, args.out, fp32=args.fp32)


if __name__ == "__main__":
    main()
