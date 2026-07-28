"""Ternary matrix multiply for ONNX / TensorRT export.

Computes ``output = (x @ weight_int8.T + bias) * output_scale`` where
``weight_int8`` holds values in {-1, 0, 1}. Exports as ONNX custom op
``mmfreelm::TernaryMatMul`` for TensorRT plugin fusion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

ONNX_DOMAIN = "mmfreelm"
ONNX_OP_NAME = "TernaryMatMul"
TRT_PLUGIN_ID = f"{ONNX_DOMAIN}::{ONNX_OP_NAME}"


class TernaryMatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_int8: torch.Tensor,
        output_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        w = weight_int8.to(dtype=x2.dtype)
        y = F.linear(x2, w, bias)
        y = y * output_scale.to(dtype=y.dtype)
        return y.reshape(*orig_shape[:-1], y.shape[-1])

    @staticmethod
    def symbolic(g, x, weight_int8, output_scale, bias):
        inputs = [x, weight_int8, output_scale]
        if bias is not None:
            inputs.append(bias)
        return g.op(ONNX_OP_NAME, *inputs, domain_s=ONNX_DOMAIN)


def ternary_matmul(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    output_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ternary linear: int8 weights in {-1,0,1}, per-tensor output scale."""
    return TernaryMatMulFunction.apply(x, weight_int8, output_scale, bias)
