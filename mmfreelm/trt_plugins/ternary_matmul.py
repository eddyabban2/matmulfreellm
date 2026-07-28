"""TensorRT Python plugin for ``mmfreelm::TernaryMatMul``.

Requires TensorRT >= 10.6 (``tensorrt.plugin`` decorator API). The ONNX parser
maps custom ONNX nodes in domain ``mmfreelm`` to this plugin when the op name
matches ``TernaryMatMul``.

Reference: NVIDIA TensorRT quickly_deployable_plugins sample.
"""

import torch
import tensorrt.plugin as trtp
from tensorrt.plugin import TensorDesc, Tensor
from typing import Tuple

from mmfreelm.ops.ternary_matmul import TRT_PLUGIN_ID

_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return

    @trtp.register(TRT_PLUGIN_ID)
    def ternary_matmul_desc(
        x: TensorDesc,
        weight: TensorDesc,
        output_scale: TensorDesc,
    ) -> TensorDesc:
        out = x.like()
        out.shape_expr = [x.shape_expr[0], weight.shape_expr[0]]
        return out

    @trtp.autotune(TRT_PLUGIN_ID)
    def ternary_matmul_autotune(
        x: TensorDesc,
        weight: TensorDesc,
        output_scale: TensorDesc,
        outputs: Tuple[TensorDesc],
    ):
        del weight, output_scale, outputs
        return [
            trtp.AutoTuneCombination("FP16, INT8, FP32, FP16"),
            trtp.AutoTuneCombination("FP32, INT8, FP32, FP32"),
        ]

    @trtp.impl(TRT_PLUGIN_ID)
    def ternary_matmul_impl(
        x: Tensor,
        weight: Tensor,
        output_scale: Tensor,
        outputs: Tuple[Tensor],
        stream: int,
    ) -> None:
        del stream
        x_t = torch.as_tensor(x, device="cuda")
        w_t = torch.as_tensor(weight, device="cuda").to(dtype=x_t.dtype)
        scale = torch.as_tensor(output_scale, device="cuda").to(dtype=x_t.dtype).reshape(())
        out_t = torch.as_tensor(outputs[0], device="cuda")
        torch.matmul(x_t, w_t.t(), out=out_t)
        out_t.mul_(scale)

    _REGISTERED = True


def register_ternary_matmul_plugin() -> None:
    """Import side-effect: register plugin with TensorRT registry."""
    _register()
