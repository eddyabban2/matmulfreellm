"""Post-process ONNX graphs for MMfreeLM custom ops."""

from __future__ import annotations

import torch
import onnx
from onnx import helper

from mmfreelm.ops.hgrn.naive import onnx_recurrent_hgrn
from mmfreelm.ops.ternary_matmul import ONNX_DOMAIN, ONNX_OP_NAME


def patch_ternary_matmul_domain(onnx_path: str) -> int:
    """Assign ``mmfreelm`` domain to exported TernaryMatMul nodes for TRT plugins."""
    model = onnx.load(onnx_path)
    if not any(opset.domain == ONNX_DOMAIN for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid(ONNX_DOMAIN, 1))

    patched = 0
    for node in model.graph.node:
        if node.op_type == ONNX_OP_NAME and (not node.domain or node.domain == ""):
            node.domain = ONNX_DOMAIN
            patched += 1
    if patched:
        onnx.save(model, onnx_path)
    return patched


def export_onnx_for_test(fwd, onnx_path: str, *, patch_ternary_domain: bool = True) -> None:
    """Export a traced forward wrapper for ONNX/TRT tests."""
    import mmfreelm.layers.hgrn_bit as hgrn_bit_mod
    import mmfreelm.models.hgrn_bit.modeling_hgrn_bit as modeling_mod
    import mmfreelm.ops.hgrn.recurrent_fuse as recurrent_fuse_mod

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
    try:
        with torch.no_grad():
            torch.onnx.export(
                fwd,
                (dummy,),
                onnx_path,
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

    if patch_ternary_domain:
        patch_ternary_matmul_domain(onnx_path)
