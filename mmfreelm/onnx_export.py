"""Post-process ONNX graphs for MMfreeLM custom ops."""

from __future__ import annotations

import onnx
from onnx import helper

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
