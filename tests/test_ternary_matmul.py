"""Tests for mmfreelm::TernaryMatMul ONNX op and TensorRT plugin."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mmfreelm  # noqa: F401
from mmfreelm.ops.ternary_matmul import ONNX_OP_NAME, ONNX_DOMAIN, ternary_matmul
from mmfreelm.ops.fusedbitnet import weight_quant


@pytest.mark.parametrize("m,k,n", [(4, 32, 16), (1, 1024, 1024)])
def test_ternary_matmul_matches_linear(m, k, n):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = weight_quant(torch.randn(n, k, device="cuda", dtype=torch.float16))
    w_int8 = w.round().clamp(-1, 1).to(torch.int8)
    scale = w.float().abs().mean().clamp(min=1e-5)

    ref = torch.nn.functional.linear(x, w_int8.to(x.dtype)) * scale.to(x.dtype)
    out = ternary_matmul(x, w_int8, scale)
    max_diff = (ref.float() - out.float()).abs().max().item()
    assert max_diff < 1e-3, f"max_diff={max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_onnx_export_contains_ternary_matmul_op():
    onnx = pytest.importorskip("onnx")
    from transformers import AutoModelForCausalLM

    from mmfreelm.tensorrt import ModelForwardWrapper, patch_all_triton_ops
    from mmfreelm.onnx_export import export_onnx_for_test

    patched = patch_all_triton_ops(
        AutoModelForCausalLM.from_pretrained("ridger/MMfreeLM-370M").cuda().half().eval()
    )
    fwd = ModelForwardWrapper(patched)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/model.onnx"
        export_onnx_for_test(fwd, path)
        model = onnx.load(path)
    ops = [n for n in model.graph.node if n.op_type == ONNX_OP_NAME and n.domain == ONNX_DOMAIN]
    assert len(ops) >= 100, f"expected many TernaryMatMul nodes, got {len(ops)}"


@pytest.mark.skipif(
    os.environ.get("MMFREELM_TRT_BUILD_TEST", "") != "1",
    reason="Set MMFREELM_TRT_BUILD_TEST=1 to run full TRT engine build (~4GB RAM)",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trt_build_with_ternary_matmul_plugin():
    trt = pytest.importorskip("tensorrt")
    try:
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        pytest.skip("pycuda required for TRT build test")

    from transformers import AutoModelForCausalLM
    from mmfreelm.tensorrt import ONNXTRTAccelerator

    model = AutoModelForCausalLM.from_pretrained("ridger/MMfreeLM-370M").cuda().half().eval()
    with tempfile.TemporaryDirectory() as d:
        engine_path = f"{d}/test.engine"
        onnx_path = f"{d}/test.onnx"
        accel = ONNXTRTAccelerator(
            model,
            max_batch=1,
            max_seq=32,
            model_name="ridger/MMfreeLM-370M-ternary-test",
            rebuild=True,
            engine_path=engine_path,
            onnx_path=onnx_path,
        )
        assert os.path.exists(engine_path)
        assert os.path.getsize(onnx_path) < 600e6, "INT8 ONNX should stay compact"
        # smoke inference
        ids = torch.zeros((1, 4), dtype=torch.long, device="cuda")
        logits = accel._step(ids)
        assert logits.shape[-1] > 1000
