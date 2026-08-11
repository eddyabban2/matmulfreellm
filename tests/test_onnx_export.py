"""ONNX export correctness for TensorRT path (370M)."""

from __future__ import annotations

import os

import pytest
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mmfreelm  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn, onnx_recurrent_hgrn
from mmfreelm.onnx_export import export_onnx_for_test
from mmfreelm.tensorrt import (
    ModelForwardWrapper,
    default_trt_cache_paths,
    patch_all_triton_ops,
)

MODEL = "ridger/MMfreeLM-370M"


@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 24, 32])
def test_onnx_recurrent_matches_naive(seq_len):
    torch.manual_seed(0)
    x = torch.randn(1, 2, seq_len, 32, device="cuda", dtype=torch.float16) * 0.1
    g = torch.sigmoid(torch.randn(1, 2, seq_len, 32, device="cuda", dtype=torch.float16) * 0.5 + 1)
    ref, _ = naive_recurrent_hgrn(x, g, None, False)
    vec, _ = onnx_recurrent_hgrn(x, g, None, False)
    max_diff = (ref.float() - vec.float()).abs().max().item()
    assert max_diff < 1e-2, f"seq={seq_len} max_diff={max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_patched_pytorch_matches_raw_hf():
    """Export patches must match Triton/HF inference (argmax parity)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    raw = AutoModelForCausalLM.from_pretrained(MODEL).cuda().half().eval()
    patched = patch_all_triton_ops(
        AutoModelForCausalLM.from_pretrained(MODEL).cuda().half().eval()
    )
    prompts = ["The quick brown fox", "Machine learning is", "Once upon a time"]
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        for seq_len in sorted({1, ids.shape[1]}):
            sl = ids[:, :seq_len]
            with torch.no_grad():
                raw_logits = raw(sl).logits[:, -1]
                pat_logits = patched(sl).logits[:, -1]
            assert raw_logits.argmax().item() == pat_logits.argmax().item(), (
                f"argmax mismatch seq={seq_len} prompt={prompt!r}"
            )
            max_diff = (raw_logits.float() - pat_logits.float()).abs().max().item()
            assert max_diff < 1.0, f"seq={seq_len} max_diff={max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_onnx_runtime_matches_patched_pytorch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    patched = patch_all_triton_ops(
        AutoModelForCausalLM.from_pretrained(MODEL).cuda().half().eval(),
        use_ternary_op=False,
    )
    pt_fwd = ModelForwardWrapper(patched)

    onnx_path = str(tmp_path / "model.onnx")
    export_onnx_for_test(pt_fwd, onnx_path, patch_ternary_domain=False)

    sess = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    prompts = ["The quick brown fox", "Machine learning is", "Once upon a time"]
    for prompt in prompts:
        full_ids = tokenizer(prompt, return_tensors="pt").input_ids
        test_lengths = sorted({1, full_ids.shape[1]})
        for seq_len in test_lengths:
            ids = full_ids[:, :seq_len]
            with torch.no_grad():
                pt_logits = pt_fwd(ids.cuda())
            onnx_out = sess.run(None, {"input_ids": ids.numpy().astype("int64")})[0]
            max_diff = abs(pt_logits.float().cpu().numpy() - onnx_out).max()
            assert pt_logits.argmax(-1).item() == onnx_out.argmax(-1), (
                f"argmax mismatch seq={seq_len} prompt={prompt!r}"
            )
            # FP16 ONNX weights vs GPU PyTorch; allow ~2 logits tolerance.
            assert max_diff < 2.0, f"seq={seq_len} max_diff={max_diff}"
