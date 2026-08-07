#!/usr/bin/env python3
"""
Compare FP16 inference vs simulated 8-bit and 4-bit *weight* quantization (per-tensor
symmetric fake-quant). bitsandbytes is optional; on Jetson it often lacks matching libs.

Writes a plain-text report to --out (default: outputs/benchmark_runs/weight_bits_<ts>.txt).

Run from repo root with the in-tree mmfreelm on PYTHONPATH, e.g.:
  PYTHONPATH=/path/to/eddy_matmulfreellm python3 scripts/benchmark_weight_bits.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mmfreelm  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "ridger/MMfreeLM-370M"


def fake_quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric fake-quant in float32, cast back to t.dtype."""
    if bits >= 16:
        return t
    qmax = (1 << (bits - 1)) - 1
    tf = t.float()
    amax = tf.abs().max().clamp(min=1e-8)
    scale = amax / float(qmax)
    q = (tf / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(t.dtype)


def apply_weight_fake_quant(model: torch.nn.Module, bits: int) -> int:
    """In-place fake-quant on all floating-point parameters. Returns param count."""
    n = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.data.is_floating_point():
                p.data.copy_(fake_quantize_tensor(p.data, bits))
                n += 1
    return n


def bench_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    warmup: int,
    iters: int,
    pad_id: int,
) -> Tuple[float, float]:
    times: List[float] = []
    tok_counts: List[int] = []
    for p in prompts:
        input_ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        plen = input_ids.shape[1]
        kwargs = dict(
            max_length=plen + max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            attention_mask=torch.ones_like(input_ids),
        )
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.generate(input_ids, **kwargs)
        torch.cuda.synchronize()
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(input_ids, **kwargs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ntok = out.shape[1] - plen
            times.append(dt)
            tok_counts.append(ntok)
    tps = [tk / t for t, tk in zip(times, tok_counts)]
    return statistics.mean(tps), statistics.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--out",
        default="",
        help="Report path (default: outputs/benchmark_runs/weight_bits_<utc>.txt)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "outputs", "benchmark_runs")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_path = args.out or os.path.join(out_dir, f"weight_bits_{ts}.txt")

    lines: List[str] = []
    w = lines.append

    w("MMfreeLM weight precision comparison (fake per-tensor quant)")
    w("=" * 72)
    w(f"UTC: {ts}")
    w(f"Model: {args.model}")
    w(f"max_new_tokens: {args.max_new_tokens}  warmup: {args.warmup}  iters: {args.iters}")
    w("Note: 8b/4b here = weights rounded to symmetric int grid (simulated storage), not INT8 kernels.")
    w("bitsandbytes: skipped (Jetson builds often lack libbitsandbytes_cuda*.so).")
    w("")

    prompts = [
        "The quick brown fox",
        "Machine learning is transforming",
        "In a shocking finding, scientist discovered",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    probe_text = "Hello world, this is a test."
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids.cuda()

    results = []

    for label, bits in [("fp16_baseline", 16), ("fake_weight_8bit", 8), ("fake_weight_4bit", 4)]:
        w("-" * 72)
        w(f"Mode: {label} (internal bits={bits})")
        torch.cuda.reset_peak_memory_stats()
        t_load0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model).cuda().half().eval()
        load_s = time.perf_counter() - t_load0
        npar = sum(1 for _ in model.parameters())

        if bits < 16:
            q0 = time.perf_counter()
            n_changed = apply_weight_fake_quant(model, bits)
            q_s = time.perf_counter() - q0
            w(f"  load_wall_s: {load_s:.2f}  fake_quant_wall_s: {q_s:.4f}  float_params: {n_changed}")
        else:
            w(f"  load_wall_s: {load_s:.2f}  (no quant)  param_tensors: {npar}")

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        w(f"  peak_cuda_alloc_mib: {peak_mb:.1f}")

        mean_tps, mean_wall = bench_generate(
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            args.warmup,
            args.iters,
            pad_id,
        )
        w(f"  generate_mean_tok_per_s: {mean_tps:.2f}  mean_wall_s_per_run: {mean_wall:.4f}")

        sample = prompts[0]
        sid = tokenizer(sample, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                sid,
                max_length=sid.shape[1] + 24,
                do_sample=False,
                pad_token_id=pad_id,
                attention_mask=torch.ones_like(sid),
            )
        w(f'  sample_greedy: {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}...')

        results.append((label, bits, mean_tps, mean_wall, peak_mb))
        del model
        torch.cuda.empty_cache()
        w("")

    # Second pass reference logits: reload fp16 only for comparison tensor
    w("-" * 72)
    w("Logit drift vs fresh FP16 reference (same probe text, last-position L2 on logits vector)")
    model_ref = AutoModelForCausalLM.from_pretrained(args.model).cuda().half().eval()
    with torch.no_grad():
        ref_logits = model_ref(probe_ids).logits[:, -1, :].float().clone()
    for label, bits in [("fake_weight_8bit", 8), ("fake_weight_4bit", 4)]:
        m = AutoModelForCausalLM.from_pretrained(args.model).cuda().half().eval()
        apply_weight_fake_quant(m, bits)
        with torch.no_grad():
            cur = m(probe_ids).logits[:, -1, :].float()
        diff = torch.sqrt(((cur - ref_logits) ** 2).mean()).item()
        w(f"  {label}: rmse_last_token_logits={diff:.6f}")
        del m
        torch.cuda.empty_cache()
    del model_ref
    torch.cuda.empty_cache()
    w("")

    w("=" * 72)
    w("Summary (mean tok/s, wall per prompt-run, peak MiB)")
    for label, bits, mtps, mw, pk in results:
        w(f"  {label:18} bits={bits:2}  tps={mtps:6.2f}  wall={mw:.4f}s  peak_MiB={pk:.0f}")
    w("")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"[saved] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
