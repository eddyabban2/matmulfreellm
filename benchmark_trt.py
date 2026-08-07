"""
TensorRT-Accelerated Benchmark for MMfreeLM / HGRN  –  Jetson-native
======================================================================
Patches Triton-backed ops (see tensorrt_generation.patch_all_triton_ops).

Modes
──────────────────
  cudagraphs   CUDA Graph on single-step forward  (default, zero extra installs)
  onnx_trt     ONNX → JetPack TRT engine  (pip install pycuda)
  baseline     Original FP16 HF generate()
"""

import argparse
import os
import statistics
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mmfreelm  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorrt_generation import (
    CUDAGraphAccelerator,
    ONNXTRTAccelerator,
    trt_dependencies_available,
)

MODEL_NAME = "ridger/MMfreeLM-370M"

PROMPTS = [
    "The quick brown fox",
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
    "Once upon a time in a faraway kingdom, there lived a wise old wizard who possessed magical powers",
    "Machine learning is",
    "The future of artificial intelligence will bring transformative changes to society, economy, and daily life",
]


class BaselineModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self._pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    def generate(
        self,
        input_ids,
        max_length=32,
        do_sample=True,
        top_p=0.4,
        temperature=0.6,
        **_,
    ):
        with torch.no_grad():
            return self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self._pad,
                max_length=max_length,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )

    def parameters(self):
        return self.model.parameters()


def n_params(model):
    raw = model
    for attr in ("fwd", "model"):
        raw = getattr(raw, attr, raw)
    return sum(p.numel() for p in raw.parameters())


def benchmark(model, prompts, tokenizer, max_length, n_iter, batches):
    res = {"tps": [], "time": [], "tokens": [], "gflops": [], "plen": []}
    np_ = n_params(model)
    print(f"\n{'='*80}")
    print(
        f"BENCHMARK  prompts={len(prompts)}  iters={n_iter}  "
        f"max_length={max_length}  batch={batches}"
    )
    print(f"{'='*80}\n")

    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]
        batch_ids = ids.repeat(batches, 1)
        gen_kw = dict(
            max_length=max_length, do_sample=True, top_p=0.4, temperature=0.6
        )

        print(f"Prompt {pi+1}/{len(prompts)} len={plen}: \"{prompt[:55]}…\"")
        with torch.no_grad():
            _ = model.generate(batch_ids, **gen_kw)

        for it in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(batch_ids, **gen_kw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ntok = out.shape[0] * out.shape[1] - plen * batches
            tps = ntok / dt
            gf = (2 * np_ * ntok / dt) / 1e9
            res["tps"].append(tps)
            res["time"].append(dt)
            res["tokens"].append(ntok)
            res["gflops"].append(gf)
            res["plen"].append(plen)
            print(f"  [{it+1}] {tps:8.1f} tok/s  {gf:8.1f} GFLOPS/s  {dt:.4f}s")
        print()
    return res


def print_results(res, model, label):
    def _s(v, u, d=2):
        f = f"{{:>12.{d}f}}"
        return (
            f"  Mean   {f.format(statistics.mean(v))} {u}\n"
            f"  Median {f.format(statistics.median(v))} {u}\n"
            f"  Std    {f.format(statistics.stdev(v) if len(v) > 1 else 0)} {u}\n"
            f"  Min    {f.format(min(v))} {u}\n"
            f"  Max    {f.format(max(v))} {u}"
        )

    print(f"\n{'='*80}\nRESULTS – {label}\n{'='*80}")
    print(f"  Params: {n_params(model):,}   Runs: {len(res['tps'])}")
    print(f"\nTokens/s:\n{_s(res['tps'],'tok/s')}")
    print(f"\nGFLOPS/s:\n{_s(res['gflops'],'GFLOPS/s')}")
    print(f"\nTime:\n{_s(res['time'],'s',d=4)}")
    print(
        f"\nTokens: total={sum(res['tokens']):,}  mean={statistics.mean(res['tokens']):.1f}"
    )
    unique = sorted(set(res["plen"]))
    if len(unique) > 1:
        print(f"\n  {'Len':<8} {'TPS':>12} {'GFLOPS/s':>12}")
        for l in unique:
            idx = [i for i, p in enumerate(res["plen"]) if p == l]
            print(
                f"  {l:<8} "
                f"{statistics.mean([res['tps'][i] for i in idx]):>12.2f} "
                f"{statistics.mean([res['gflops'][i] for i in idx]):>12.2f}"
            )
    print(f"\n{'='*80}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["cudagraphs", "onnx_trt", "baseline"], default="cudagraphs"
    )
    p.add_argument("--fixed-point", action="store_true")
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--rebuild-engine", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().half()
    model.eval()

    if args.fixed_point:
        try:
            from generate_integrated import replace_with_fixed_point_hgrn

            model = replace_with_fixed_point_hgrn(model)
            print("[INFO] ✓ HGRN → fixed-point.")
        except ImportError:
            print("[WARN] generate_integrated not found.")

    if args.mode == "cudagraphs":
        print("[INFO] Mode: CUDA Graphs")
        accel = CUDAGraphAccelerator(model)
        label = "CUDA Graphs FP16"

    elif args.mode == "onnx_trt":
        if not trt_dependencies_available():
            print("[WARN] Falling back to CUDA Graphs.")
            accel = CUDAGraphAccelerator(model)
            label = "CUDA Graphs FP16 (fallback)"
        else:
            print("[INFO] Mode: ONNX → TensorRT (JetPack native)")
            accel = ONNXTRTAccelerator(
                model,
                max_batch=args.batches,
                max_seq=args.max_length + 64,
                model_name=MODEL_NAME,
                use_fp16=True,
                rebuild=args.rebuild_engine,
            )
            label = "TensorRT FP16 (JetPack)"
    else:
        print("[INFO] Mode: Baseline FP16")
        accel = BaselineModel(model, tokenizer)
        label = "Baseline FP16"

    res = benchmark(
        accel,
        PROMPTS,
        tokenizer,
        max_length=args.max_length,
        n_iter=args.iterations,
        batches=args.batches,
    )
    print_results(res, accel, label)

    print(f"{'='*80}\nSAMPLE GENERATION (5 batches)\n{'='*80}")
    ids = tokenizer(PROMPTS[1], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outs = accel.generate(
            ids.repeat(5, 1),
            max_length=45,
            do_sample=True,
            top_p=0.4,
            temperature=2.0,
        )
    print(f"\nInput: {PROMPTS[1]}")
    for o in outs:
        print(f"  → {tokenizer.decode(o, skip_special_tokens=True)}")

    print(f"\n{'='*80}")
    print("TIPS:")
    print("  --mode cudagraphs   Zero installs, graphs the forward pass")
    print("  --mode onnx_trt     Max throughput (pip install pycuda)")
    print("  --rebuild-engine    Force TRT engine rebuild")
    print("  --fixed-point       Ternary HGRN, combinable with any mode")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
