#!/usr/bin/env python3
"""TensorRT generation benchmark with Jetson tegrastats power monitoring.

Zeus NVML power APIs are not supported on Jetson Orin; this script uses tegrastats
(VDD_IN, VDD_CPU_GPU_CV, VDD_SOC) instead.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mmfreelm  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmfreelm.tensorrt import CUDAGraphAccelerator, ONNXTRTAccelerator, trt_dependencies_available

DEFAULT_MODEL = "ridger/MMfreeLM-370M"

PROMPTS = [
    "The quick brown fox",
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
    "Once upon a time in a faraway kingdom, there lived a wise old wizard who possessed magical powers",
    "Machine learning is",
    "The future of artificial intelligence will bring transformative changes to society, economy, and daily life",
]

# tegrastats: VDD_IN 6407mW/6348mW/6407mW VDD_CPU_GPU_CV 1579mW/...
_TEGRA_RE = re.compile(
    r"VDD_IN\s+(\d+)mW/\d+mW/\d+mW\s+"
    r"VDD_CPU_GPU_CV\s+(\d+)mW/\d+mW/\d+mW\s+"
    r"VDD_SOC\s+(\d+)mW/\d+mW/\d+mW"
)


@dataclass
class PowerSample:
    vdd_in_w: float
    cpu_gpu_cv_w: float
    soc_w: float
    t: float


@dataclass
class PowerStats:
  samples: List[PowerSample] = field(default_factory=list)

  def add(self, s: PowerSample) -> None:
      self.samples.append(s)

  def mean(self, attr: str) -> float:
      vals = [getattr(s, attr) for s in self.samples]
      return statistics.mean(vals) if vals else 0.0

  def max(self, attr: str) -> float:
      vals = [getattr(s, attr) for s in self.samples]
      return max(vals) if vals else 0.0

  def min(self, attr: str) -> float:
      vals = [getattr(s, attr) for s in self.samples]
      return min(vals) if vals else 0.0

  def integrate_joules(self, attr: str) -> float:
      if len(self.samples) < 2:
          return 0.0
      joules = 0.0
      for a, b in zip(self.samples, self.samples[1:]):
          dt = b.t - a.t
          joules += getattr(a, attr) * dt
      return joules


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = PowerStats()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _TEGRA_RE.search(line)
            if not m:
                continue
            self.stats.add(
                PowerSample(
                    vdd_in_w=int(m.group(1)) / 1000.0,
                    cpu_gpu_cv_w=int(m.group(2)) / 1000.0,
                    soc_w=int(m.group(3)) / 1000.0,
                    t=time.perf_counter(),
                )
            )

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> PowerStats:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.stats


def n_params(model) -> int:
    if hasattr(model, "parameter_count"):
        return model.parameter_count
    raw = model
    for attr in ("fwd", "model"):
        raw = getattr(raw, attr, raw)
    if raw is None:
        return 0
    return sum(p.numel() for p in raw.parameters())


def build_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    if not args.trt_fp32:
        model = model.half()
    model.eval()

    if args.mode == "pytorch":
        return model, tokenizer, "PyTorch FP16"

    if args.mode == "cudagraphs":
        return CUDAGraphAccelerator(model), tokenizer, "CUDA Graphs FP16"

    if not trt_dependencies_available():
        print("WARN: TensorRT/pycuda unavailable; falling back to CUDA graphs.", file=sys.stderr)
        return CUDAGraphAccelerator(model), tokenizer, "CUDA Graphs FP16 (fallback)"

    accel = ONNXTRTAccelerator(
        model,
        max_batch=1,
        max_seq=args.max_length + 96,
        model_name=args.model + ("-fp32" if args.trt_fp32 else ""),
        use_fp16=not args.trt_fp32,
        rebuild=args.rebuild_trt_engine,
    )
    precision = "FP32" if args.trt_fp32 else "FP16"
    return accel, tokenizer, f"TensorRT {precision}"


def run_benchmark(model, tokenizer, args, label: str) -> Dict:
    results = {
        "tps": [],
        "time_s": [],
        "tokens": [],
        "gflops": [],
        "vdd_in_w": [],
        "cpu_gpu_cv_w": [],
        "soc_w": [],
        "j_per_token_vdd_in": [],
        "w_per_token_vdd_in": [],
        "j_per_token_cpu_gpu": [],
        "w_per_token_cpu_gpu": [],
    }
    np_ = n_params(model)
    gen_kw = dict(max_length=args.max_length, do_sample=True, top_p=0.4, temperature=0.6)

    print(f"\n{'='*80}")
    print(f"BENCHMARK – {label}")
    print(f"model={args.model}  max_length={args.max_length}  iters={args.iterations}")
    print(f"{'='*80}\n")

    for pi, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]
        print(f'Prompt {pi+1}/{len(PROMPTS)} len={plen}: "{prompt[:55]}…"')

        with torch.no_grad():
            _ = model.generate(ids, **gen_kw)

        for it in range(args.iterations):
            monitor = TegrastatsMonitor(interval_ms=100)
            monitor.start()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, **gen_kw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            pstats = monitor.stop()

            ntok = out.shape[1] - plen
            tps = ntok / dt
            gf = (2 * np_ * ntok / dt) / 1e9

            vdd_mean = pstats.mean("vdd_in_w")
            cg_mean = pstats.mean("cpu_gpu_cv_w")
            soc_mean = pstats.mean("soc_w")
            n_samples = len(pstats.samples)
            # J/tok = avg_power * time / tokens = avg_power / tps (robust vs sparse tegrastats)
            j_per_tok_vdd = (vdd_mean * dt / ntok) if ntok else 0.0
            j_per_tok_cg = (cg_mean * dt / ntok) if ntok else 0.0
            w_per_tok_vdd = (vdd_mean / tps) if tps else 0.0
            w_per_tok_cg = (cg_mean / tps) if tps else 0.0

            results["tps"].append(tps)
            results["time_s"].append(dt)
            results["tokens"].append(ntok)
            results["gflops"].append(gf)
            results["vdd_in_w"].append(vdd_mean)
            results["cpu_gpu_cv_w"].append(cg_mean)
            results["soc_w"].append(soc_mean)
            results["j_per_token_vdd_in"].append(j_per_tok_vdd)
            results["w_per_token_vdd_in"].append(w_per_tok_vdd)
            results["j_per_token_cpu_gpu"].append(j_per_tok_cg)
            results["w_per_token_cpu_gpu"].append(w_per_tok_cg)

            print(
                f"  [{it+1}] {tps:7.1f} tok/s  {gf:7.1f} GFLOPS/s  {dt:.4f}s  "
                f"VDD_IN={vdd_mean:.2f}W  CPU/GPU/CV={cg_mean:.2f}W  "
                f"{j_per_tok_vdd*1000:.1f} mJ/tok (board)  {j_per_tok_cg*1000:.1f} mJ/tok (CPU/GPU)  "
                f"samples={n_samples}"
            )
        print()

    return results


def print_summary(results: Dict, label: str, model) -> None:
    def stat(key: str, unit: str, digits: int = 2) -> str:
        v = results[key]
        fmt = f"{{:>10.{digits}f}}"
        return (
            f"  Mean   {fmt.format(statistics.mean(v))} {unit}\n"
            f"  Median {fmt.format(statistics.median(v))} {unit}\n"
            f"  Std    {fmt.format(statistics.stdev(v) if len(v) > 1 else 0)} {unit}\n"
            f"  Min    {fmt.format(min(v))} {unit}\n"
            f"  Max    {fmt.format(max(v))} {unit}"
        )

    print(f"\n{'='*80}\nSUMMARY – {label}\n{'='*80}")
    print(f"  Params: {n_params(model):,}   Runs: {len(results['tps'])}")
    print(f"\nTokens/s:\n{stat('tps', 'tok/s')}")
    print(f"\nGFLOPS/s:\n{stat('gflops', 'GFLOPS/s')}")
    print(f"\nGeneration time:\n{stat('time_s', 's', 4)}")
    print(f"\nBoard power VDD_IN (avg during run):\n{stat('vdd_in_w', 'W')}")
    print(f"\nCPU/GPU/CV rail power (avg during run):\n{stat('cpu_gpu_cv_w', 'W')}")
    print(f"\nSOC rail power (avg during run):\n{stat('soc_w', 'W')}")
    print(f"\nEnergy per token – VDD_IN (J/tok = avg_power/tps):\n{stat('j_per_token_vdd_in', 'J/tok', 4)}")
    print(f"\nEnergy per token – CPU/GPU/CV rail (J/tok):\n{stat('j_per_token_cpu_gpu', 'J/tok', 4)}")
    print(f"\nEffective W/tok (board avg_power ÷ tok/s, equals J/tok):\n{stat('w_per_token_vdd_in', 'J/tok', 4)}")
    print(f"\nEffective W/tok (CPU/GPU rail):\n{stat('w_per_token_cpu_gpu', 'J/tok', 4)}")
    print(f"\n{'='*80}\n")


def parse_args():
    p = argparse.ArgumentParser(description="TensorRT benchmark with Jetson power metrics")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--mode",
        choices=["tensorrt", "cudagraphs", "pytorch"],
        default="tensorrt",
        help="tensorrt = ONNX→TRT engine; cudagraphs/pytorch for comparison",
    )
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--rebuild-trt-engine", action="store_true")
    p.add_argument("--trt-fp32", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "tensorrt":
        mode_key = "tensorrt"
    elif args.mode == "cudagraphs":
        mode_key = "cudagraphs"
    else:
        mode_key = "pytorch"

    # Zeus probe
    print("Checking Zeus power monitoring…")
    try:
        from zeus.monitor import ZeusMonitor

        zm = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
        zm.begin_window("probe", sync_execution=True)
        zm.end_window("probe", sync_execution=True)
        print("  Zeus: OK")
    except Exception as e:
        print(f"  Zeus: NOT AVAILABLE on this device ({type(e).__name__}: {e})")
        print("  Using tegrastats (VDD_IN / VDD_CPU_GPU_CV / VDD_SOC) instead.\n")

    print(f"Loading {args.model} …")
    model, tokenizer, label = build_model(
        argparse.Namespace(
            model=args.model,
            mode=mode_key,
            max_length=args.max_length,
            rebuild_trt_engine=args.rebuild_trt_engine,
            trt_fp32=args.trt_fp32,
        )
    )
    results = run_benchmark(model, tokenizer, args, label)
    print_summary(results, label, model)


if __name__ == "__main__":
    main()
