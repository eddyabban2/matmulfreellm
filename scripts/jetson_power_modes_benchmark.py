#!/usr/bin/env python3
"""Benchmark power/perf across Jetson workload configurations and nvpmodel modes.

On Jetson Orin Nano Super, nvpmodel mode changes require a reboot. This script:
  1. Reports current mode and clock caps from sysfs
  2. Optionally requests a mode switch via jtop (fails without reboot on Super)
  3. Runs tegrastats-instrumented workloads: idle, cpu_stress, gpu_matmul, llm_generate

Usage (after manually setting nvpmodel + rebooting):
  sudo nvpmodel -m 0 && sudo reboot   # 15W
  mmfreelm-bench-jetson-power --workloads all

  sudo nvpmodel -m 1 && sudo reboot   # 25W
  mmfreelm-bench-jetson-power --workloads all

  sudo nvpmodel -m 2 && sudo reboot   # MAXN_SUPER
  mmfreelm-bench-jetson-power --workloads all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

TEGRA_RE = re.compile(
    r"VDD_IN\s+(\d+)mW/\d+mW/\d+mW\s+"
    r"VDD_CPU_GPU_CV\s+(\d+)mW/\d+mW/\d+mW\s+"
    r"VDD_SOC\s+(\d+)mW/\d+mW/\d+mW"
)

NVP_MODEL_CAPS = {
    "15W": {"cpu_max_mhz": 1497, "gpu_max_mhz": 612, "emc_max_mhz": 2133, "budget_w": 15},
    "25W": {"cpu_max_mhz": 1344, "gpu_max_mhz": 918, "emc_max_mhz": 3199, "budget_w": 25},
    "MAXN_SUPER": {"cpu_max_mhz": 1728, "gpu_max_mhz": 1020, "emc_max_mhz": 3199, "budget_w": None},
}


@dataclass
class PowerSample:
    vdd_in_w: float
    cpu_gpu_cv_w: float
    soc_w: float
    t: float


@dataclass
class PowerStats:
    samples: List[PowerSample] = field(default_factory=list)

    def mean(self, attr: str) -> float:
        vals = [getattr(s, attr) for s in self.samples]
        return statistics.mean(vals) if vals else 0.0


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = PowerStats()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = TEGRA_RE.search(line)
            if not m:
                continue
            self.stats.samples.append(
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
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)
        return self.stats


def read_sysfs_clocks() -> Dict[str, int]:
    def _read(path: str) -> int:
        try:
            return int(open(path).read().strip())
        except OSError:
            return -1

    return {
        "cpu_cur_khz": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "cpu_max_khz": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "gpu_cur_hz": _read("/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"),
        "gpu_max_hz": _read("/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
        "emc_hz": _read("/sys/kernel/nvpmodel_clk_cap/emc"),
    }


def query_nvpmodel() -> Dict:
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True).strip().splitlines()
        name = out[0].replace("NV Power Mode:", "").strip()
        mode_id = int(out[1])
        return {"name": name, "id": mode_id}
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return {"name": "unknown", "id": -1}


def try_jtop_switch(mode_id: int) -> bool:
    try:
        from jtop import jtop
    except ImportError:
        print("[WARN] jtop Python module not available (use /usr/bin/python3)", file=sys.stderr)
        return False
    with jtop() as jetson:
        before = jetson.nvpmodel.name
        jetson.nvpmodel = mode_id
        for _ in range(20):
            time.sleep(0.5)
            if not jetson.nvpmodel.is_running():
                break
        time.sleep(2)
        after = jetson.nvpmodel.name
        return after != before
    return False


def _cpu_burn(stop: threading.Event) -> None:
    x = 1.0
    while not stop.is_set():
        x = (x * 1.000001 + 0.000001) % 1e6


def workload_idle(duration: float) -> Dict:
    time.sleep(duration)
    return {"metric": "idle", "value": 0.0, "unit": ""}


def workload_cpu_stress(duration: float, cores: int) -> Dict:
    stop = threading.Event()
    threads = [threading.Thread(target=_cpu_burn, args=(stop,), daemon=True) for _ in range(cores)]
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    return {"metric": "cpu_cores", "value": cores, "unit": "cores"}


def workload_gpu_matmul(duration: float) -> Dict:
    if not torch.cuda.is_available():
        return {"metric": "gflops", "value": 0.0, "unit": "GFLOPS/s"}
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        c = a @ b
        torch.cuda.synchronize()
        iters += 1
    dt = time.perf_counter() - t0
    flops = 2 * n * n * n * iters / dt / 1e9
    return {"metric": "gflops", "value": flops, "unit": "GFLOPS/s"}


def workload_llm_generate(duration: float, model_name: str, use_cudagraphs: bool) -> Dict:
    import mmfreelm  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mmfreelm.tensorrt import CUDAGraphAccelerator

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half().eval()
    if use_cudagraphs:
        model = CUDAGraphAccelerator(model)
    prompt = "The quick brown fox jumps over the lazy dog"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    gen_kw = dict(max_length=ids.shape[1] + 24, do_sample=True, top_p=0.4, temperature=0.6)
    tokens = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t0 < duration:
            out = model.generate(ids, **gen_kw)
            tokens += out.shape[1] - ids.shape[1]
    dt = time.perf_counter() - t0
    tps = tokens / dt if dt else 0.0
    label = "cudagraphs" if use_cudagraphs else "pytorch"
    return {"metric": "tps", "value": tps, "unit": "tok/s", "backend": label}


def run_workload(name: str, fn: Callable[[float], Dict], duration: float) -> Dict:
    monitor = TegrastatsMonitor(interval_ms=100)
    clocks_before = read_sysfs_clocks()
    monitor.start()
    time.sleep(0.5)  # settle
    t0 = time.perf_counter()
    metrics = fn(duration)
    dt = time.perf_counter() - t0
    pstats = monitor.stop()
    clocks_after = read_sysfs_clocks()
    n = len(pstats.samples)
    vdd = pstats.mean("vdd_in_w")
    cg = pstats.mean("cpu_gpu_cv_w")
    soc = pstats.mean("soc_w")
    return {
        "workload": name,
        "duration_s": round(dt, 3),
        "samples": n,
        "vdd_in_w": round(vdd, 2),
        "cpu_gpu_cv_w": round(cg, 2),
        "soc_w": round(soc, 2),
        "j_per_sec_vdd": round(vdd * dt, 3),
        "cpu_cur_mhz": round(clocks_after["cpu_cur_khz"] / 1000, 0),
        "gpu_cur_mhz": round(clocks_after["gpu_cur_hz"] / 1e6, 0),
        **metrics,
    }


def print_table(rows: List[Dict], nvpmodel: Dict) -> None:
    caps = NVP_MODEL_CAPS.get(nvpmodel["name"], {})
    print(f"\n{'='*90}")
    print(f"Jetson Power Benchmark — nvpmodel: {nvpmodel['name']} (id={nvpmodel['id']})")
    if caps:
        print(
            f"Mode caps: CPU≤{caps['cpu_max_mhz']}MHz  GPU≤{caps['gpu_max_mhz']}MHz  "
            f"EMC≤{caps['emc_max_mhz']}MHz  budget≈{caps['budget_w']}W"
        )
    clocks = read_sysfs_clocks()
    print(
        f"Active clocks: CPU max={clocks['cpu_max_khz']/1000:.0f}MHz cur={clocks['cpu_cur_khz']/1000:.0f}MHz  "
        f"GPU max={clocks['gpu_max_hz']/1e6:.0f}MHz cur={clocks['gpu_cur_hz']/1e6:.0f}MHz  "
        f"EMC={clocks['emc_hz']/1e6:.0f}MHz"
    )
    print(f"{'='*90}")
    hdr = f"{'Workload':<22} {'Metric':>10} {'VDD_IN':>8} {'CPU/GPU':>8} {'SOC':>7} {'CPU MHz':>8} {'GPU MHz':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        metric = f"{r.get('value', 0):.1f} {r.get('unit', '')}" if r.get("unit") else ""
        if r.get("backend"):
            metric += f" ({r['backend']})"
        print(
            f"{r['workload']:<22} {metric:>10} {r['vdd_in_w']:>7.2f}W {r['cpu_gpu_cv_w']:>7.2f}W "
            f"{r['soc_w']:>6.2f}W {r['cpu_cur_mhz']:>7.0f} {r['gpu_cur_mhz']:>7.0f}"
        )
    print(f"{'='*90}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Jetson power mode / workload benchmark")
    p.add_argument(
        "--workloads",
        nargs="+",
        default=["idle", "cpu_stress", "gpu_matmul", "llm_pytorch", "llm_cudagraphs"],
        choices=["idle", "cpu_stress", "gpu_matmul", "llm_pytorch", "llm_cudagraphs", "all"],
    )
    p.add_argument("--duration", type=float, default=8.0, help="Seconds per workload")
    p.add_argument("--model", default="ridger/MMfreeLM-370M")
    p.add_argument("--cpu-cores", type=int, default=6)
    p.add_argument("--try-switch-mode", type=int, default=None, help="Attempt jtop nvpmodel switch to this ID")
    p.add_argument("--json-out", default=None, help="Write results JSON to this path")
    return p.parse_args()


def main():
    args = parse_args()
    workloads = (
        ["idle", "cpu_stress", "gpu_matmul", "llm_pytorch", "llm_cudagraphs"]
        if "all" in args.workloads
        else args.workloads
    )

    nvpmodel = query_nvpmodel()
    print(f"Current nvpmodel: {nvpmodel['name']} (id={nvpmodel['id']})")

    if args.try_switch_mode is not None:
        print(f"Attempting jtop switch to mode {args.try_switch_mode} …")
        ok = try_jtop_switch(args.try_switch_mode)
        nvpmodel = query_nvpmodel()
        print(f"Switch {'succeeded' if ok else 'FAILED (reboot likely required)'} — now: {nvpmodel['name']}")

    rows = []
    workload_order = {
        "idle": 0,
        "cpu_stress": 1,
        "llm_pytorch": 2,
        "llm_cudagraphs": 3,
        "gpu_matmul": 4,
    }
    for w in sorted(workloads, key=lambda x: workload_order.get(x, 99)):
        print(f"Running {w} ({args.duration}s) …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if w == "idle":
            row = run_workload(w, lambda d: workload_idle(d), args.duration)
        elif w == "cpu_stress":
            row = run_workload(
                w, lambda d: workload_cpu_stress(d, args.cpu_cores), args.duration
            )
        elif w == "gpu_matmul":
            row = run_workload(w, workload_gpu_matmul, args.duration)
        elif w == "llm_pytorch":
            row = run_workload(
                w,
                lambda d: workload_llm_generate(d, args.model, use_cudagraphs=False),
                args.duration,
            )
        elif w == "llm_cudagraphs":
            row = run_workload(
                w,
                lambda d: workload_llm_generate(d, args.model, use_cudagraphs=True),
                args.duration,
            )
        rows.append(row)

    print_table(rows, nvpmodel)

    if args.json_out:
        payload = {"nvpmodel": nvpmodel, "clocks": read_sysfs_clocks(), "results": rows}
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
