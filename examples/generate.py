import argparse
import os
import statistics
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import mmfreelm  # noqa: F401 — registers custom HF model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "ridger/MMfreeLM-2.7B"
# BitNet-style ternary weights ~1.58 bits vs 16 for FP16 — rough MAC/memory-intensity scaler.
TERNARY_INFO_BITS = 1.58
FP16_STORAGE_BITS = 16.0


def calculate_model_flops(model, num_tokens):
    num_params = sum(p.numel() for p in model.parameters())
    dense_flops = 2 * num_params * num_tokens
    return dense_flops, num_params


def dense_flops_to_ternary_mac_heuristic(dense_flops: float) -> float:
    """Scale dense FLOP proxy toward ternary-linear effective MACs (literature heuristic)."""
    return dense_flops * (TERNARY_INFO_BITS / FP16_STORAGE_BITS)


def _max_tokenized_length(tokenizer, prompts):
    return max(
        tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts
    )


def benchmark_generation(
    model, tokenizer, prompts, max_length=32, num_iterations=5, use_cache=True
):
    results = {
        "tps": [],
        "generation_time": [],
        "tokens_generated": [],
        "gflops_per_sec": [],
        "gflops_ternary_heuristic_per_sec": [],
        "prompt_lengths": [],
    }

    print(f"\n{'='*80}")
    print("RUNNING BENCHMARK")
    print(f"{'='*80}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Iterations per prompt: {num_iterations}")
    print(f"Max generation length: {max_length}")
    print(f"{'='*80}\n")

    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        prompt_length = input_ids.shape[1]

        print(
            f'Prompt {prompt_idx + 1}/{len(prompts)} (length={prompt_length}): "{prompt[:50]}..."'
        )

        gen_kw = dict(
            max_length=max_length,
            do_sample=True,
            top_p=0.4,
            temperature=0.6,
            use_cache=use_cache,
        )

        _ = model.generate(input_ids, **gen_kw)

        for iteration in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model.generate(input_ids, **gen_kw)
            torch.cuda.synchronize()
            end_time = time.time()

            generation_time = end_time - start_time
            output_length = outputs.shape[1]
            tokens_generated = output_length - prompt_length
            tokens_per_second = tokens_generated / generation_time

            total_flops, _ = calculate_model_flops(model, tokens_generated)
            flops_per_second = total_flops / generation_time
            gflops_per_sec = flops_per_second / 1e9
            eff = dense_flops_to_ternary_mac_heuristic(total_flops) / generation_time / 1e9

            results["tps"].append(tokens_per_second)
            results["generation_time"].append(generation_time)
            results["tokens_generated"].append(tokens_generated)
            results["gflops_per_sec"].append(gflops_per_sec)
            results["gflops_ternary_heuristic_per_sec"].append(eff)
            results["prompt_lengths"].append(prompt_length)

            print(
                f"  Iteration {iteration + 1}: {tokens_per_second:.2f} tok/s, "
                f"{gflops_per_sec:.2f} GFLOPS/s (dense θ), {eff:.2f} GFLOPS/s (ternary MAC~), "
                f"{generation_time:.4f}s"
            )

        print()

    return results


def print_benchmark_results(results, model, implementation_type):
    tps_values = results["tps"]
    gflops_values = results["gflops_per_sec"]
    gflops_t = results["gflops_ternary_heuristic_per_sec"]
    time_values = results["generation_time"]
    tokens_values = results["tokens_generated"]

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - {implementation_type}")
    print(f"{'='*80}")
    print("\nModel Configuration:")
    print(f"  Total Parameters: {num_params:,}")
    print(f"  Total Runs: {len(tps_values)}")

    print("\nTokens Per Second (TPS):")
    print(f"  Mean:   {statistics.mean(tps_values):>10.2f} tok/s")
    print(f"  Median: {statistics.median(tps_values):>10.2f} tok/s")
    print(
        f"  Std:    {statistics.stdev(tps_values) if len(tps_values) > 1 else 0:>10.2f} tok/s"
    )
    print(f"  Min:    {min(tps_values):>10.2f} tok/s")
    print(f"  Max:    {max(tps_values):>10.2f} tok/s")

    print("\nCompute Performance (GFLOPS/s) — dense 2|θ| heuristic:")
    print(f"  Mean:   {statistics.mean(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Median: {statistics.median(gflops_values):>10.2f} GFLOPS/s")
    print(
        f"  Std:    {statistics.stdev(gflops_values) if len(gflops_values) > 1 else 0:>10.2f} GFLOPS/s"
    )
    print(f"  Min:    {min(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Max:    {max(gflops_values):>10.2f} GFLOPS/s")

    print(
        "\nTernary-weight MAC heuristic (~1.58 bit/weight vs FP16; scaled dense rate):"
    )
    print(f"  Mean:   {statistics.mean(gflops_t):>10.2f} GFLOPS/s")
    print(f"  Median: {statistics.median(gflops_t):>10.2f} GFLOPS/s")

    print("\nGeneration Time:")
    print(f"  Mean:   {statistics.mean(time_values):>10.4f} s")
    print(f"  Median: {statistics.median(time_values):>10.4f} s")
    print(
        f"  Std:    {statistics.stdev(time_values) if len(time_values) > 1 else 0:>10.4f} s"
    )
    print(f"  Min:    {min(time_values):>10.4f} s")
    print(f"  Max:    {max(time_values):>10.4f} s")

    print("\nTokens Generated:")
    print(f"  Mean:   {statistics.mean(tokens_values):>10.1f}")
    print(f"  Total:  {sum(tokens_values):>10}")

    unique_lengths = sorted(set(results["prompt_lengths"]))
    if len(unique_lengths) > 1:
        print("\nPerformance by Prompt Length:")
        print(
            f"  {'Length':<10} {'Avg TPS':<15} {'Dense GFLOP/s':<14} {'MAC~ GFLOP/s':<14}"
        )
        print(f"  {'-'*10} {'-'*15} {'-'*14} {'-'*14}")
        for length in unique_lengths:
            indices = [i for i, l in enumerate(results["prompt_lengths"]) if l == length]
            avg_tps = statistics.mean([tps_values[i] for i in indices])
            avg_gflops = statistics.mean([gflops_values[i] for i in indices])
            avg_gt = statistics.mean([gflops_t[i] for i in indices])
            print(
                f"  {length:<10} {avg_tps:<15.2f} {avg_gflops:<14.2f} {avg_gt:<14.2f}"
            )

    print(f"\n{'='*80}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="MMfreeLM generation benchmark (PyTorch or TensorRT single-step engine)."
    )
    p.add_argument(
        "--fixed-point",
        action="store_true",
        help="Use fixed-point HGRN (requires generate_integrated.py).",
    )
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Use TensorRT for autoregressive steps (ONNX export + engine).",
    )
    p.add_argument(
        "--rebuild-trt-engine",
        action="store_true",
        help="Delete cached ONNX/engine and rebuild (TensorRT only).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument(
        "--low-cpu-mem",
        action="store_true",
        help="Pass low_cpu_mem_usage=True to from_pretrained (chunked CPU load).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable HF use_cache (saves some state during long generations; HGRN cache is small).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    name = args.model
    use_cache = not args.no_cache
    load_kw = {}
    if args.low_cpu_mem:
        load_kw["low_cpu_mem_usage"] = True

    prompts = [
        "The quick brown fox",
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
        "Once upon a time in a faraway kingdom, there lived a wise old wizard who possessed magical powers beyond imagination",
        "Machine learning is",
        "The future of artificial intelligence will bring transformative changes to society, economy, and daily life in ways we are only beginning",
    ]

    if args.fixed_point:
        print("Using fixed-point HGRN implementation with ternary_matmul operations...")
        from generate_integrated import replace_with_fixed_point_hgrn

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, **load_kw).cuda().half()
        model = replace_with_fixed_point_hgrn(model)
        print("✓ HGRN layers replaced with fixed-point implementation using ternary_matmul")
        implementation_type = "Fixed-Point HGRN"
    else:
        print("Using standard floating-point implementation...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, **load_kw).cuda().half()
        implementation_type = "Floating-Point"

    if not use_cache:
        model.config.use_cache = False

    if args.tensorrt:
        from tensorrt_generation import ONNXTRTAccelerator, trt_dependencies_available

        if not trt_dependencies_available():
            print(
                "ERROR: --tensorrt requires tensorrt (JetPack) and pycuda. "
                "Install pycuda in the same venv, then retry.",
                file=sys.stderr,
            )
            sys.exit(1)
        max_prompt = _max_tokenized_length(tokenizer, prompts)
        max_seq = max_prompt + args.max_length + 16
        model = ONNXTRTAccelerator(
            model,
            max_batch=1,
            max_seq=max_seq,
            model_name=name,
            use_fp16=True,
            rebuild=args.rebuild_trt_engine,
        )
        implementation_type = f"{implementation_type} + TensorRT"

    results = benchmark_generation(
        model,
        tokenizer,
        prompts,
        max_length=args.max_length,
        num_iterations=args.iterations,
        use_cache=use_cache,
    )
    print_benchmark_results(results, model, implementation_type)

    print(f"{'='*80}")
    print("SAMPLE GENERATION")
    print(f"{'='*80}")
    sample_prompt = prompts[1]
    input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(
        input_ids,
        max_length=args.max_length,
        do_sample=True,
        top_p=0.4,
        temperature=0.6,
        use_cache=use_cache,
    )
    print(f"\nInput:  {sample_prompt}")
    print(f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}")

    if args.fixed_point:
        print("\n✓ Benchmark completed using fixed-point HGRN with ternary_matmul operations!")
        print("  Each HGRN cell performed 4 ternary_matmul operations:")
        print("  1. Input gate projection (w_i)")
        print("  2. Forget gate projection (w_f)")
        print("  3. Gate projection (w_g)")
        print("  4. Output projection (w_o)")


if __name__ == "__main__":
    main()
