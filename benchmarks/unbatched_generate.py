import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import argparse

parser = argparse.ArgumentParser(
    description="performs generation using ridgers model"
)

# this is used to prevent the calculation of metrics and print statements
# when evaluating which kernal is taking the most amount of time
parser.add_argument(
    "-m",
    "--metrics",
    action="store_true",
    help="Produces Metrics",
)

args = parser.parse_args()

if not args.metrics:
    logging.set_verbosity_error()
    logging.disable_default_handler()
    logging.disable_propagation()

    # 2. Suppress HuggingFace Hub logging (where shard-loading prints originate)
    logging.set_verbosity_error()
    logging.disable_default_handler()
    logging.disable_propagation()

def calculate_model_flops(model, num_tokens):
    """
    Estimate FLOPS for model generation.
    For transformer models: FLOPS ≈ 2 * num_params * num_tokens
    """
    num_params = sum(p.numel() for p in model.parameters())
    # Each parameter is used twice per token (forward multiply-add)
    flops = 2 * num_params * num_tokens
    return flops, num_params


def benchmark_generation(model, tokenizer, prompts, max_length=32, num_iterations=5):
    """Run benchmark with multiple prompts and iterations."""
    results = {
        'tps': [],
        'generation_time': [],
        'tokens_generated': [],
        'gflops_per_sec': [],
        'prompt_lengths': []
    }
    if args.metrics:
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK")
        print(f"{'='*80}")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Iterations per prompt: {num_iterations}")
        print(f"Max generation length: {max_length}")
        print(f"{'='*80}\n")

    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        prompt_length = input_ids.shape[1]
        if(args.metrics):
            print(f"Prompt {prompt_idx + 1}/{len(prompts)} (length={prompt_length}): \"{prompt[:50]}...\"")

        # Warm-up
        _ = model.generate(input_ids, max_length=max_length, top_p=0.4, do_sample=True, temperature=0.6)

        # Benchmark iterations
        for iteration in range(num_iterations):
            if args.metrics:
                torch.cuda.synchronize()
                start_time = time.time()
            outputs = model.generate(input_ids, max_length=max_length, top_p=0.4,  do_sample=True, temperature=0.6)
            if args.metrics:
                torch.cuda.synchronize()
                end_time = time.time()
                # Calculate metrics
                generation_time = end_time - start_time
                output_length = outputs.shape[1]
                tokens_generated = output_length - prompt_length
                tokens_per_second = tokens_generated / generation_time

                # Calculate FLOPS
                total_flops, _ = calculate_model_flops(model, tokens_generated)
                flops_per_second = total_flops / generation_time
                gflops_per_sec = flops_per_second / 1e9

                results['tps'].append(tokens_per_second)
                results['generation_time'].append(generation_time)
                results['tokens_generated'].append(tokens_generated)
                results['gflops_per_sec'].append(gflops_per_sec)
                results['prompt_lengths'].append(prompt_length)

                print(f"  Iteration {iteration + 1}: {tokens_per_second:.2f} tok/s, {gflops_per_sec:.2f} GFLOPS/s, {generation_time:.4f}s")
        if(args.metrics):  
            print()
    
    return results

def print_benchmark_results(results, model, implementation_type):
    """Print comprehensive benchmark statistics."""
    import statistics

    # Convert to tensors for easier calculation
    tps_values = results['tps']
    gflops_values = results['gflops_per_sec']
    time_values = results['generation_time']
    tokens_values = results['tokens_generated']

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - {implementation_type}")
    print(f"{'='*80}")
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {num_params:,}")
    print(f"  Total Runs: {len(tps_values)}")

    print(f"\nTokens Per Second (TPS):")
    print(f"  Mean:   {statistics.mean(tps_values):>10.2f} tok/s")
    print(f"  Median: {statistics.median(tps_values):>10.2f} tok/s")
    print(f"  Std:    {statistics.stdev(tps_values) if len(tps_values) > 1 else 0:>10.2f} tok/s")
    print(f"  Min:    {min(tps_values):>10.2f} tok/s")
    print(f"  Max:    {max(tps_values):>10.2f} tok/s")

    print(f"\nCompute Performance (GFLOPS/s):")
    print(f"  Mean:   {statistics.mean(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Median: {statistics.median(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Std:    {statistics.stdev(gflops_values) if len(gflops_values) > 1 else 0:>10.2f} GFLOPS/s")
    print(f"  Min:    {min(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Max:    {max(gflops_values):>10.2f} GFLOPS/s")

    print(f"\nGeneration Time:")
    print(f"  Mean:   {statistics.mean(time_values):>10.4f} s")
    print(f"  Median: {statistics.median(time_values):>10.4f} s")
    print(f"  Std:    {statistics.stdev(time_values) if len(time_values) > 1 else 0:>10.4f} s")
    print(f"  Min:    {min(time_values):>10.4f} s")
    print(f"  Max:    {max(time_values):>10.4f} s")

    print(f"\nTokens Generated:")
    print(f"  Mean:   {statistics.mean(tokens_values):>10.1f}")
    print(f"  Total:  {sum(tokens_values):>10}")

    # Breakdown by prompt length
    unique_lengths = sorted(set(results['prompt_lengths']))
    if len(unique_lengths) > 1:
        print(f"\nPerformance by Prompt Length:")
        print(f"  {'Length':<10} {'Avg TPS':<15} {'Avg GFLOPS/s':<15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15}")
        for length in unique_lengths:
            indices = [i for i, l in enumerate(results['prompt_lengths']) if l == length]
            avg_tps = statistics.mean([tps_values[i] for i in indices])
            avg_gflops = statistics.mean([gflops_values[i] for i in indices])
            print(f"  {length:<10} {avg_tps:<15.2f} {avg_gflops:<15.2f}")

    print(f"\n{'='*80}\n")

def main():

    # Benchmark prompts with varying lengths
    prompts = [
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
    ]

    name = 'ridger/MMfreeLM-2.7B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()

    implementation_type = "Floating-Point"

    # Run benchmark
    results = benchmark_generation(model, tokenizer, prompts, max_length=256, num_iterations=1)

    # Print results
    if(args.metrics):
        print_benchmark_results(results, model, implementation_type)

        # Show a sample generation
        print(f"{'='*80}")
        print(f"SAMPLE GENERATION")
        print(f"{'='*80}")
        sample_prompt = prompts[0]
        input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids, max_length=256, top_p=0.4, temperature=0.6)
        print(f"\nInput:  {sample_prompt}")
        print(f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()