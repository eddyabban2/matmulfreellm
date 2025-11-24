import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import argparse

parser = argparse.ArgumentParser(
    description="performs Batched Generation"
)

# this is used to prevent the calculation of metrics and print statements
# when evaluating which kernal is taking the most amount of time
parser.add_argument(
    "-m",
    "--metrics",
    action="store_true",
    help="Produces Metrics",
)

parser.add_argument(
    "-b", 
    "--batch_size",
    default=64,
    help="sets the batch size"
)

parser.add_argument(
    "-s", 
    "--sequence_length",
    default=10,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-i", 
    "--iterations",
    default=5,
    help="Determines the number of iterations to benchmark for"
)

parser.add_argument (
    "-f",
    "--fixed_point",
    action="store_true",
    help="Switches the model to fixed point",
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

def generate_random_input_ids(model_name, batch_size, sequence_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer.vocab)

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)

    # 3. Generate attention mask (typically all ones for fully valid random inputs)
    # attention_mask shape: (batch_size, sequence_length)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def benchmark_generation(model, batch_size, seq_len, num_iterations, max_length=32, model_name='ridger/MMfreeLM-2.7B'):
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
        print(f"Sequence_length: {seq_len}")
        print(f"Iterations: {num_iterations}")
        print(f"Max generation length: {max_length}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*80}\n")
    
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    

    for iter in range(num_iterations):
        if args.metrics:
            torch.cuda.synchronize()
            start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
        if args.metrics:
            torch.cuda.synchronize()
            end_time = time.time()

        # calculate metrics
        if args.metrics:
            generation_time = end_time - start_time
            tokens_generated = (outputs.shape[1] - seq_len)*batch_size
            tps = (tokens_generated) / generation_time

            total_flops, _ = calculate_model_flops(model, tokens_generated)
            flops_per_second = total_flops / generation_time
            gflops_per_sec = flops_per_second / 1e9

            results['generation_time'].append(generation_time)
            results['tokens_generated'].append(tokens_generated)
            results['tps'].append(tps)
            results['gflops_per_sec'].append(gflops_per_sec)
            results['prompt_lengths'].append(seq_len)
            print(f"  Iteration {iter + 1}: {tps:.2f} tok/s, {gflops_per_sec:.2f} GFLOPS/s, {generation_time:.4f}s")
    if args.metrics:
        return results
    return None

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
    name = 'ridger/MMfreeLM-2.7B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
    if args.fixed_point:
        from generate_integrated import replace_with_fixed_point_hgrn
        # Replace HGRN layers with fixed-point implementation
        model = replace_with_fixed_point_hgrn(model)
        implementation_type = "Fixed-Point HGRN"
        print("fixed point is currently broken exiting")
        quit()

    else:
        implementation_type = "Floating-Point"

    batch_size=int(args.batch_size)
    sequence_length=int(args.sequence_length)
    iter=int(args.iterations)
    # Run benchmark
    results = benchmark_generation(model, batch_size, sequence_length, iter)

    # Print results
    if(args.metrics):
        print_benchmark_results(results, model, implementation_type)

if __name__ == "__main__":
    main()