import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from bench_utils import generate_random_input_ids
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv

parser = argparse.ArgumentParser(
    description="creates a csv file with benchmark results"
)

parser.add_argument(
    "-s", 
    "--sequence_length",
    default=32,
    help="sets the sequence length of input tokens"
)

parser.add_argument( 
    "--max_new_tokens",
    default=32,
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

parser.add_argument(
    "--max_batch_power", 
    default=1,
    help="stores the maximum batch power to go up to when profiling",
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

# 2. Suppress HuggingFace Hub logging (where shard-loading prints originate)
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

def profile_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(num_iterations):
            _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
                )
    return prof

def calculate_model_flops(model, num_tokens):
    """
    Estimate FLOPS for model generation.
    For transformer models: FLOPS ≈ 2 * num_params * num_tokens
    """
    num_params = sum(p.numel() for p in model.parameters())
    # Each parameter is used twice per token (forward multiply-add)
    flops = 2 * num_params * num_tokens
    return flops, num_params

def benchmark_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B'):
    """Run benchmark with multiple prompts and iterations."""
    
    results = {
            'tps': [],
            'generation_time': [],
            'tokens_generated': [],
            'gflops_per_sec': [],
            'prompt_lengths': []
        }
    
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    

    for iter in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
        torch.cuda.synchronize()
        end_time = time.time()

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
    return results

def first_token_time(model, batch_size, seq_len, num_iterations, model_name='ridger/MMfreeLM-2.7B'):
    """Run benchmark with multiple prompts and iterations."""
    
    
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    times = []
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=seq_len+5,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    

    for iter in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
        torch.cuda.synchronize()
        end_time = time.time()
        generation_time = end_time - start_time
        times.append(generation_time)
    return times

def profile_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(num_iterations):
            _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
                )
    return prof

def get_power_data(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    power_monitor = PowerMonitor(gpu_indices=[torch.cuda.current_device()])
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    window_key = f"Batch Size {batch_size} Seq Len {seq_len}"
    start_time = time.time()
    monitor.begin_window(window_key, sync_execution=True)
    for _ in range(num_iterations):
        _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.4,
                temperature=0.6
            )
    print("Waiting for power data to be collected...")
    mes = monitor.end_window(window_key, sync_execution=True)
    end_time = time.time()
    timeline = power_monitor.get_power_timeline(
        power_domain="device_instant",  # or "device_average" or "memory_average"
        gpu_index=0,  # specify GPU, or None for all GPUs
        start_time=start_time,
        end_time=end_time
    )
    for gpu_idx, data in timeline.items():
        powers = [power_watts for timestamp, power_watts in data]
        row['average_power_watts'] = sum(powers) / len(powers)
        row['max_power_watts'] = max(powers)
        row['min_power_watts'] = min(powers)
    row['total_energy_joules'] = mes.gpu_energy[0]
    row['energy_per_iteration_joules'] = mes.gpu_energy[0] / num_iterations
    row['joules_per_token'] = row['energy_per_iteration_joules'] / (batch_size * max_new_tokens * num_iterations)

def create_csv_data(model, sequence_length, iters, max_new_tokens):
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Collecting Data to be used in a CSV")
    first_row = True

    max_batch_power = int(args.max_batch_power)
    from datetime import datetime
    filename =  'benchmark_results-{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.now() )
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        for batch_power in range(max_batch_power):
            batch_size = 2**batch_power
            row = {'device': device, 'batch size': batch_size}
            print(f"Collecting data for batch size: {batch_size}")
            benchmark_results = benchmark_generation(model, batch_size, sequence_length, iters, max_new_tokens, row)
            profile_results = profile_generation(model, batch_size, sequence_length, iters, max_new_tokens)
            time_to_first_token = statistics.mean(first_token_time(model, batch_size, sequence_length, iters))
            get_power_data(model, batch_size, sequence_length, iters, max_new_tokens, row)
            print("\tCalculating Metrics")
            tps = statistics.mean(benchmark_results['tps'])
            run_time = statistics.mean(benchmark_results['generation_time'])
            events = profile_results.key_averages()
            table_string = profile_results.key_averages().table().split('\n')
            cpu_time = table_string[-3].split()[4]
            if cpu_time.endswith('ms'):
                cpu_time = float(cpu_time[:-2]) / 1e3
            else:
                cpu_time = float(cpu_time[:-1])
            cuda_time = table_string[-2].split()[4]
            if cuda_time.endswith('ms'):
                cuda_time = float(cuda_time[:-2]) / 1e3
            else:
                cuda_time = float(cuda_time[:-1])
            flops = sum(e.flops for e in events) / run_time
            # data=[device, batch_size, tps, run_time, cuda_time, cpu_time, time_to_first_token, flops]
            # for key, value in mydict.items():
            #     writer.writerow([key, value])
            if(first_row):
                csvwriter = csv.DictWriter(csvfile, row.keys())
                csvwriter.writeheader()
                first_row = False
            csvwriter.writerow(row) 
        print(f"Data written to {filename}")

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

    sequence_length=int(args.sequence_length)
    iters=int(args.iterations)
    max_new_tokens=int(args.max_new_tokens)
    
    create_csv_data(model, sequence_length, iters, max_new_tokens)

if __name__ == "__main__":
    main()
