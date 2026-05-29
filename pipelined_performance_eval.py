"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    python generate_csv.py -s 32 --max_new_tokens 32 -i 15 --min_batch_power 0 --max_batch_power 12
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from utils import generate_random_input_ids, generate_dataset_input_ids
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
    "-m", 
    "--micro_batches",
    default=32,
    help="sets the number of micro batches"
)

parser.add_argument(
    "-w", 
    "--weight_multiplier",
    default=1,
    help="sets the number of we multiply the number of weights in each layer by"
)

parser.add_argument(
    "-l", 
    "--layer_multiplier",
    default=1,
    help="sets the number  we multiply the number of layers by"
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


parser.add_argument(
    "--min_batch_power", 
    default=0,
    help="stores the minimum batch power to go up to when profiling",
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


def benchmark_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):
    """Run benchmark with multiple prompts and iterations."""
    
    results = {
            'tps': [],
            'generation_time': [],
            'average_power_watts': [],
            'max_power_watts': [],
            'min_power_watts': [],
            'total_energy_joules': [],
            'energy_per_iteration_joules': [],
            'joules_per_token': []
        }
    batch = None
    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
        batch = generate_random_input_ids(model_name, batch_size, seq_len)

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    power_monitor = PowerMonitor(gpu_indices=[torch.cuda.current_device()])
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    window_key = f"Batch Size {batch_size} Seq Len {seq_len}"

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    for iter in range(num_iterations):
        start_time = time.time()
        torch.cuda.synchronize()
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

        results['generation_time'].append(generation_time)
        results['tps'].append(tps)
        
    row['tokens_per_second'] = statistics.mean(results['tps'])
    row['run_time_seconds'] = statistics.mean(results["generation_time"])

def detailed_runtime_metrics(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):

    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
        batch = generate_random_input_ids(model_name, batch_size, seq_len)

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    end_time = time.time()

    prefill_time = end_time - start_time
    decode_times = []
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(-1)
    with torch.no_grad():
        for i in range(max_new_tokens-1):
                torch.cuda.synchronize()
                start_time = time.time()
                out = model(input_ids=next_tok, past_key_values=past,
                            use_cache=True, return_dict=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(-1)
                torch.cuda.synchronize()
                end_time = time.time()
                decode_times.append(end_time - start_time)
    row["Prefill Time (s)"] = prefill_time
    row["Deocde Time"] = sum(decode_times)
    row["Avg Single Decode Forward Pass (s)"] = statistics.mean(decode_times)

def first_token_time(model, batch_size, seq_len, num_iterations, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):
    """Run benchmark with multiple prompts and iterations."""
    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
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

def profile_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B'):
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
    events = prof.key_averages()
    table_string = prof.key_averages().table().split('\n')
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
    flops = sum(e.flops for e in events) / float(row['run_time_seconds'])
    row["FLOPS"] = flops
    row ["CPU_time_seconds"] = cpu_time
    row["CUDA_time_seconds"] = cuda_time

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
    row['joules_per_token'] = row['energy_per_iteration_joules'] / (batch_size * max_new_tokens)

def create_csv_data(
        sequence_length, 
        max_new_tokens, 
        iters, 
        min_batch_power, 
        max_batch_power, 
        count_micro_batches, 
        weight_multiplier, 
        layer_multiplier):

    first_row = True
    from datetime import datetime
    rank = int(os.environ.get("RANK", 0))
    filename =  f'outputs/csvs/benchmark_results-{date:%Y-%m-%d_%H:%M:%S}Rank:{rank}.csv'.format(date=datetime.now())
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        for model_name in models:
            row = {'device': device, 'model': model_name}
            print(f"Collecting data for model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
            for batch_power in range(min_batch_power, max_batch_power):
                batch_size = 2**batch_power
                row['batch size'] = batch_size
                print(f"\tCollecting data for batch size: {batch_size}")
                print(f"\t\tRunning Benchmarks...")
                # start_time = time.time()
                # benchmark_generation(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=True)
                # end_time = time.time()
                # print(f"\t\t\tBenchmarks completed in {end_time-start_time} sec")

                start_time = time.time()
                detailed_runtime_metrics(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=True)
                end_time = time.time()
                print(f"\t\t\tBenchmarks completed in {end_time-start_time} sec")

                # print(f"\t\tCollecting time to first token data...")
                # start_time = time.time()
                # row['time_to_first_token_sec'] = statistics.mean(first_token_time(model, batch_size, sequence_length, iters, model_name=model_name, use_dataset_prompts=True))
                # end_time = time.time()
                # print(f"\t\t\tTime fo first token completed in {end_time-start_time} sec")

                # print("\t\t Collecting Power data")
                # start_time = time.time()
                # get_power_data(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name)
                # end_time = time.time()
                # print(f"\t\t\tPower Data completed in {end_time-start_time} sec")
                if(first_row):
                    csvwriter = csv.DictWriter(csvfile, row.keys())
                    csvwriter.writeheader()
                    first_row = False
                csvwriter.writerow(row) 
        print(f"Data written to {filename}")

def main():
    sequence_length=int(args.sequence_length)
    max_new_tokens=int(args.max_new_tokens)
    iters=int(args.iterations)
    min_batch_power=int(args.min_batch_power)
    max_batch_power=int(args.max_batch_power)
    count_micro_batches=int(args.micro_batches)
    weight_multiplier=float(args.weight_multiplier)
    layer_multiplier=float(args.layer_multiplier)

    
    create_csv_data()

if __name__ == "__main__":
    main()
