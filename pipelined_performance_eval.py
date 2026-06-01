"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    torchrun --nproc_per_node=2 pipelined_performance_eval.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv
from utils import generate_random_input_ids, generate_dataset_input_ids
from pipeline_mmfreelm import PipelineParallelMatMulFreeLM

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
    "--layers_multiplier",
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


def benchmark_generation(pipelined_model, batch_size, seq_len, num_iterations, max_new_tokens, row, num_micro_batches, rank, use_dataset_prompts=False):
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
    micro_batches = []
    generate_function = generate_random_input_ids
    if use_dataset_prompts: 
        generate_function = generate_dataset_input_ids
    if rank == 0: 
        for _ in range(num_micro_batches):
            inputs = generate_function(MODEL_ID, batch_size, seq_len)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                })
    
    dist.barrier()

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



def create_csv_data(
        sequence_length, 
        max_new_tokens, 
        iters, 
        min_batch_power, 
        max_batch_power, 
        count_micro_batches, 
        weight_multiplier, 
        layers_multiplier):

    first_row = True
    rank = int(os.environ.get("RANK", 0))
    from datetime import datetime
    filename =  'outputs/csvs/benchmark_results-{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.now())
    pipelined_model = PipelineParallelMatMulFreeLM(weight_multiplier=weight_multiplier, layers_multiplier=layers_multiplier)  
    device = pipelined_model.device
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        original_hidden_layer_size = 2560
        original_num_layers = 32
        row = {
            'device': device, 
            'Hiden Layer Size': original_hidden_layer_size*weight_multiplier, 
            'Number of Layers': original_num_layers*layers_multiplier}
        for batch_power in range(min_batch_power, max_batch_power):
            batch_size = 2**batch_power
            row['Batch Size'] = batch_size
            print(f"\tCollecting data for batch size: {batch_size}")
            print(f"\t\tRunning Benchmarks...")

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
    layers_multiplier=float(args.layers_multiplier)

    create_csv_data(sequence_length, max_new_tokens, iters, min_batch_power, max_batch_power, count_micro_batches, weight_multiplier, layers_multiplier)

if __name__ == "__main__":
    main()
