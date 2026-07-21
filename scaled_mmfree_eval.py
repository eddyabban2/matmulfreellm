"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    python scaled_mmfree_eval.py -s 32 --max_new_tokens 32 -i 5 --min_batch_power 0 --max_batch_power 2 \
        --weight_multiplier 1 --layers_multiplier 1 --vocab_multiplier 1
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import gc
from transformers import logging
import argparse
import csv
from scaled_mmfree import create_scaled_mmfree
from generate_csv import benchmark_generation, detailed_runtime_metrics, run_warmup

parser = argparse.ArgumentParser(
    description="creates a csv file with benchmark results"
)


parser.add_argument(
    "-s", 
    "--sequence_length",
    default=5,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-w", 
    "--weight_multiplier",
    default=0.5,
    help="sets the number of we multiply the number of weights in each layer by"
)

parser.add_argument(
    "-l", 
    "--layers_multiplier",
    default=0.5,
    help="sets the number  we multiply the number of layers by"
)

parser.add_argument(
    "-v", 
    "--vocab_multiplier",
    default=1.5,
    help="sets the number  we multiply the number of layers by"
)

parser.add_argument( 
    "--max_new_tokens",
    default=20,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-i", 
    "--iterations",
    default=1,
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

parser.add_argument(
    "--print_csv",
    action='store_true',
    default=False,
    help="prints csv after creating data"
)

parser.add_argument(
    "--disable_compress_weights",
    action='store_true',
    default=False,
    help="disables weight compression"
)

parser.add_argument(
    "--disable_uncompress_weights",
    action='store_true',
    default=False,
    help="performs a run without weight compression"
)

parser.add_argument(
    "--print_model",
    action='store_true',
    default=False,
    help="prints the model"
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

def create_csv_data(
        sequence_length, 
        iters, 
        max_new_tokens, 
        weight_multiplier, 
        layers_multiplier, 
        vocab_multiplier, 
        weight_compression_settings,):
    model_name = "ridger/MMfreeLM-2.7B"
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Collecting Data to be used in a CSV")
    first_row = True
    min_batch_power = int(args.min_batch_power)
    max_batch_power = int(args.max_batch_power)
    from datetime import datetime
    filename =  'outputs/csvs/scaled_mmfree_single_gpu_results-{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.now() )
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        row = {
            'Device': device, 
            'Model': "Scaled Up mmfree single GPU", 
            "weight multiplier": weight_multiplier,
            "layers_multiplier": layers_multiplier, 
            "vocab_multiplier": vocab_multiplier
            }
        for weight_compression in weight_compression_settings:
            print(f"\tCollecting data for scaled mmfree with compression status {weight_compression}")
            model = create_scaled_mmfree(
                layers_multiplier=layers_multiplier, 
                weight_multiplier=weight_multiplier, 
                vocab_size_multiplier=vocab_multiplier, 
                print_model_config=args.print_model, 
                weight_compression=weight_compression)

            print("\tmodel loaded\n\tRunning warmup")
            run_warmup(model, model_name)
            print("\tfinished warmup")
            gc.collect()
            torch.cuda.empty_cache()
            memory_usage = torch.cuda.memory_allocated()
            row['Memory Usage (GiB)'] = memory_usage/(1024**3)
            print(torch.cuda.memory_summary())
            print(f"memory usage: {memory_usage/(1024**3)}")
            row["Weight Compression"] = weight_compression

            for batch_power in reversed(range(min_batch_power, max_batch_power)):
                batch_size = 2**batch_power
                row['batch size'] = batch_size
                print(f"\t\tCollecting data for batch size: {batch_size}")
                print(f"\t\tRunning Benchmarks...")
                start_time = time.time()
                benchmark_generation(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=False)
                end_time = time.time()
                print(f"\t\t\tBenchmarks completed in {end_time-start_time} sec")

                start_time = time.time()
                detailed_runtime_metrics(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=False)
                end_time = time.time()
                print(f"\t\t\tPrefill and Decode Times completed in {end_time-start_time} sec")
                if(first_row):
                    csvwriter = csv.DictWriter(csvfile, row.keys())
                    csvwriter.writeheader()
                    first_row = False
                csvwriter.writerow(row) 
            del model
            gc.collect()
            torch.cuda.empty_cache()
            


        print(f"Data written to {filename}")
    if args.print_csv:
        with open(filename, "r") as file:
            print(file.read())

def main():
    sequence_length=int(args.sequence_length)
    iters=int(args.iterations)
    max_new_tokens=int(args.max_new_tokens)
    weight_multiplier=float(args.weight_multiplier)
    layers_multiplier=float(args.layers_multiplier)
    vocab_multiplier=float(args.vocab_multiplier)
    weight_compression_settings = [False, True]
    if args.disable_compress_weights:
        weight_compression_settings.remove(True)
    if args.disable_uncompress_weights:
        weight_compression_settings.remove(False)
    
    create_csv_data(sequence_length, iters, max_new_tokens, weight_multiplier, layers_multiplier, vocab_multiplier, weight_compression_settings)
if __name__ == "__main__":
    main()
