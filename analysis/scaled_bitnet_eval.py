"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    python scaled_bitnet_eval.py -s 32 --max_new_tokens 32 -i 15 --min_batch_power 0 --max_batch_power 2 \
        --hidden_size 2560 --intermediate_size 6912 --max_position_embeddings 4096 --num_attention_heads 20 \
        --num_hidden_layers 40 --num_key_value_heads 5 --vocab_size 128256
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import gc
from transformers import logging
import argparse
import csv
import transformers.integrations.bitnet as bitnet
import bitnet as local_bitnet
from scaled_bitnet import create_custom_bitnet, standard_model_config
from generate_csv import benchmark_generation, detailed_runtime_metrics, run_warmup

bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear

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
    default=20,
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

parser.add_argument(
    "--print_csv",
    action='store_true',
    default=False,
    help="prints csv after creating data"
)

parser.add_argument(
    "--print_model",
    action='store_true',
    default=False,
    help="prints the model"
)

parser.add_argument(
    "--hidden_size", 
    default=2560,
    help="sets the hidden size of the model"
)

parser.add_argument(
    "--intermediate_size", 
    default=6912,
    help="sets the intermediate size of the model"
)

parser.add_argument(
    "--max_position_embeddings", 
    default=4096,
    help="sets the max position embeddings of the model"
)

parser.add_argument(
    "--num_attention_heads", 
    default=20,
    help="sets the number of attention heads of the model"
)

parser.add_argument(
    "--num_hidden_layers", 
    default=30,
    help="sets the number of hidden layer of the model"
)

parser.add_argument(
    "--num_key_value_heads", 
    default=5,
    help="sets the number of key value heads of the model"
)

parser.add_argument(
    "--vocab_size", 
    default=128256,
    help="sets the vocab size of the model"
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

# 2. Suppress HuggingFace Hub logging (where shard-loading prints originate)
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

def create_csv_data(sequence_length, iters, max_new_tokens, model_config):
    print("using this create csv func")
    model_name = "microsoft/bitnet-b1.58-2B-4T"
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Collecting Data to be used in a CSV")
    first_row = True
    min_batch_power = int(args.min_batch_power)
    max_batch_power = int(args.max_batch_power)
    from datetime import datetime
    filename =  'outputs/csvs/scaled_bitnet_results-{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.now() )
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        row = {'Device': device, 'Model': "Scaled Up Bitnet"}
        print(f"Collecting data for model: {model_name}")
        model = create_custom_bitnet(model_config=model_config)
        row['Total Parameters'] = f"{model.total_params:,}"
        row['Total Trainable Parameters'] = f"{model.trainable_params:,}"
        if args.print_model: 
            print(model)
        print("model loaded\nRunning warmup")
        run_warmup(model, model_name)
        print("finished warmup")
        gc.collect()
        torch.cuda.empty_cache()
        memory_usage = torch.cuda.memory_allocated()
        row['Memory Usage (GiB'] = memory_usage/(1024**3)
        for batch_power in reversed(range(min_batch_power, max_batch_power)):
            batch_size = 2**batch_power
            row['batch size'] = batch_size
            print(f"\tCollecting data for batch size: {batch_size}")
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


        print(f"Data written to {filename}")
    if args.print_csv:
        with open(filename, "r") as file:
            print(file.read())

def main():
    model_config = standard_model_config 
    model_config.hidden_size = int(args.hidden_size)
    model_config.intermediate_size = int(args.intermediate_size)
    model_config.max_position_embeddings = int(args.max_position_embeddings)
    model_config.num_attention_heads = int(args.num_attention_heads)
    model_config.num_hidden_layers = int(args.num_hidden_layers)
    model_config.num_key_value_heads = int(args.num_key_value_heads)
    model_config.vocab_size = int(args.vocab_size)
    print("running this test")
    sequence_length=int(args.sequence_length)
    iters=int(args.iterations)
    max_new_tokens=int(args.max_new_tokens)
    
    create_csv_data(sequence_length, iters, max_new_tokens, model_config)
print("testing")
if __name__ == "__main__":
    main()
