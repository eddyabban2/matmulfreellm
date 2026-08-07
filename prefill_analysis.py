import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from utils import  generate_dataset_input_ids
import argparse
import statistics
import csv
import nvtx
import mmfreelm
from datetime import datetime

parser = argparse.ArgumentParser(
    description="performs Batched Generation"
)

parser.add_argument(
    "-s",
    "--seq_len",
    default=161,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-n",
    "--max_new_tokens",
    default=338,
    help="sets the number of new tokens to be generated"
)

parser.add_argument(
    "-i",
    "--iterations",
    default=5,
    help="Determines the number of iterations to benchmark for"
)

parser.add_argument(
    "--model_name",
    default='ridger/MMfreeLM-2.7B',
    help="sets the model name to be used"
)

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

# 2. Suppress HuggingFace Hub logging (where shard-loading prints originate)
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

args = parser.parse_args()

def run_workload(model_name, batch_size, seq_len, max_new_tokens, num_iterations, model):
    batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    prefill_times = []
    decode_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        end_time = time.time()
        prefill_time = end_time - start_time
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)
        start_time = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            for i in range(max_new_tokens-1):
                with nvtx.annotate(f"decodingStep{i}", color="cyan"):
                    out = model(input_ids=next_tok, past_key_values=past,
                                use_cache=True, return_dict=True)
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:, :].argmax(-1)
        torch.cuda.synchronize()
        end_time = time.time()
        decode_time = end_time - start_time
        
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
    return statistics.mean(prefill_times), statistics.mean(decode_times)

def main():
    model_name = args.model_name
    num_iterations = int(args.iterations)
    seq_len = int(args.seq_len)
    max_new_tokens = int(args.max_new_tokens)

    warmup_inputs = generate_dataset_input_ids(model_name, 5, seq_len)
    input_ids = warmup_inputs["input_ids"].cuda()
    attention_mask = warmup_inputs["attention_mask"].cuda()

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()

    print("warmup running")
    # run a warm up generate
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)

    print("warmup finished")
    filename =  'outputs/csvs/prefill_analysis-{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.now())
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    row = {}
    row["Model Name"] = model_name
    row["Sequence Length"] = seq_len
    row["Max New Tokens"] = max_new_tokens
    row["Device"] = device
    first_row = True
    csvwriter = None
    with open(filename, 'w') as csvfile:
        for batch_power in range(11):
            batch_size = 2**batch_power
            row["Batch Size"] = batch_size
            prefill_time, decode_time = run_workload(model_name, batch_size, seq_len, max_new_tokens, num_iterations, model)
            print(f"prefill_time: {prefill_time} decode_time: {decode_time}")
            row["Prefill Time (s)"] = prefill_time
            row["Decode Time (s)"] = decode_time

            if first_row: 
                csvwriter = csv.DictWriter(csvfile, row.keys())
                csvwriter.writeheader()
                first_row = False
            csvwriter.writerow(row) 
        

    print("inference worked")

if __name__ == "__main__":
    main()

