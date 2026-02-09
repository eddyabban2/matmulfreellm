"""
Profilings Generation using the pytorch profiler 

Example usage:
    python profile_generate.py -s 32 -m 32 -i 5 -b 32

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from bench_utils import generate_random_input_ids
import transformers
import argparse
import statistics


parser = argparse.ArgumentParser(
    description="profiles generate"
)

parser.add_argument(
    "-s", 
    "--sequence_length",
    default=32,
    help="sets the sequence length of input tokens"
)

parser.add_argument( 
    "-m"
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
    "-b",
    "--batch_size",
    default="32",
    help="Sets the batch size",
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

    num_warmups = 1
    my_schedule = schedule(
        warmup=num_warmups,
        active=num_iterations, 
        repeat=1, 
        wait=0
    )
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
        with_flops=True, record_shapes=True, profile_memory=True, with_stack=True, acc_events=True
    ) as prof:
        for _ in range(num_warmups + num_iterations):
            _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
            )
            prof.step()


    with open("profile_cuda.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total"))

def main():
    sequence_length=int(args.sequence_length)
    iters=int(args.iterations)
    max_new_tokens=int(args.m__max_new_tokens)
    batch_size=int(args.batch_size)

    name = 'ridger/MMfreeLM-2.7B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()

    profile_generation(model, batch_size,sequence_length, iters, max_new_tokens, model_name=name)

if __name__ == "__main__":
    main()
