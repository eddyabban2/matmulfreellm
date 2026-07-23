# does running a model and then saving create a compressed model?import os
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from utils import generate_random_input_ids, generate_dataset_input_ids, get_free_gpu
import transformers
import argparse
import statistics
import csv
import nvtx
import mmfreelm

model_name = 'ridger/MMfreeLM-2.7B'
num_iterations = 1
batch_size = 1
seq_len = 1
max_new_tokens = 1
device = get_free_gpu()
print(f"Device: {device}")
batch = generate_random_input_ids(model_name, batch_size, seq_len)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map=str(device),      # e.g. "cuda:1"
    torch_dtype=torch.float16
    )

_ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
                )

# print(model)
# print(model.state_dict())
# torch.save(model, "uncompressed.pt")

print("inference worked")
