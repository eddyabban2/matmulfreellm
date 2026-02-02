import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench_utils import generate_random_input_ids

def main():
    
    # Original implementation
    name = 'ridger/MMfreeLM-2.7B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
    batch_size = 1
    sequence_length = 1
    # input token is "enough"
    input_token = torch.load('example_input_tensor.pt')
    print(f"Input Tokens: {tokenizer.batch_decode(input_token, skip_special_tokens=True)[0]}")

    output_token = model.generate(input_token.cuda(), max_new_tokens=1, do_sample=False)
    torch.save(output_token, 'example_output_tensor.pt')
    print(f"\nInput:  {input_token}")
    print(f"Output: {tokenizer.batch_decode(output_token, skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()