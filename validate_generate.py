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
    name = 'ridger/MMfreeLM-370M'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
    batch_size = 1
    sequence_length = 1
    example_input_token = torch.load('example_input_tensor.pt')
    print(f"Input Tokens: {tokenizer.batch_decode(example_input_token, skip_special_tokens=True)[0]}")

    current_output_token = model.generate(example_input_token.cuda(), max_new_tokens=1, do_sample=False)
    expected_output_token = torch.load('example_output_tensor.pt')

    if torch.equal(current_output_token, expected_output_token):
        print("Output Tensor Matches expected input tensor")
    else: 
        print("Output Tensor does not match expected input tensor")
    print(f"current output token {current_output_token}")
    print(f"current output token {example_input_token}")
    print(f"\nInput:  {tokenizer.batch_decode(example_input_token, skip_special_tokens=True)[0]}")
    print(f"Current Output: {tokenizer.batch_decode(current_output_token, skip_special_tokens=True)[0]}")
    print(f"Expected Output: {tokenizer.batch_decode(expected_output_token, skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()