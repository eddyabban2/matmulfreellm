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
    input_token = generate_random_input_ids(name, batch_size, sequence_length)["input_ids"]
    print((input_token))
    torch.save(input_token, 'example_input_tensor.pt')

    output_token = model.generate(input_token.cuda(), max_new_tokens=1, do_sample=False)
    torch.save(output_token, 'example_output_tensor.pt')
    
    print(f"Input Tokens: {tokenizer.batch_decode(input_token, skip_special_tokens=True)[0]}")
    print(f"Output: {tokenizer.batch_decode(output_token, skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()