import torch
from transformers import AutoTokenizer
from threading import Thread
import pandas as pd
import numpy as np

def generate_random_input_ids(model_name, batch_size, sequence_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer.vocab)

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)

    # 3. Generate attention mask (typically all ones for fully valid random inputs)
    # attention_mask shape: (batch_size, sequence_length)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def generate_dataset_input_ids(model_name, batch_size, sequence_length):
    # Load tokenizer only once using caching
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Ensure pad token exists (required for padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Warning: no pad token exists")


    df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
    
    sampled_indices = np.random.choice(len(df), size=batch_size, replace=False)
    prompts = df.iloc[sampled_indices]['text'].to_list()


    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length
    )
    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    has_padding = (attention_mask == 0).any()
    if(has_padding):
        print("Warning: Using Padding because sequence is very long")

    # Move to GPU and return
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return

def main():
    generate_dataset_input_ids("ridger/MMfreeLM-2.7B", 5, 200)

if __name__ == "__main__":
    main()