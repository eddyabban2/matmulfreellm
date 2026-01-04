import torch
from transformers import AutoTokenizer

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