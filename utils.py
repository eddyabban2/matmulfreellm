import torch
from transformers import AutoTokenizer
from threading import Thread

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