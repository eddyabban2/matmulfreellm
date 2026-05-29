import os
import torch
import torch.distributed as dist
import torch.nn as nn
import nvtx
from datetime import timedelta
import sys
# from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append('..')
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from utils import generate_dataset_input_ids, create_string_from_tokens
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "ridger/MMfreeLM-2.7B"

full_model = HGRNBitForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

original_embedding = full_model.model.embeddings.to(device)

weight_multiplier = 2 
hidden_size = 2560*weight_multiplier
created_embeddings = nn.Embedding(
                    num_embeddings=full_model.vocab_size, 
                    embedding_dim=hidden_size, 
                    padding_idx=full_model.model.padding_idx, 
                    device=device)

print("full model")
print(full_model)
print("original embeddings")
print(original_embedding)
print("created embeddings")
print(created_embeddings)

input_tensor = torch.randint(
    0, 
    10, 
    (32000, ),
    dtype=torch.int32,
    device=device
)
for _ in range(5):
    original_embeddings_output = original_embedding(input_tensor)
    created_embeddings_output = created_embeddings(input_tensor)

    print(f"original_embeddings_output: {original_embeddings_output}")
    print(f"original shape: {original_embeddings_output.shape}")
    print(f"created_embeddings_output: {created_embeddings_output}")
    print(f"created_embeddings_output shape: {created_embeddings_output.shape}")

