import os
import torch
import torch.distributed as dist
import torch.nn as nn

# Assuming mmfreelm or matmulfreellm package is installed and imported
import sys
# from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append('..')
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from utils import generate_dataset_input_ids, create_input_ids_from_text, create_string_from_tokens

class PipelineParallelMatMulFreeInference:
    def __init__(self, model_id="ridger/MMfreeLM-2.7B"):
        # 1. Initialize Distributed Environment
        # os.environ['RANK'] = '0'
        # os.environ['WORLD_SIZE'] = '2'
        # os.environ['MASTER_ADDR'] = '2'
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            
        self.device = torch.device(f"cuda:{self.rank}")
        
        # 2. Load the full configuration to understand model depth
        self.config = HGRNBitConfig.from_pretrained(model_id)
        self.num_layers = self.config.num_hidden_layers
        
        # 3. Calculate layer allocation per GPU
        # Simple balanced splitting strategy
        layers_per_gpu = self.num_layers // self.world_size
        remainder = self.num_layers % self.world_size
        
        # Assign blocks of layers to specific ranks
        self.layer_start = self.rank * layers_per_gpu + min(self.rank, remainder)
        self.layer_end = self.layer_start + layers_per_gpu + (1 if self.rank < remainder else 0)
        
        print(f"Rank {self.rank}: Managing layers {self.layer_start} to {self.layer_end - 1}")
        
        # 4. Sharded Layer Loading
        # To avoid OOMing a single machine loading a huge model, we load the meta or CPU structure
        # and only assign the required weights to our local GPU.
        # For simplicity in this script, we initialize the full model structural skeleton on CPU 
        # and extract only our shard.
        full_model = HGRNBitForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        
        # Extract components assigned to this rank
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        
        if self.rank == 0:
            self.embed_tokens = full_model.model.embeddings.to(self.device)
            
        if self.rank == self.world_size - 1:
            self.norm = full_model.model.norm.to(self.device)
            self.lm_head = full_model.lm_head.to(self.device)
            
        # Extract our slice of MatMul-free layers (HGRNBit layers)
        self.local_layers = nn.ModuleList([
            full_model.model.layers[i].to(self.device) 
            for i in range(self.layer_start, self.layer_end)
        ])
        
        # Clean up the rest of the model structure from memory
        del full_model
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def forward_step(self, input_ids=None, hidden_states=None):
        """
        Executes a single pipeline step for inference.
        """
        # Rank 0 handles the embedding layer
        if self.rank == 0:
            assert input_ids is not None, "Rank 0 requires input_ids"
            hidden_states = self.embed_tokens(input_ids)
        else:
            # Other ranks expect to receive hidden_states from the previous rank
            # Allocate space dynamically or use static sizes (batch, seq_len, hidden_dim)
            # For autoregressive generation, seq_len is usually 1 during decoding steps
            # You must communicate or ensure shape agreements between stages
            # Assuming shape tensor metadata is passed or static
            pass 

        # --- Pipeline Receive Communication ---
        if self.rank > 0:
            # Receive activation size shape metadata first if dynamic
            shape_tensor = torch.zeros(3, dtype=torch.int64, device=self.device)
            dist.recv(shape_tensor, src=self.rank - 1)
            
            # Allocate and receive hidden_states
            hidden_states = torch.zeros(tuple(shape_tensor.tolist()), dtype=torch.float16, device=self.device)
            dist.recv(hidden_states, src=self.rank - 1)

        # --- Local Computation ---
        # Execute the assigned slice of MatMul-free/BitLinear layers
        for layer in self.local_layers:
            # Note: If passing persistent recurrent states for generation, 
            # layer outputs may look like: hidden_states, recurrent_state = layer(hidden_states, recurrent_state)
            # Ensure your state lists are tracked inside your loop if doing token-by-token generation.
            outputs = layer(hidden_states)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # --- Pipeline Send Communication ---
        if self.rank < self.world_size - 1:
            # Send tensor shape metadata first
            shape_tensor = torch.tensor(list(hidden_states.shape), dtype=torch.int64, device=self.device)
            dist.send(shape_tensor, dst=self.rank + 1)
            # Send the hidden states matrix
            dist.send(hidden_states, dst=self.rank + 1)
            return None
        else:
            # The Final Rank normalizes and projects to vocabulary space
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=20):
        """
        Coordinated text generation across pipeline stages.
        """
        current_input_ids = input_ids.to(self.device) if self.rank == 0 else None
        generated_tokens = []
        batch_size = input_ids.shape[0]

        for step in range(max_new_tokens):
            # Run one forward pass through the pipeline
            logits = self.forward_step(input_ids=current_input_ids)
            next_token = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
            if self.rank == self.world_size - 1:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            dist.broadcast(next_token, src=self.world_size - 1)
            
            # Setup for next autoregressive step
            current_input_ids = next_token if self.rank == 0 else None
            generated_tokens.append(next_token.cpu())
            
        return torch.cat(generated_tokens, dim=1)

# --- Orchestration Trigger ---
if __name__ == "__main__":
    # Replace with your local or Hugging Face repository pointer for MatMulFree LLM
    MODEL_ID = "ridger/MMfreeLM-2.7B"
    
    pipeline_model = PipelineParallelMatMulFreeInference(MODEL_ID)    
    batch_size = 2
    sequence_length = 10
    # Input example on Rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        # Simple dummy input token representation [Batch=1, Seq=4]
        input_tokens = generate_dataset_input_ids(MODEL_ID, batch_size, sequence_length)["input_ids"]
        print(input_tokens)
    else:
        input_tokens = torch.zeros((batch_size, sequence_length), dtype=torch.long) # placeholders for other ranks
        
    # Synchronize all processes before starting inference loop
    dist.barrier()
    
    output = pipeline_model.generate(input_tokens, max_new_tokens=10)
    
    if int(os.environ.get("RANK", 0)) == 0:
        print("Generated Token IDs sequence:", output)
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"Input: {create_string_from_tokens(MODEL_ID, input_tokens)}\nOutput: {create_string_from_tokens(MODEL_ID, output)}")