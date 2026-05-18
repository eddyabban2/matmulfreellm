# Example Usage:
#   torchrun --nproc_per_node=2 pipeline_mmfreelm.py 

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import nvtx
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from utils import generate_dataset_input_ids, create_input_ids_from_text, create_string_from_tokens

class PipelineParallelMatMulFreeInference:
    def __init__(self, model_id="ridger/MMfreeLM-2.7B"):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            
        self.device = torch.device(f"cuda:{self.rank}")
        
        # 2. Load the full configuration to understand model depth
        self.config = HGRNBitConfig.from_pretrained(model_id)
        self.num_layers = self.config.num_hidden_layers

        layers_per_gpu = self.num_layers // self.world_size
        remainder = self.num_layers % self.world_size
        
        # Assign blocks of layers to specific ranks
        self.layer_start = self.rank * layers_per_gpu + min(self.rank, remainder)
        self.layer_end = self.layer_start + layers_per_gpu + (1 if self.rank < remainder else 0)
        
        # print(f"Rank {self.rank}: Managing layers {self.layer_start} to {self.layer_end - 1}")
        
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
            
        
        self.local_layers = nn.ModuleList([
            full_model.model.layers[i].to(self.device) 
            for i in range(self.layer_start, self.layer_end)
        ])
        self.past_key_values = None
        
        # Clean up the rest of the model structure from memory
        del full_model
        torch.cuda.empty_cache()

    def clear_cache(self):
        self.past_key_values = None

    @torch.inference_mode()
    def forward_step(self, input_ids=None, hidden_states=None, attention_mask=None, is_prefill=True):
        # Rank 0 handles the embedding layer
        if self.rank == 0:
            assert input_ids is not None, "Rank 0 requires input_ids"
            hidden_states = self.embed_tokens(input_ids)

        if self.rank > 0:
            shape_tensor = torch.zeros(3, dtype=torch.int64, device=self.device)
            dist.recv(shape_tensor, src=self.rank - 1)
            
            hidden_states = torch.zeros(tuple(shape_tensor.tolist()), dtype=torch.float16, device=self.device)
            dist.recv(hidden_states, src=self.rank - 1)
            mask_seq_len = shape_tensor[1].item()
            batch_size = hidden_states.shape[0]

            attention_mask = torch.zeros((batch_size, mask_seq_len), dtype=torch.long, device=self.device)
            dist.recv(attention_mask, src=self.rank - 1)

        if not is_prefill and attention_mask is not None:
            attention_mask = torch.ones((hidden_states.shape[0], 1), dtype=torch.long, device=self.device)
        new_past_key_values = []
        for idx,layer in enumerate(self.local_layers):
            layer_past = self.past_key_values[idx] if self.past_key_values is not None else None            
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=layer_past,
                use_cache=True,
                output_attentions=True,
                lower_bound=True,
                )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            new_past_key_values.append(outputs[1])
        self.past_key_values = new_past_key_values

        if self.rank < self.world_size - 1:
            shape_tensor = torch.tensor(list(hidden_states.shape), dtype=torch.int64, device=self.device)
            dist.send(shape_tensor, dst=self.rank + 1)
            dist.send(hidden_states, dst=self.rank + 1)
            mask_to_send = attention_mask if attention_mask is not None else torch.zeros((1, 0), dtype=torch.long, device=self.device)
            dist.send(mask_to_send, dst=self.rank + 1)
            return None
        else:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=20):
        self.clear_cache()

        current_input_ids = input_ids.to(self.device) if self.rank == 0 else None
        attention_mask = attention_mask.to(self.device) if self.rank == 0 else None
        generated_tokens = []
        batch_size, seq_len = input_ids.shape
        def generate_token(current_input_ids, is_prefill=False, temperature=0.75):
                    logits = self.forward_step(
                        input_ids=current_input_ids, 
                        attention_mask=attention_mask, 
                        is_prefill=is_prefill
                    )
                    next_token = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
                    if self.rank == self.world_size - 1:
                        next_token_logits = logits[:, -1, :]
                        if temperature > 0.0:
                            probs = torch.softmax(next_token_logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    dist.broadcast(next_token, src=self.world_size - 1)
                    current_input_ids = next_token if self.rank == 0 else None
                    generated_tokens.append(next_token.cpu())
                    return current_input_ids

        with nvtx.annotate("prefill", color="red"):
            current_input_ids = generate_token(current_input_ids, is_prefill=True)
        with nvtx.annotate("decode", color="red"):
            for _ in range(max_new_tokens-1):
                current_input_ids = generate_token(current_input_ids, is_prefill=False)
            
        return torch.cat(generated_tokens, dim=1)

def main():
    MODEL_ID = "ridger/MMfreeLM-2.7B"
    
    pipeline_model = PipelineParallelMatMulFreeInference(MODEL_ID)   
    batch_size = 5
    sequence_length = 20
    max_new_tokens = 10
    # Input example on Rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        # Simple dummy input token representation [Batch=1, Seq=4]
        inputs = generate_dataset_input_ids(MODEL_ID, batch_size, sequence_length)
        input_tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    else:
        input_tokens = torch.zeros((batch_size, sequence_length), dtype=torch.long) # placeholders for other ranks
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        
    # Synchronize all processes before starting inference loop
    dist.barrier()
    
    output = pipeline_model.generate(input_tokens, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    if int(os.environ.get("RANK", 0)) == 0:
        for i in range(batch_size):
            # 1. Decode the input tokens
            input_text = create_string_from_tokens(MODEL_ID, input_tokens[i])
            
            # 2. Extract and decode the specific output slice for this batch element
            # output shape is [Batch, Max_New_Tokens]
            generated_slice = output[i] 
            output_text = create_string_from_tokens(MODEL_ID, generated_slice)
            
            print(f"\n--- Batch Item {i+1} ---")
            print(f"Input Tokens:  {input_tokens[i].tolist()}")
            print(f"Input Text:    {input_text}")
            print(f"Output Tokens: {generated_slice.tolist()}")
            print(f"Output Text:   {output_text}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()