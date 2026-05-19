# Example Usage:
#   torchrun --nproc_per_node=2 pipeline_mmfreelm.py 

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import nvtx
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from utils import generate_dataset_input_ids, create_string_from_tokens


class PipelineParallelMatMulFreeLM:
    def __init__(self, model_id="ridger/MMfreeLM-2.7B"):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            
        self.device = torch.device(f"cuda:{self.rank}")

        self.config = HGRNBitConfig.from_pretrained(model_id)
        self.num_layers = self.config.num_hidden_layers

        layers_per_gpu = self.num_layers // self.world_size
        remainder = self.num_layers % self.world_size

        self.layer_start = self.rank * layers_per_gpu + min(self.rank, remainder)
        self.layer_end = self.layer_start + layers_per_gpu + (1 if self.rank < remainder else 0)
        
        # print(f"Rank {self.rank}: Managing layers {self.layer_start} to {self.layer_end - 1}")
        
        full_model = HGRNBitForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        
        if self.rank == 0:
            self.embed_tokens = full_model.model.embeddings.to(self.device)
            
        if self.rank == self.world_size - 1:
            self.norm = full_model.model.norm.to(self.device)
            self.lm_head = full_model.lm_head.to(self.device)

        self.local_layers = nn.ModuleList(
            [full_model.model.layers[i].to(self.device) for i in range(self.layer_start, self.layer_end)]
        )

        self.past_key_values_dict = {}
        del full_model
        torch.cuda.empty_cache()

    def clear_cache(self):
        self.past_key_values_dict = {}

    @torch.inference_mode()
    def pipelined_forward_step(self, mb_id, input_ids=None, hidden_states=None, attention_mask=None, is_prefill=True):
        with nvtx.annotate(f"micro batch: {mb_id}", color="orange"):
            if self.rank == 0:
                assert input_ids is not None, "Rank 0 requires input_ids"
                hidden_states = self.embed_tokens(input_ids)
            else:
                shape_tensor = torch.zeros(3, dtype=torch.int64, device=self.device)
                dist.recv(shape_tensor, src=self.rank - 1)

                hidden_states = torch.zeros(
                    tuple(shape_tensor.tolist()), dtype=torch.float16, device=self.device
                )
                dist.recv(hidden_states, src=self.rank - 1)

                mask_shape = torch.zeros(2, dtype=torch.int64, device=self.device)
                dist.recv(mask_shape, src=self.rank - 1)

                attention_mask = torch.zeros(
                    tuple(mask_shape.tolist()), dtype=torch.long, device=self.device
                )
                dist.recv(attention_mask, src=self.rank - 1)

            if not is_prefill and attention_mask is not None:
                attention_mask = torch.ones(
                    (hidden_states.shape[0], 1), dtype=torch.long, device=self.device
                )
            mb_past_kvs = self.past_key_values_dict.get(mb_id, None)
            new_past_key_values = []

            for idx, layer in enumerate(self.local_layers):
                layer_past = mb_past_kvs[idx] if mb_past_kvs is not None else None
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

            self.past_key_values_dict[mb_id] = new_past_key_values

            if self.rank < self.world_size - 1:
                shape_tensor = torch.tensor(list(hidden_states.shape), dtype=torch.int64, device=self.device)
                dist.send(shape_tensor, dst=self.rank + 1)
                dist.send(hidden_states, dst=self.rank + 1)

                if attention_mask is None:
                    attention_mask = torch.ones(
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        dtype=torch.long,
                        device=self.device,
                    )
                mask_shape = torch.tensor(list(attention_mask.shape), dtype=torch.int64, device=self.device)
                dist.send(mask_shape, dst=self.rank + 1)
                dist.send(attention_mask, dst=self.rank + 1)
                return None

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    @torch.inference_mode()
    def generate_pipelined(self, micro_batches, max_new_tokens=20, temperature=0.75):
        self.clear_cache()

        num_mbs_tensor = torch.tensor(
            [len(micro_batches) if self.rank == 0 else 0],
            dtype=torch.int64,
            device=self.device,
        )
        dist.broadcast(num_mbs_tensor, src=0)
        num_mbs = num_mbs_tensor.item() 

        all_generated_tokens = {mb_id: [] for mb_id in range(num_mbs)}
        current_mb_inputs: dict = {}
        current_mb_masks: dict = {}

        batch_sizes: dict = {}
        for mb_id in range(num_mbs):
            if self.rank == 0:
                mb = micro_batches[mb_id]
                current_mb_inputs[mb_id] = mb["input_ids"].to(self.device)
                current_mb_masks[mb_id] = mb["attention_mask"].to(self.device)
                bs = mb["input_ids"].shape[0]
            else:
                bs = 0

            bs_tensor = torch.tensor([bs], dtype=torch.int64, device=self.device)
            dist.broadcast(bs_tensor, src=0)          
            batch_sizes[mb_id] = bs_tensor.item()

        def generate_token_loop(is_prefill: bool) -> None:

            total_steps = num_mbs + self.world_size - 1

            for step in range(total_steps):
                with nvtx.annotate(f"step: {step}", color="violet"):
                    mb_id = step - self.rank
                    active = 0 <= mb_id < num_mbs

                    logits = None
                    if active:
                        inp = current_mb_inputs.get(mb_id) if self.rank == 0 else None
                        mask = current_mb_masks.get(mb_id) if (self.rank == 0 and is_prefill) else None
                        logits = self.pipelined_forward_step(
                            mb_id=mb_id,
                            input_ids=inp,
                            attention_mask=mask,
                            is_prefill=is_prefill,
                        )

                    broadcasting_mb_id = step - (self.world_size - 1)
                    bc_active = 0 <= broadcasting_mb_id < num_mbs

                    if bc_active:
                        mb_bs = batch_sizes[broadcasting_mb_id]
                        next_token = torch.zeros((mb_bs, 1), dtype=torch.int64, device=self.device)

                        if self.rank == self.world_size - 1:
                            assert logits is not None, (
                                f"Rank {self.rank}: expected logits for mb {broadcasting_mb_id} at step {step}"
                            )
                            next_token_logits = logits[:, -1, :]
                            if temperature > 0.0:
                                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                        dist.broadcast(next_token, src=self.world_size - 1)

                        if self.rank == 0:
                            current_mb_inputs[broadcasting_mb_id] = next_token
                        all_generated_tokens[broadcasting_mb_id].append(next_token.cpu())

        with nvtx.annotate("pipelined_prefill", color="blue"):
            generate_token_loop(is_prefill=True)

        with nvtx.annotate("pipelined_decode", color="green"):
            for _ in range(max_new_tokens - 1):
                generate_token_loop(is_prefill=False)

        final_outputs = {}
        for mb_id in range(num_mbs):
            if all_generated_tokens[mb_id]:
                final_outputs[mb_id] = torch.cat(all_generated_tokens[mb_id], dim=1)
        return final_outputs


def main():
    MODEL_ID = "ridger/MMfreeLM-2.7B"
    pipeline_model = PipelineParallelMatMulFreeLM(MODEL_ID)

    num_micro_batches = 3
    batch_size_per_mb = 5
    sequence_length = 20
    max_new_tokens = 10

    micro_batches = []
    if int(os.environ.get("RANK", 0)) == 0:
        for _ in range(num_micro_batches):
            inputs = generate_dataset_input_ids(MODEL_ID, batch_size_per_mb, sequence_length, max_new_tokens=max_new_tokens)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
    print("generated input tokens")
    dist.barrier()

    outputs = pipeline_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
    print("tokens generated")

    if int(os.environ.get("RANK", 0)) == 0:
        for mb_id, generated_tensor in outputs.items():
            print(f"\n================ MICRO-BATCH {mb_id} ================")
            original_input = micro_batches[mb_id]["input_ids"]
            for i in range(batch_size_per_mb):
                input_text = create_string_from_tokens(MODEL_ID, original_input[i])
                output_text = create_string_from_tokens(MODEL_ID, generated_tensor[i])
                print(f"\n--- Item {i+1} ---")
                print(f"Input Text:    {input_text}")
                print(f"Output Text:   {output_text}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()