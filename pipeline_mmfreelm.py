# Example Usage:
#   torchrun --nproc_per_node=2 pipeline_mmfreelm.py 

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import nvtx
import copy 
import sys
from datetime import timedelta
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
import random
import gc 
from utils import generate_dataset_input_ids, create_string_from_tokens
from mmfreelm.ops.fusedbitnet import CompressedType, FusedBitLinear
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitModel, HGRNBitPreTrainedModel, HGRNBitBlock
from mmfreelm.modules import RMSNorm
from scaled_mmfree import print_system_ram
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["NCCL_IB_DISABLE"] = "1"
OMP_NUM_THREADS=1

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')

print_deadlocking_checks = True
class PipelineParallelMatMulFreeLM(HGRNBitModel):
    def __init__(self, layers_multiplier=1, weight_multiplier=1, vocab_size_multiplier=1, weight_compression=False, print_model_config=False):
        torch.set_default_dtype(torch.float16)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.model_device = torch.device(f"cuda:{self.rank}")
        compression_type = CompressedType.NAIVE if weight_compression else CompressedType.FLOAT16
        if not dist.is_initialized():
            timeout = timedelta(seconds=30)
            torch.cuda.set_device(self.local_rank)
            self.model_device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, device_id=self.model_device, timeout=timeout)
        config = HGRNBitConfig(
            vocab_size = int(32000*vocab_size_multiplier),
            hidden_size = int(2560*weight_multiplier),
            num_hidden_layers = int(32*layers_multiplier),
            attn_mode = "fused_recurrent",
            num_heads = 1,
            expand_ratio = 1,
            use_short_conv = False,
            conv_size = 4,
            share_conv_kernel = True,
            use_lower_bound = True,
            hidden_ratio = 1,
            intermediate_size = int(6912*weight_multiplier),
            hidden_act = "swish",
            max_position_embeddings = 2048,
            rms_norm_eps = 1e-6,
            use_cache = True,
            pad_token_id = None,
            bos_token_id = 1,
            eos_token_id = 2,
            tie_word_embeddings = False,
            initializer_range = 0.02,
            fuse_cross_entropy = True, 
            model_type = "hgrn_bit", 
            compressed_type=compression_type, 
            device=self.model_device, 
            print_model_config=print_model_config
        )
        super(HGRNBitModel, self).__init__(config)
        self.embeddings = None
        self.norm = None
        self.lm_head = None
        if self.rank == 0:
            self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size).to(self.model_device)
            nn.init.uniform_(self.embeddings.weight, a=-1, b=1)
        if self.rank == self.world_size - 1:
            self.lm_head = FusedBitLinear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.compressed_type = config.compressed_type
            self.lm_head.convert_weights(device=config.device)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer_count = config.num_hidden_layers // self.world_size
        if self.rank == self.world_size-1:
            remainder = config.num_hidden_layers % self.world_size
            if remainder > 0:
                print(f"Warning layer count: {config.num_hidden_layers} does not evenly go into world size: {self.world_size}. Last Layer will have {remainder} more rows than others")
            layer_count += remainder
        layers = []
        for layer_idx in range(layer_count):
            curr_layer = HGRNBitBlock(config, layer_idx)
            self.init_weights(curr_layer.attn.i_proj)
            self.init_weights(curr_layer.attn.f_proj)
            self.init_weights(curr_layer.attn.g_proj)
            self.init_weights(curr_layer.attn.o_proj)

            self.init_weights(curr_layer.mlp.gate_proj)
            self.init_weights(curr_layer.mlp.down_proj)
            layers.append(curr_layer)
            layers[-1].set_compression(config.compressed_type, config.device, convert_weights=True)
            if(print_model_config):
                print_system_ram(f"Allocating layer {layer_idx} from scratch")
        self.local_layers = nn.ModuleList(layers)
        self.local_layers.to(self.model_device)
        self.to(self.model_device)

    def init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, FusedBitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

        
    def old_init(self, layers_multiplier=1, weight_multiplier=1, vocab_size_multiplier=1, weight_compression=False, model_id="ridger/MMfreeLM-2.7B", print_model_config=False):
        if vocab_size_multiplier < 1:
            sys.exit("Vocab multiplier's smaller than 1 are unsupported")
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            timeout = timedelta(seconds=180)
            torch.cuda.set_device(self.local_rank)
            self.model_device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, device_id=self.model_device, timeout=timeout) 
        dist.barrier()
        self.model_device = torch.device(f"cuda:{self.rank}")

        self.config = HGRNBitConfig.from_pretrained(model_id)
        self.num_layers = self.config.num_hidden_layers

        layers_per_gpu = self.num_layers // self.world_size
        remainder = self.num_layers % self.world_size

        self.layer_start = self.rank * layers_per_gpu + min(self.rank, remainder)
        self.layer_end = self.layer_start + layers_per_gpu + (1 if self.rank < remainder else 0)
        
        full_model = HGRNBitForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.embeddings = None
        self.norm = None
        self.lm_head = None

        weight_compression = CompressedType.NAIVE if weight_compression else CompressedType.FLOAT16
        
        if self.rank == 0:
            if weight_multiplier == 1 and vocab_size_multiplier == 1: 
                self.embeddings = full_model.model.embeddings.to(self.model_device)
            else: 
                hidden_size = int(2560*weight_multiplier)
                vocab_size = int(full_model.vocab_size*vocab_size_multiplier)
                self.embeddings = nn.Embedding(
                    num_embeddings=vocab_size, 
                    embedding_dim=hidden_size, 
                    padding_idx=full_model.model.padding_idx, 
                    device=self.model_device)
                nn.init.uniform_(self.embeddings.weight, a=-1, b=1)
                self.embeddings.to(torch.float16)
        if self.rank == self.world_size - 1:
            self.norm = full_model.model.norm.to(self.model_device)
            self.lm_head = full_model.lm_head.to(self.model_device)
            if weight_multiplier != 1 or vocab_size_multiplier != 1:
                self.norm.increase_size(weight_multiplier)
                self.lm_head.increase_size(weight_multiplier, vocab_size_multiplier, compressed_type=weight_compression)
        model_layers = []
        if layers_multiplier == 1:
            model_layers = [copy.deepcopy(full_model.model.layers[i])
                for i in range(self.layer_start, self.layer_end)]
        else:
            layer_count = int(layers_multiplier * (self.layer_end - self.layer_start))
            for _ in range(layer_count):
                random_layer_index = random.randint(self.layer_start, self.layer_end - 1)
                model_layers.append(copy.deepcopy(full_model.model.layers[random_layer_index]))

        self.local_layers = nn.ModuleList(model_layers)

        if weight_multiplier != 1:
            for layer in self.local_layers:
                layer.attn.i_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.attn.f_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.attn.g_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.attn.o_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.mlp.gate_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.mlp.down_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
                layer.attn_norm.increase_size(weight_multiplier)
                layer.mlp_norm.increase_size(weight_multiplier)
                layer.attn.g_norm.increase_size(weight_multiplier)

        self.local_layers.to(self.model_device)
        self.past_key_values_dict = {}
        del full_model
        torch.cuda.empty_cache()

        if print_model_config: 
            # embedding 
            if self.embeddings != None:
                print(f"[rank{self.rank}] Embedding Layer: {self.embeddings}")
            # layers 
            print(f"[rank{self.rank}] Local Layers: {self.local_layers}")
            # norm
            if self.norm != None: 
                print(f"[rank{self.rank}] norm: {self.norm}")
            # lm head 
            if self.lm_head != None: 
                print(f"[rank{self.rank}] lm_head: {self.lm_head}")

    def clear_cache(self):
        self.past_key_values_dict = {}

    @torch.inference_mode()
    def pipelined_forward_step(self, mb_id, input_ids=None, hidden_states=None, attention_mask=None, is_prefill=True, step=None, num_mbs=None, current_mb_inputs=None, all_generated_tokens=None, batch_sizes=None):
        with nvtx.annotate(f"micro batch: {mb_id}", color="orange"):
            mb_past_kvs = self.past_key_values_dict.get(mb_id, None)
            new_past_key_values = []
            def broadcast_next_token():
                broadcasting_mb_id = (step - self.world_size + 1) % num_mbs
                bc_active = step >= self.world_size - 1 
                
                if print_deadlocking_checks:
                    print(f"[{self.rank}] on step {step} bc_active: {bc_active}")
                if bc_active:
                    mb_bs = batch_sizes[broadcasting_mb_id]
                    next_token = torch.zeros((mb_bs, 1), dtype=torch.int64, device=self.model_device)
                    if self.rank == self.world_size - 1:
                        assert logits is not None, (
                            f"Rank {self.rank}: expected logits for mb {broadcasting_mb_id} at step {step}"
                        )
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    next_token = next_token.contiguous()
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] reached broadcast for micro batch {broadcasting_mb_id}")
                        print(f"[{self.rank}] Next token tensor before broadcast: {next_token}")
                        print(f"[{self.rank}] Next token tensor shape before broadcast: {next_token.shape}")
                        print(f"[{self.rank}] Next token tensor dtype before broadcast: {next_token.dtype}")
                    dist.broadcast(next_token, src=self.world_size-1)
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] Next token tensor after broadcast: {next_token}")
                        print(f"[{self.rank}] Next token tensor shape after broadcast: {next_token.shape}")
                        print(f"[{self.rank}] Next token tensor dtype after broadcast: {next_token.dtype}")
                        print(f"[{self.rank}] finished broadcast for micro batch {broadcasting_mb_id}")
                    if self.rank == 0:
                        current_mb_inputs[broadcasting_mb_id] = next_token
                    all_generated_tokens[broadcasting_mb_id].append(next_token.cpu())
            if self.rank == 0:
                assert input_ids is not None, "Rank 0 requires input_ids"
                next_rank = (self.rank + 1) % self.world_size
                if next_rank == self.world_size-1:
                    print(f"[{self.rank}] Next rank is last broadcasting now")
                    broadcast_next_token()
                hidden_states = self.embeddings(input_ids)
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.long, device=self.model_device
                    )
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
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] iterating on layer: {idx}")
                    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                    new_past_key_values.append(outputs[1])
                if next_rank != self.world_size-1:
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] Next rank is not last broadcasting now")
                    broadcast_next_token()
                shape_tensor = torch.tensor(list(hidden_states.shape), dtype=torch.int64, device=self.model_device)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sending hidden states shape: {shape_tensor}")
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sending hidden states and information to [{next_rank}] for micro batch: {mb_id}")
                dist.send(shape_tensor, dst=next_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sent hidden states and information to [{next_rank}] for micro batch: {mb_id}")
                dist.send(hidden_states, dst=next_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sent hidden states and information to [{next_rank}] for micro batch: {mb_id}")
            elif 0 < self.rank < self.world_size - 1:
                prev_rank = (self.rank - 1) % self.world_size
                next_rank = (self.rank + 1) % self.world_size
                shape_tensor = torch.empty(3, dtype=torch.int64, device=self.model_device)

                if print_deadlocking_checks:
                    print(f"[{self.rank}] Recieving hidden states and information to [{self.rank-1}] for micro batch: {mb_id}")
                    print(f"[{self.rank}] waiting for hidden state shape tensor from {self.rank-1}")
                    print(f"[{self.rank}] Expecting shape tensor to fit in {shape_tensor}")
                dist.recv(shape_tensor, src=prev_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Recieving hidden states tensor shape {shape_tensor}")
                hidden_states = torch.empty(tuple(shape_tensor.tolist()), dtype=torch.float16, device=self.model_device)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Shape tensor is {shape_tensor}")
                    print(f"[{self.rank}] waiting for hidden states tensor from {self.rank-1}")
                dist.recv(hidden_states, src=prev_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] recieved hidden states tensor from {self.rank-1}")
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.long, device=self.model_device
                    )
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
                    if print_deadlocking_checks:
                        print(f"[{self.rank}]iterating on layer: {idx}")
                if next_rank != self.world_size-1:
                    broadcast_next_token()
                shape_tensor = torch.tensor(list(hidden_states.shape), dtype=torch.int64, device=self.model_device)
                assert torch.all(shape_tensor != 0), "Tensor contains a zero!"
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sending hidden states and information to [{self.rank+1}] for micro batch: {mb_id}")
                dist.send(shape_tensor, dst=next_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sending hidden states shape: {shape_tensor}")
                dist.send(hidden_states, dst=next_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Sent hidden states and information to [{self.rank+1}] for micro batch: {mb_id}")
                if next_rank == self.world_size-1:
                    broadcast_next_token()
            elif self.rank == self.world_size - 1:
                prev_rank = (self.rank - 1) % self.world_size
                next_rank = (self.rank + 1) % self.world_size
                shape_tensor = torch.zeros(3, dtype=torch.int64, device=self.model_device)

                if print_deadlocking_checks:
                    print(f"[{self.rank}] Recieving hidden states and information to [{self.rank-1}] for micro batch: {mb_id}")
                    print(f"[{self.rank}] waiting for hidden state shape tensor from {self.rank-1}")
                    print(f"[{self.rank}] Expecting shape tensor to fit in {shape_tensor}")
                dist.recv(shape_tensor, src=prev_rank)
                assert torch.all(shape_tensor != 0), "Tensor contains a zero!"
                hidden_states = torch.zeros(tuple(shape_tensor.tolist()), dtype=torch.float16, device=self.model_device)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] Shape tensor is {shape_tensor}")
                    print(f"[{self.rank}] waiting for hidden states tensor from {self.rank-1}")
                dist.recv(hidden_states, src=prev_rank)
                if print_deadlocking_checks:
                    print(f"[{self.rank}] recieved hidden states tensor from {self.rank-1}")
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.long, device=self.model_device
                    )
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
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] iterating on layer: {idx}")
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)  
                broadcast_next_token()
            else:
                assert False, f"Invalid rank number: {self.rank}"
                
    @torch.inference_mode()
    def generate_pipelined(self, micro_batches, max_new_tokens=20, temperature=0.75, single_call=True):
        self.clear_cache()
        if self.rank == 0 and len(micro_batches) < self.world_size:
            sys.exit(f"Number of micro batches is smaller than world size\n\tWorld size: {self.world_size} Micro Batches: {len(micro_batches)}")

        num_mbs_tensor = torch.tensor(
            [len(micro_batches) if self.rank == 0 else 0],
            dtype=torch.int64,
            device=self.model_device,
        )
        if print_deadlocking_checks:
            print(f"[{self.rank}] broadcasting number of microbatches {num_mbs_tensor}")
        dist.broadcast(num_mbs_tensor, src=0)
        if print_deadlocking_checks:
            print(f"[{self.rank}] finished broadcasting number of microbatches {num_mbs_tensor}")
        num_mbs = num_mbs_tensor.item()

        all_generated_tokens = {mb_id: [] for mb_id in range(num_mbs)}
        current_mb_inputs: dict = {}
        current_mb_masks: dict = {}

        batch_sizes: dict = {}
        for mb_id in range(num_mbs):
            if self.rank == 0:
                mb = micro_batches[mb_id]
                current_mb_inputs[mb_id] = mb["input_ids"].to(self.model_device)
                current_mb_masks[mb_id] = mb["attention_mask"].to(self.model_device)
                bs = mb["input_ids"].shape[0]
            else:
                bs = 0

            bs_tensor = torch.tensor([bs], dtype=torch.int64, device=self.model_device)
            if print_deadlocking_checks:
                print(f"[{self.rank}] broadcasting batch size tensor {bs_tensor}")
            dist.broadcast(bs_tensor, src=0)
            if print_deadlocking_checks:
                print(f"[{self.rank}] finished broadcasting batch size tensor {bs_tensor}")
            batch_sizes[mb_id] = bs_tensor.item()
        if self.rank == 0:
            print(f"[{self.rank}] Attention Masks:")
            for key, value in current_mb_masks.items():
                if print_deadlocking_checks:
                    print(f"[{self.rank}] \tmb_id: {key} mask: {value}")

        def generate_token_loop(is_prefill, num_tokens) -> None:
            total_steps = num_mbs * num_tokens + self.world_size - 1
            if print_deadlocking_checks:
                print(f"[{self.rank}] total steps = {total_steps}")
            for step in range(total_steps):
                with nvtx.annotate(f"step: {step}", color="violet"):
                    mb_id = (step - self.rank) % num_mbs
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] on step {step} working on mb: {mb_id}")
                    active = self.rank <=  step and step < (num_mbs*num_tokens + self.rank)
                    if print_deadlocking_checks:
                        print(f"[{self.rank}] on step {step} active: {active}")
                    if active:
                        if self.rank == 0:
                            print(f"[{self.rank}] Attention Masks in loop:")
                            for key, value in current_mb_masks.items():
                                print(f"[{self.rank}] \tmb_id: {key} mask: {value}")
                        inp = current_mb_inputs[mb_id] if self.rank == 0 else None
                        mask = current_mb_masks[mb_id] if (self.rank == 0 and step == mb_id) else None
                        
                        self.pipelined_forward_step(
                            mb_id=mb_id,
                            input_ids=inp,
                            attention_mask=mask,
                            is_prefill=is_prefill,
                            step=step, 
                            num_mbs=num_mbs, 
                            current_mb_inputs=current_mb_inputs, 
                            all_generated_tokens=all_generated_tokens,
                            batch_sizes=batch_sizes
                        )

        if single_call: 
            generate_token_loop(False, max_new_tokens)
        else:
            with nvtx.annotate("pipelined_prefill", color="blue"):
                generate_token_loop(True, 1)
            with nvtx.annotate("pipelined_decode", color="green"):
                generate_token_loop(False, max_new_tokens-1)

        final_outputs = {}
        for mb_id in range(num_mbs):
            if all_generated_tokens[mb_id]:
                final_outputs[mb_id] = torch.cat(all_generated_tokens[mb_id], dim=1)
        return final_outputs


def main():
    MODEL_ID = "ridger/MMfreeLM-2.7B"
    layers_multiplier = 0.125
    weight_multiplier = 3.9375
    vocab_size_multiplier = 1
    print_model_config = True
    use_weight_compression = False
    pipeline_model = PipelineParallelMatMulFreeLM(layers_multiplier=layers_multiplier, weight_multiplier=weight_multiplier, vocab_size_multiplier=vocab_size_multiplier, print_model_config=print_model_config, weight_compression=use_weight_compression)
    memory_bytes = torch.cuda.memory_allocated()
    memory_gb = memory_bytes / (1024 ** 3)
    print(f"GPU memory usage: {memory_gb:.2f} GB")
    num_micro_batches = 3
    batch_size_per_mb = 5
    sequence_length = 5000
    max_new_tokens = 5

    micro_batches = []
    if int(os.environ.get("RANK", 0)) == 0:
        for _ in range(num_micro_batches):
            inputs = generate_dataset_input_ids(MODEL_ID, batch_size_per_mb, sequence_length)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
    print("generated input tokens")
    dist.barrier()
    print("running warmup")
    for _ in range(100):
        outputs = pipeline_model.generate_pipelined(micro_batches, max_new_tokens=3)
    gc.collect()
    torch.cuda.empty_cache()
    print("finished running warmup runnning workload")
    print("attempting to run pipeline")
    dist.barrier()

    outputs = pipeline_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
    print(f"Outputs of Model: {outputs}")
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
                print(f"Output Tensor: {generated_tensor[i]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()