"""Memory-bounded packed checkpoint loader for HGRN Bit models."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from mmfreelm.models import HGRNBitConfig, HGRNBitForCausalLM
from mmfreelm.ops.fusedbitnet import CompressedType, FusedBitLinear, pack_weights


def _parent(module, key):
    parts = key.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


@torch.inference_mode()
def load_packed_hgrn(model_id: str, device: str = "cuda", packed: bool = True):
    """Load checkpoint tensors one at a time; BitLinear weights stay 2-bit packed."""
    snapshot = Path(snapshot_download(model_id))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    config = HGRNBitConfig.from_pretrained(snapshot)
    with init_empty_weights():
        model = HGRNBitForCausalLM(config)
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(snapshot / filename, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                module, name = _parent(model, key)
                tensor = shard.get_tensor(key).to(torch.float16)
                if isinstance(module, FusedBitLinear) and name == "weight":
                    scale = 1.0 / tensor.abs().mean().clamp_(min=1e-5)
                    ternary = (tensor * scale).round().clamp_(-1, 1)
                    module.cached_scale = scale.to(device)
                    if packed:
                        module.compressed_type = CompressedType.NAIVE
                        module.compressed_weights = pack_weights(ternary).to(device)
                        module._packed_orig_shape = tuple(ternary.shape)
                        module._packed_orig_numel = ternary.numel()
                    else:
                        module.cached_weights = tensor.to(device)
                        module.compressed_type = CompressedType.FLOAT16
                        module.use_compressed_weights = False
                    del module.weight
                else:
                    setattr(module, name, torch.nn.Parameter(tensor.to(device), requires_grad=False))
                del tensor
    return model.eval()
