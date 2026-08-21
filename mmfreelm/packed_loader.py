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
from mmfreelm.ops.ternary_packing import pack_for_type


def _parent(module, key):
    parts = key.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


@torch.inference_mode()
def load_packed_hgrn(
    model_id: str,
    device: str = "cuda",
    packed: bool = True,
    compressed_type: CompressedType = CompressedType.NAIVE,
):
    """Load checkpoint tensors one at a time; BitLinear weights stay packed.

    compressed_type selects NAIVE/PACKED_2BIT, TQ1_0, or TQ2_0 when packed=True.
    packed=False stores FLOAT16 weights (higher memory).
    """
    if not packed:
        compressed_type = CompressedType.FLOAT16
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
                    module.compressed_type = compressed_type
                    if compressed_type.is_packed:
                        if compressed_type in (CompressedType.TQ1_0, CompressedType.TQ2_0):
                            packed_w = pack_for_type(ternary, compressed_type)
                        else:
                            packed_w = pack_weights(ternary.clone())
                        module.compressed_weights = packed_w.to(device)
                        module._packed_orig_shape = tuple(ternary.shape)
                        module._packed_orig_numel = ternary.numel()
                        module._packed_unit_scale = True
                    else:
                        # Unpacked fp16 checkpoint weights (not ternary).
                        module.cached_weights = tensor.to(device)
                        module.use_compressed_weights = False
                    del module.weight
                else:
                    setattr(module, name, torch.nn.Parameter(tensor.to(device), requires_grad=False))
                del tensor
    return model.eval()
