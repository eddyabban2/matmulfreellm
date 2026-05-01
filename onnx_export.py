#!/usr/bin/env python3
"""
onnx_export_patched.py
Replaces Triton-backed ops with pure PyTorch equivalents, then exports to ONNX.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────
# 1. Pure-PyTorch RMSNorm to replace Triton one
# ─────────────────────────────────────────────
class PureTorchRMSNorm(nn.Module):
    def __init__(self, original):
        """
        Copies weight (and optional bias) from the original Triton-backed RMSNorm.
        Works for both RMSNorm and LayerNorm variants in mmfreelm.
        """
        super().__init__()
        self.eps = getattr(original, "eps", 1e-6) or 1e-6
        self.weight = nn.Parameter(original.weight.data.clone().float())
        if hasattr(original, "bias") and original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone().float())
        else:
            self.bias = None

    def forward(self, x, residual=None, *args, **kwargs):
        # Handle optional residual (some mmfreelm norms accept it)
        if residual is not None:
            x = x + residual

        orig_dtype = x.dtype
        x = x.float()

        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        weight = self.weight
        out = x_normed * weight
        if self.bias is not None:
            out = out + self.bias

        return out.to(orig_dtype)


def replace_triton_norms(model: nn.Module) -> nn.Module:
    """
    Walk every module; replace anything that uses Triton LayerNorm/RMSNorm
    with PureTorchRMSNorm.
    """
    # Class names used in mmfreelm
    triton_norm_names = {
        "RMSNorm", "LayerNorm", "RmsNorm",
        "FusedRMSNorm", "FusedLayerNorm",
    }

    for name, module in list(model.named_modules()):
        type_name = type(module).__name__
        if type_name not in triton_norm_names:
            continue

        # Check it actually has a weight (skip stubs)
        if not hasattr(module, "weight"):
            continue

        replacement = PureTorchRMSNorm(module)

        # Navigate to the parent and swap the attribute
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
        print(f"  Replaced {name} ({type_name}) → PureTorchRMSNorm")

    return model


# ─────────────────────────────────────────────
# 2. Patch the Triton LayerNormFn at import time
#    so any call that slips through still works
# ─────────────────────────────────────────────
def patch_layernorm_module():
    try:
        import mmfreelm.modules.layernorm as ln_mod

        def safe_rms_norm_fn(x, weight, bias=None, residual=None,
                             eps=1e-6, prenorm=False,
                             residual_in_fp32=False, is_rms_norm=True):
            if residual is not None:
                x = x + residual.to(x.dtype)
            orig = x.dtype
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + eps) * weight.float()
            if bias is not None:
                x = x + bias.float()
            x = x.to(orig)
            if prenorm:
                return x, residual
            return x

        ln_mod.rms_norm_fn = safe_rms_norm_fn
        ln_mod.layer_norm_fn = safe_rms_norm_fn
        print("  Patched mmfreelm layernorm module functions")
    except Exception as e:
        print(f"  Warning: could not patch layernorm module: {e}")


# ─────────────────────────────────────────────
# 3. Export
# ─────────────────────────────────────────────
def export(model_name: str = "ridger/MMfreeLM-370M", out_path: str = "mmfreelm.onnx"):
    print("Patching Triton layernorm functions...")
    patch_layernorm_module()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda().eval()

    print("Replacing Triton norm modules...")
    model = replace_triton_norms(model)

    # Cast everything to fp16 after replacement
    model = model.half()

    dummy = torch.randint(0, tokenizer.vocab_size, (1, 16), dtype=torch.long).cuda()

    # Verify forward pass works before attempting export
    print("Testing patched forward pass...")
    with torch.no_grad():
        out = model(dummy)
    print(f"  Forward OK — logits shape: {out.logits.shape}")

    print(f"Exporting to {out_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            # Pass logits only (unwrap CausalLMOutput)
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
    print(f"Done → {out_path}")


if __name__ == "__main__":
    export()