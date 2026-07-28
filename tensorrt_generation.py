# Shared TensorRT / CUDA-graph path for MMfreeLM (ONNX export + JetPack TRT).
# Used by generate.py and benchmark_trt.py.

import gc
import hashlib
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.modules.layernorm import rms_norm_ref
from mmfreelm.ops.fusedbitnet import unpack_weights, weight_quant

try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import numpy as np
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False


def trt_dependencies_available() -> bool:
    return HAS_TRT and HAS_PYCUDA


def default_trt_cache_paths(model_name: str) -> Tuple[str, str]:
    key = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/mmfreelm_{key}.engine", f"/tmp/mmfreelm_{key}.onnx"


# ── Pure-PyTorch replacements for Triton-backed ops ──────────────────────────


def _activation_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-row int8 quant/dequant (no packing) matching ``_layer_norm_fwd_quant``."""
    xf = x.float()
    scale = 127.0 / xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q = torch.floor(xf * scale + 0.5).clamp(-128, 127).to(torch.int8)
    return (q.float() / scale).to(x.dtype)


class _DequantInt8Weight(torch.autograd.Function):
    """ONNX DequantizeLinear(scale=1, zp=0) for ternary INT8 weights; TRT-compatible."""

    @staticmethod
    def forward(ctx, weight_int8: torch.Tensor):
        return weight_int8.float()

    @staticmethod
    def symbolic(g, weight_int8):
        scale = g.op(
            "Constant",
            value_t=torch.tensor([1.0], dtype=torch.float32),
        )
        zero_point = g.op(
            "Constant",
            value_t=torch.tensor([0], dtype=torch.int8),
        )
        return g.op("DequantizeLinear", weight_int8, scale, zero_point)


def _dequant_int8_weight(weight_int8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    w = _DequantInt8Weight.apply(weight_int8)
    return w.to(dtype)


def _layer_norm_fwd_quant_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """RMSNorm + activation quant matching ``layer_norm_linear_quant_fn``."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return _activation_quant_int8(y)


class PureTorchRMSNorm(nn.Module):
    """Replaces flash-attn / mmfreelm Triton RMSNorm."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.weight = src.weight
        self.bias = getattr(src, "bias", None)
        self.eps = getattr(src, "eps", 1e-6)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None and residual_in_fp32:
            residual = residual.float()
        return rms_norm_ref(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            upcast=residual_in_fp32,
        )


class PureTorchFusedRMSNormSwishGate(nn.Module):
    """
    ONNX/TRT-friendly drop-in for FusedRMSNormSwishGate (Triton).
    y = RMSNorm(x) * weight * (o * sigmoid(o)) — matches fused_norm_gate kernel.
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        self.weight = src.weight
        self.bias = getattr(src, "bias", None)
        self.eps = getattr(src, "eps", 1e-5)

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual.to(x.dtype)
        x_merged = x
        orig = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        y = xf.to(orig) * self.weight
        if self.bias is not None:
            y = y + self.bias
        og = o.float()
        y = y * (og * torch.sigmoid(og)).to(y.dtype)
        return (y, x_merged) if prenorm else y


def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """Match FusedBitLinear.cached_weights / mmfreelm.ops.fusedbitnet.weight_quant."""
    return weight_quant(w)


def activation_quantize(x: torch.Tensor) -> torch.Tensor:
    """Match the original per-token int8 activation quantize/dequantize step."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


class PureTorchFusedBitLinear(nn.Module):
    """Inference-only RMSNorm + linear with INT8 ternary weights (no 2-bit packing).

    Weights are stored as ``int8`` in {-1, 0, 1} and cast to the activation dtype
    at runtime so ONNX/TRT materialize compact INT8 initializers instead of FP16.
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        self.bias = getattr(src, "bias", None)
        src_weight = src.weight
        norm = (
            getattr(src, "norm", None)
            or getattr(src, "layer_norm", None)
            or getattr(src, "in_norm", None)
        )
        if norm is not None:
            self.norm_weight = norm.weight
            self.norm_bias = getattr(norm, "bias", None)
            self.norm_eps = getattr(norm, "eps", 1e-6)
        else:
            self.norm_weight = getattr(
                src, "norm_weight", getattr(src, "weight_norm", None)
            )
            self.norm_bias = getattr(src, "norm_bias", None)
            self.norm_eps = getattr(src, "eps", 1e-6)
        self.has_norm = self.norm_weight is not None
        with torch.no_grad():
            cached_w = getattr(src, "cached_weights", None)
            compressed = getattr(src, "compressed_weights", None)
            cached_scale = getattr(src, "cached_scale", None)

            if compressed is not None:
                quant_w = unpack_weights(compressed, src_weight.dtype)
            elif cached_w is not None:
                quant_w = cached_w
            else:
                orig_w = src_weight.float()
                if cached_scale is None:
                    cached_scale = 1.0 / orig_w.abs().mean().clamp_(min=1e-5)
                quant_w = weight_quant(orig_w)

            if cached_scale is None:
                cached_scale = 1.0 / src_weight.float().abs().mean().clamp_(min=1e-5)

            weight_int8 = quant_w.round().clamp(-1, 1).to(torch.int8)
            device = src_weight.device
            self.register_buffer("weight_int8", weight_int8, persistent=True)
            # FusedBitLinear divides the matmul by cached_scale (= 1/mean(|W|)).
            self.register_buffer(
                "_output_scale",
                cached_scale.reciprocal().to(dtype=torch.float32, device=device),
                persistent=True,
            )

    def forward(self, x):
        if self.has_norm:
            x = _layer_norm_fwd_quant_ref(
                x, self.norm_weight, self.norm_bias, self.norm_eps
            )
        else:
            x = _activation_quant_int8(x)
        w = _dequant_int8_weight(self.weight_int8, x.dtype)
        return F.linear(x, w, self.bias) * self._output_scale.to(dtype=x.dtype)


def _module_source_file(module: nn.Module) -> str:
    mod = sys.modules.get(type(module).__module__)
    return getattr(mod, "__file__", "") or ""


def patch_all_triton_ops(model: nn.Module) -> nn.Module:
    norm_replaced = 0
    linear_replaced = 0
    swish_replaced = 0
    replacements = {}

    for name, module in model.named_modules():
        src = _module_source_file(module)
        if "mmfreelm" not in src:
            continue
        cls_name = type(module).__name__.lower()
        is_fused_rms_swish = "swishgate" in cls_name and "rms" in cls_name
        is_fused_linear = (
            hasattr(module, "weight")
            and module.weight.dim() == 2
            and (
                hasattr(module, "norm")
                or hasattr(module, "layer_norm")
                or hasattr(module, "in_norm")
                or "linear" in cls_name
                or "proj" in cls_name
            )
            and (
                "fusedbit" in cls_name
                or "bitlinear" in cls_name
                or "layernormlinear" in cls_name
                or "fused" in cls_name
            )
        )
        is_norm = (
            hasattr(module, "weight")
            and module.weight.dim() == 1
            and ("norm" in cls_name or "norm" in name.split(".")[-1].lower())
            and not is_fused_linear
            and not is_fused_rms_swish
        )
        if is_fused_rms_swish:
            replacements[name] = ("fused_rms_swish", module)
        elif is_fused_linear:
            replacements[name] = ("fused_linear", module)
        elif is_norm:
            replacements[name] = ("norm", module)

    for name, (kind, src_module) in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        if kind == "fused_linear":
            new_mod = PureTorchFusedBitLinear(src_module)
            new_mod = new_mod.to(
                next(src_module.parameters()).device,
                next(src_module.parameters()).dtype,
            )
            setattr(parent, child_name, new_mod)
            linear_replaced += 1
        elif kind == "fused_rms_swish":
            new_mod = PureTorchFusedRMSNormSwishGate(src_module).to(
                next(src_module.parameters()).device,
                next(src_module.parameters()).dtype,
            )
            setattr(parent, child_name, new_mod)
            swish_replaced += 1
        else:
            setattr(parent, child_name, PureTorchRMSNorm(src_module))
            norm_replaced += 1

    print(
        f"[PATCH] Replaced {norm_replaced} Triton norms, "
        f"{swish_replaced} RMSNorm+SwishGate, "
        f"{linear_replaced} fused BitLinear → PyTorch (INT8 ternary weights, no packing)."
    )
    return model


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    sp[(torch.cumsum(sp, dim=-1) - sp) > top_p] = 0.0
    sp /= sp.sum(dim=-1, keepdim=True)
    return si.gather(-1, torch.multinomial(sp, 1))


def decode_loop(
    step_fn,
    input_ids,
    max_length,
    do_sample=True,
    top_p=0.4,
    temperature=0.6,
):
    generated = input_ids.clone()
    for _ in range(max_length - input_ids.shape[1]):
        logits = step_fn(generated)
        tok = (
            top_p_sample(logits, top_p, temperature)
            if do_sample
            else logits.argmax(-1, keepdim=True)
        )
        generated = torch.cat([generated, tok], dim=1)
    return generated


class ModelForwardWrapper(nn.Module):
    """model(input_ids) → last-token logits (B, V). Safe to graph / trace."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits[:, -1, :]

    def parameters(self, **kw):
        return self.model.parameters(**kw)


class CUDAGraphAccelerator:
    def __init__(self, model: nn.Module):
        self.fwd = ModelForwardWrapper(model)
        self._graphs = {}
        print("[CUDAGRAPH] Ready.")

    def _capture(self, batch: int, seq: int):
        print(f"[CUDAGRAPH] Capturing forward graph batch={batch} seq={seq} …")
        static_in = torch.zeros((batch, seq), dtype=torch.long, device="cuda")
        static_out = torch.zeros(
            (batch, self.fwd.model.config.vocab_size),
            dtype=torch.float16,
            device="cuda",
        )
        with torch.no_grad():
            for _ in range(3):
                self.fwd(static_in)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = self.fwd(static_in)
            static_out.copy_(out)
        self._graphs[batch] = (g, static_in, static_out)
        print("[CUDAGRAPH] ✓ Captured.")

    def generate(
        self,
        input_ids,
        max_length=32,
        do_sample=True,
        top_p=0.4,
        temperature=0.6,
        **_,
    ):
        batch, prompt_len = input_ids.shape
        if batch not in self._graphs:
            self._capture(batch, prompt_len)
        g, static_in, static_out = self._graphs[batch]

        def step_fn(ids):
            if ids.shape == static_in.shape:
                static_in.copy_(ids)
                g.replay()
                return static_out.clone()
            with torch.no_grad():
                return self.fwd(ids)

        return decode_loop(step_fn, input_ids, max_length, do_sample, top_p, temperature)

    def parameters(self):
        return self.fwd.model.parameters()


class ONNXTRTAccelerator:
    def __init__(
        self,
        model: nn.Module,
        max_batch: int,
        max_seq: int,
        *,
        model_name: str = "",
        use_fp16: bool = True,
        rebuild: bool = False,
        engine_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
    ):
        if not trt_dependencies_available():
            raise RuntimeError("Needs JetPack tensorrt + pycuda.")
        if engine_path is None or onnx_path is None:
            ep, op = default_trt_cache_paths(model_name or "default")
            engine_path = engine_path or ep
            onnx_path = onnx_path or op
        self.engine_path = engine_path
        self.onnx_path = onnx_path

        model = patch_all_triton_ops(model)
        model.eval()
        self.fwd = ModelForwardWrapper(model)
        self.parameter_count = sum(p.numel() for p in self.fwd.model.parameters())
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        if rebuild:
            for p in [self.engine_path, self.onnx_path]:
                if os.path.exists(p):
                    os.remove(p)

        self.engine = self._load_or_build()
        self.context = self.engine.create_execution_context()
        self._alloc_buffers()
        print("[TRT] Engine ready ✓")

    def parameters(self):
        # Kept for the benchmark's model-like interface; the PyTorch model is
        # deliberately released before TensorRT engine construction.
        return iter(())

    def _release_export_model(self):
        if self.fwd is not None:
            del self.fwd
            self.fwd = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[TRT] Released PyTorch export model before engine build.")

    def _load_or_build(self):
        if os.path.exists(self.engine_path):
            print(f"[TRT] Loading cached engine: {self.engine_path}")
            with open(self.engine_path, "rb") as f:
                return trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        print("[TRT] Building engine …")
        if not os.path.exists(self.onnx_path):
            self._export_onnx()
        self._release_export_model()
        data = self._build_engine()
        with open(self.engine_path, "wb") as f:
            f.write(data)
        print(f"[TRT] Cached → {self.engine_path}")
        return trt.Runtime(self.logger).deserialize_cuda_engine(data)

    def _export_onnx(self):
        # hgrn_bit imports fused_recurrent_hgrn by name; ONNX trace with seq>1
        # otherwise hits Triton autotune (not traceable). Swap to naive PyTorch
        # only for export, then restore.
        import mmfreelm.layers.hgrn_bit as hgrn_bit_mod
        import mmfreelm.models.hgrn_bit.modeling_hgrn_bit as modeling_mod
        import mmfreelm.ops.hgrn.recurrent_fuse as recurrent_fuse_mod
        from mmfreelm.ops.hgrn.naive import onnx_recurrent_hgrn

        orig_rf = recurrent_fuse_mod.fused_recurrent_hgrn
        orig_hb = hgrn_bit_mod.fused_recurrent_hgrn
        orig_hb_swiglu = hgrn_bit_mod.swiglu
        orig_model_swiglu = modeling_mod.swiglu
        def export_swiglu(x, y):
            return (x * torch.sigmoid(x)) * y
        recurrent_fuse_mod.fused_recurrent_hgrn = onnx_recurrent_hgrn
        hgrn_bit_mod.fused_recurrent_hgrn = onnx_recurrent_hgrn
        hgrn_bit_mod.swiglu = export_swiglu
        modeling_mod.swiglu = export_swiglu
        print(f"[TRT] Exporting ONNX → {self.onnx_path}")
        # Trace at seq=1 so constant-folding does not bake in a fixed prompt length.
        dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    self.fwd,
                    (dummy,),
                    self.onnx_path,
                    opset_version=17,
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "seq"},
                        "logits": {0: "batch"},
                    },
                    do_constant_folding=False,
                    dynamo=False,
                )
        finally:
            recurrent_fuse_mod.fused_recurrent_hgrn = orig_rf
            hgrn_bit_mod.fused_recurrent_hgrn = orig_hb
            hgrn_bit_mod.swiglu = orig_hb_swiglu
            modeling_mod.swiglu = orig_model_swiglu
        print("[TRT] ONNX export done ✓")

    def _build_engine(self) -> bytes:
        builder = trt.Builder(self.logger)
        # TensorRT 10+ always uses explicit batch and removed this legacy enum.
        explicit_batch = getattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None
        )
        network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
        net = builder.create_network(network_flags)
        parser = trt.OnnxParser(net, self.logger)
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError(
                    "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                )
        cfg = builder.create_builder_config()
        # Jetson uses unified system memory. A 256 MiB builder workspace leaves
        # room for TensorRT tactic compilation on the 8 GiB Orin, at the cost
        # of a slower engine build.
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        # TensorRT 11 removed the legacy FP16 builder flag. It preserves the
        # FP16 precision encoded in the ONNX weights and casts automatically.
        if self.use_fp16:
            print("[TRT] FP16 ONNX weights; using TensorRT 11 default precision.")
        prof = builder.create_optimization_profile()
        prof.set_shape(
            "input_ids",
            min=(1, 1),
            opt=(max(1, self.max_batch // 2), max(1, self.max_seq // 2)),
            max=(self.max_batch, self.max_seq),
        )
        cfg.add_optimization_profile(prof)
        data = builder.build_serialized_network(net, cfg)
        if data is None:
            raise RuntimeError("TRT build failed – OOM?")
        return bytes(data)

    def _alloc_buffers(self):
        self._trt11_io = hasattr(self.engine, "num_io_tensors")
        if self._trt11_io:
            self._input_name = "input_ids"
            self._output_name = "logits"
            vocab = self.engine.get_tensor_shape(self._output_name)[-1]
            self._in_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self._input_name)))
            self._out_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self._output_name)))
        else:
            vocab = self.engine.get_binding_shape(
                self.engine.get_binding_index("logits")
            )[-1]
            self._in_dtype = np.int64
            self._out_dtype = np.float16
        self._vocab = vocab
        self._in_buf = cuda.pagelocked_empty(
            (self.max_batch * self.max_seq,), dtype=self._in_dtype
        )
        self._out_buf = cuda.pagelocked_empty(
            (self.max_batch * vocab,), dtype=self._out_dtype
        )
        self._d_in = cuda.mem_alloc(self._in_buf.nbytes)
        self._d_out = cuda.mem_alloc(self._out_buf.nbytes)
        self._stream = cuda.Stream()

    def _step(self, ids: torch.Tensor) -> torch.Tensor:
        b, s = ids.shape
        self._in_buf[: b * s] = ids.cpu().numpy().astype(self._in_dtype).ravel()
        if self._trt11_io:
            if not self.context.set_input_shape(self._input_name, (b, s)):
                raise RuntimeError(f"TensorRT rejected input shape {(b, s)}")
            self.context.set_tensor_address(self._input_name, int(self._d_in))
            self.context.set_tensor_address(self._output_name, int(self._d_out))
        else:
            self.context.set_binding_shape(0, (b, s))
        cuda.memcpy_htod_async(self._d_in, self._in_buf[: b * s], self._stream)
        if self._trt11_io:
            if not self.context.execute_async_v3(self._stream.handle):
                raise RuntimeError("TensorRT execution failed")
        else:
            self.context.execute_async_v2(
                [int(self._d_in), int(self._d_out)], self._stream.handle
            )
        n = b * self._vocab
        cuda.memcpy_dtoh_async(self._out_buf[:n], self._d_out, self._stream)
        self._stream.synchronize()
        return torch.from_numpy(self._out_buf[:n].copy()).view(b, self._vocab).to(
            device="cuda"
        )

    def generate(
        self,
        input_ids,
        max_length=32,
        do_sample=True,
        top_p=0.4,
        temperature=0.6,
        **_,
    ):
        return decode_loop(
            self._step, input_ids, max_length, do_sample, top_p, temperature
        )
