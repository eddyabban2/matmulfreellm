# Shared TensorRT / CUDA-graph path for MMfreeLM (ONNX export + JetPack TRT).
# Used by generate.py and benchmark_trt.py.

import hashlib
import gc
import json
import os
import sys
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.ops.fusedbitnet import weight_quant

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


def trt_builder_available() -> tuple:
    """
    True if trt.Builder() initializes (links CUDA correctly).
    On Jetson, pip tensorrt may fail with CUDA error 35 while PyTorch CUDA still works;
    prefer JetPack (apt) TensorRT + matching Python bindings.
    """
    if not trt_dependencies_available():
        return False, "tensorrt or pycuda not importable"
    try:
        log = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(log, "")
        b = trt.Builder(log)
        if b is None:
            return False, "trt.Builder returned None"
        return True, "ok"
    except TypeError as e:
        return (
            False,
            f"{e!r} — usually CUDA/driver mismatch (e.g. error 35); "
            "PyTorch may still run on GPU while TensorRT Builder does not.",
        )
    except Exception as e:
        return False, str(e)


def default_trt_cache_paths(model_name: str) -> Tuple[str, str]:
    key = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/mmfreelm_{key}.engine", f"/tmp/mmfreelm_{key}.onnx"


@contextmanager
def _cpu_traceable_swiglu():
    """
    SwiGLU in mmfreelm uses cuda.jiterator, which fails on CPU tensors.
    For CPU ONNX tracing, temporarily swap in element-wise PyTorch Swish(x)*y.
    """
    import mmfreelm.layers.hgrn_bit as hgrn_bit_layer
    import mmfreelm.models.hgrn_bit.modeling_hgrn_bit as modeling_hgrn_bit
    import mmfreelm.modules.activations as act_mod

    def pure_swiglu(x, y):
        return x * torch.sigmoid(x.float()).to(dtype=x.dtype) * y

    def pure_swiglu_linear(x, y, weight, bias):
        z = pure_swiglu(x, y)
        return F.linear(z.to(weight.dtype), weight, bias)

    old = (
        act_mod.swiglu,
        act_mod.swiglu_linear,
        hgrn_bit_layer.swiglu,
        modeling_hgrn_bit.swiglu,
        modeling_hgrn_bit.swiglu_linear,
    )
    act_mod.swiglu = pure_swiglu
    act_mod.swiglu_linear = pure_swiglu_linear
    hgrn_bit_layer.swiglu = pure_swiglu
    modeling_hgrn_bit.swiglu = pure_swiglu
    modeling_hgrn_bit.swiglu_linear = pure_swiglu_linear
    try:
        yield
    finally:
        act_mod.swiglu = old[0]
        act_mod.swiglu_linear = old[1]
        hgrn_bit_layer.swiglu = old[2]
        modeling_hgrn_bit.swiglu = old[3]
        modeling_hgrn_bit.swiglu_linear = old[4]


# ── Pure-PyTorch replacements for Triton-backed ops ──────────────────────────


class PureTorchRMSNorm(nn.Module):
    """Replaces flash-attn / mmfreelm Triton RMSNorm."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.weight = src.weight
        self.bias = getattr(src, "bias", None)
        self.eps = getattr(src, "eps", 1e-6)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual.to(x.dtype)
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(orig) * self.weight
        if self.bias is not None:
            x = x + self.bias
        return (x, residual) if prenorm else x


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


class PureTorchFusedBitLinear(nn.Module):
    """RMSNorm(x) → ternary_quantize(weight) → F.linear"""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.weight = src.weight
        self.bias = getattr(src, "bias", None)
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
            wq = ternary_quantize(self.weight.detach().float()).to(
                dtype=self.weight.dtype, device=self.weight.device
            )
        self.register_buffer("_weight_quant_eval", wq, persistent=False)

    def forward(self, x):
        if self.has_norm:
            orig = x.dtype
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.norm_eps)
            x = xf.to(orig) * self.norm_weight
            if self.norm_bias is not None:
                x = x + self.norm_bias
        if self.training:
            w_q = ternary_quantize(self.weight.float()).to(x.dtype)
        else:
            w_q = self._weight_quant_eval.to(dtype=x.dtype)
        return F.linear(x, w_q, self.bias)


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
        f"{linear_replaced} fused BitLinear → PyTorch (eval: cached ternary weights)."
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


def export_mmfreelm_onnx_for_trt(
    model_name: str,
    onnx_path: str,
    *,
    dummy_seq: int = 8,
) -> None:
    """
    Load the HF checkpoint on CPU, patch Triton ops, and export ONNX.
    Writes onnx_path and onnx_path + ".meta.json" (vocab_size, num_params for TRT runtime).
    Frees GPU VRAM so TensorRT can build the engine afterward on Jetson-class devices.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    export_dtype = torch.float16
    if os.environ.get("MMFREELM_EXPORT_ONNX_FP32", "").lower() in ("1", "true", "yes"):
        export_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=export_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    npar = sum(p.numel() for p in model.parameters())
    print(
        f"[TRT] CPU export dtype={export_dtype} (set MMFREELM_EXPORT_ONNX_FP32=1 for float32).",
        flush=True,
    )
    model = patch_all_triton_ops(model)
    wrapped = ModelForwardWrapper(model)
    import mmfreelm.layers.hgrn_bit as hgrn_bit_mod
    import mmfreelm.ops.hgrn.recurrent_fuse as recurrent_fuse_mod
    from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn

    orig_rf = recurrent_fuse_mod.fused_recurrent_hgrn
    orig_hb = hgrn_bit_mod.fused_recurrent_hgrn
    recurrent_fuse_mod.fused_recurrent_hgrn = naive_recurrent_hgrn
    hgrn_bit_mod.fused_recurrent_hgrn = naive_recurrent_hgrn
    dummy = torch.zeros((1, dummy_seq), dtype=torch.long)
    ddir = os.path.dirname(onnx_path)
    if ddir:
        os.makedirs(ddir, exist_ok=True)
    try:
        with _cpu_traceable_swiglu(), torch.no_grad():
            torch.onnx.export(
                wrapped,
                (dummy,),
                onnx_path,
                opset_version=17,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                do_constant_folding=True,
            )
    finally:
        recurrent_fuse_mod.fused_recurrent_hgrn = orig_rf
        hgrn_bit_mod.fused_recurrent_hgrn = orig_hb
    meta_path = onnx_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {"vocab_size": int(cfg.vocab_size), "num_params": int(npar)},
            f,
        )
    del model, wrapped
    torch.cuda.empty_cache()
    print(f"[TRT] CPU ONNX + meta written → {onnx_path} ({meta_path})")


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
        model: Optional[nn.Module],
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

        ok, msg = trt_builder_available()
        if not ok:
            raise RuntimeError(
                f"TensorRT Builder not usable: {msg}\n"
                "Inference is NOT using TensorRT until this is fixed."
            )

        self.max_batch = max_batch
        self.max_seq = max_seq
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        meta_path = self.onnx_path + ".meta.json"
        if model is not None:
            if rebuild:
                if os.path.exists(self.engine_path):
                    os.remove(self.engine_path)
                for p in (self.onnx_path, meta_path):
                    if os.path.exists(p):
                        os.remove(p)
            model = patch_all_triton_ops(model)
            self.fwd = ModelForwardWrapper(model)
            self._vocab_only = None
            self._gflops_param_count = sum(p.numel() for p in model.parameters())
        else:
            if rebuild and os.path.exists(self.engine_path):
                os.remove(self.engine_path)
            if not os.path.isfile(meta_path):
                raise RuntimeError(
                    f"Lightweight TRT init needs {meta_path} "
                    "(from export_mmfreelm_onnx_for_trt)."
                )
            with open(meta_path) as f:
                meta = json.load(f)
            self._vocab_only = int(meta["vocab_size"])
            self._gflops_param_count = int(meta["num_params"])
            self.fwd = None

        self.engine = self._load_or_build()
        self._input_name, self._output_name = self._discover_io_tensors()
        self.context = self.engine.create_execution_context()
        self._alloc_buffers()
        print(
            "[TRT] ✓ RUNTIME_ACTIVE: TensorRT engine + IExecutionContext; "
            "decode uses execute_async_v3 (TensorRT 10 tensor API)."
        )

    def parameters(self):
        if self.fwd is None:
            return iter(())
        return self.fwd.model.parameters()

    def _discover_io_tensors(self) -> Tuple[str, str]:
        """TensorRT 10+ uses named I/O tensors (bindings API removed)."""
        eng = self.engine
        inputs: list = []
        outputs: list = []
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            mode = eng.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 TRT input and 1 output; got in={inputs} out={outputs}"
            )
        print(f"[TRT] I/O tensors: input={inputs[0]!r} output={outputs[0]!r}")
        return inputs[0], outputs[0]

    def _load_or_build(self):
        if os.path.exists(self.engine_path):
            print(f"[TRT] Loading cached engine: {self.engine_path}")
            with open(self.engine_path, "rb") as f:
                return trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        print("[TRT] Building engine …")
        if not os.path.exists(self.onnx_path):
            if self.fwd is None:
                raise RuntimeError(
                    f"No ONNX at {self.onnx_path}; "
                    "run export_mmfreelm_onnx_for_trt(...) first."
                )
            self._export_onnx()
        data = self._build_engine()
        with open(self.engine_path, "wb") as f:
            f.write(data)
        print(f"[TRT] Cached → {self.engine_path}")
        return trt.Runtime(self.logger).deserialize_cuda_engine(data)

    def _export_onnx(self):
        if self.fwd is None:
            raise RuntimeError("ONNX export requires a PyTorch model (use CPU export helper).")
        # hgrn_bit imports fused_recurrent_hgrn by name; ONNX trace with seq>1
        # otherwise hits Triton autotune (not traceable). Swap to naive PyTorch
        # only for export, then restore.
        import mmfreelm.layers.hgrn_bit as hgrn_bit_mod
        import mmfreelm.ops.hgrn.recurrent_fuse as recurrent_fuse_mod
        from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn

        orig_rf = recurrent_fuse_mod.fused_recurrent_hgrn
        orig_hb = hgrn_bit_mod.fused_recurrent_hgrn
        recurrent_fuse_mod.fused_recurrent_hgrn = naive_recurrent_hgrn
        hgrn_bit_mod.fused_recurrent_hgrn = naive_recurrent_hgrn
        print(f"[TRT] Exporting ONNX → {self.onnx_path}")
        dummy = torch.zeros((1, 8), dtype=torch.long, device="cuda")
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
                    do_constant_folding=True,
                )
        finally:
            recurrent_fuse_mod.fused_recurrent_hgrn = orig_rf
            hgrn_bit_mod.fused_recurrent_hgrn = orig_hb
        print("[TRT] ONNX export done ✓")
        meta_path = self.onnx_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "vocab_size": int(self.fwd.model.config.vocab_size),
                    "num_params": int(
                        sum(p.numel() for p in self.fwd.model.parameters())
                    ),
                },
                f,
            )

    def _build_engine(self) -> bytes:
        builder = trt.Builder(self.logger)
        net = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(net, self.logger)
        # PyTorch may write initializers as external files in the ONNX directory.
        # TensorRT resolves those paths relative to process cwd, not the ONNX path.
        onnx_abs = os.path.abspath(self.onnx_path)
        onnx_dir = os.path.dirname(onnx_abs) or "."
        onnx_file = os.path.basename(onnx_abs)
        prev = os.getcwd()
        onnx_blob = None
        try:
            os.chdir(onnx_dir)
            with open(onnx_file, "rb") as f:
                onnx_blob = f.read()
            if not parser.parse(onnx_blob):
                raise RuntimeError(
                    "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                )
        finally:
            os.chdir(prev)
        if onnx_blob is not None:
            del onnx_blob
        gc.collect()
        # Large workspace makes tactics try big scratch buffers during compile on Jetson;
        # 2GiB often OOMs unified memory during autotuning.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        cfg = builder.create_builder_config()
        ws_bytes = int(os.environ.get("MMFREELM_TRT_WORKSPACE_MB", "384")) << 20
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        if os.environ.get("MMFREELM_TRT_MINIMAL_TACTICS", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            cfg.set_tactic_sources(
                int(trt.TacticSource.CUBLAS) | int(trt.TacticSource.CUBLAS_LT)
            )
            print("[TRT] Minimal tactic sources: CUBLAS | CUBLAS_LT.")
        if (
            self.use_fp16
            and builder.platform_has_fast_fp16
            and os.environ.get("MMFREELM_TRT_FP32", "").lower() not in ("1", "true", "yes")
        ):
            cfg.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled.")
        else:
            print("[TRT] FP32 engine (set MMFREELM_TRT_FP32=0 to allow FP16).")
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
        return bytes(memoryview(data))

    def _alloc_buffers(self):
        vocab = (
            int(self.fwd.model.config.vocab_size)
            if self.fwd is not None
            else int(self._vocab_only)
        )
        self._vocab = vocab
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self._input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self._output_name))
        self._in_np_dtype = in_dtype
        self._out_np_dtype = out_dtype
        in_elems = self.max_batch * self.max_seq
        out_elems = self.max_batch * vocab
        self._in_buf = cuda.pagelocked_empty((in_elems,), dtype=in_dtype)
        self._out_buf = cuda.pagelocked_empty((out_elems,), dtype=out_dtype)
        self._d_in = cuda.mem_alloc(self._in_buf.nbytes)
        self._d_out = cuda.mem_alloc(self._out_buf.nbytes)
        self._stream = cuda.Stream()

    def _step(self, ids: torch.Tensor) -> torch.Tensor:
        b, s = ids.shape
        if not self.context.set_input_shape(self._input_name, (b, s)):
            raise RuntimeError(
                f"TRT set_input_shape({self._input_name!r}, {(b, s)}) failed"
            )
        ids_np = ids.detach().cpu().numpy().astype(self._in_np_dtype, copy=False)
        self._in_buf[: b * s] = ids_np.ravel()
        cuda.memcpy_htod_async(self._d_in, self._in_buf[: b * s], self._stream)
        self.context.set_tensor_address(self._input_name, int(self._d_in))
        self.context.set_tensor_address(self._output_name, int(self._d_out))
        ok = self.context.execute_async_v3(self._stream.handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        out_elems = int(np.prod(out_shape))
        cuda.memcpy_dtoh_async(self._out_buf[:out_elems], self._d_out, self._stream)
        self._stream.synchronize()
        host = self._out_buf[:out_elems].copy()
        t = torch.from_numpy(host).view(out_shape)
        if t.dtype != torch.float16:
            t = t.to(torch.float16)
        return t.cuda()

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
