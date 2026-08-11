"""TensorRT / CUDA-graph inference path for MMfreeLM."""

from mmfreelm.tensorrt.generation import (
    CUDAGraphAccelerator,
    ModelForwardWrapper,
    ONNXTRTAccelerator,
    PureTorchFusedBitLinear,
    PureTorchRMSNorm,
    decode_loop,
    default_trt_cache_paths,
    patch_all_triton_ops,
    top_p_sample,
    trt_dependencies_available,
)

__all__ = [
    "CUDAGraphAccelerator",
    "ModelForwardWrapper",
    "ONNXTRTAccelerator",
    "PureTorchFusedBitLinear",
    "PureTorchRMSNorm",
    "decode_loop",
    "default_trt_cache_paths",
    "patch_all_triton_ops",
    "top_p_sample",
    "trt_dependencies_available",
]
