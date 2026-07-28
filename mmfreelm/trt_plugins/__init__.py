"""TensorRT plugin registrations for MMfreeLM custom ONNX ops."""

from mmfreelm.trt_plugins.ternary_matmul import register_ternary_matmul_plugin

__all__ = ["register_ternary_matmul_plugin"]
