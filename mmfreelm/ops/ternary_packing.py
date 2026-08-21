"""Ternary weight packing layouts (llama.cpp PR #8151 + legacy 2-bit).

TQ1_0 ~1.6875 bpw: 5 trits/byte (3^5 < 256) + 4-trit qh + f16 scale / 256.
TQ2_0 ~2.0625 bpw: 4 trits/byte (2 bits each) + f16 scale / 256.
NAIVE / PACKED_2BIT: legacy 4×2-bit/byte pack used by FusedBitLinear today.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

QK_K = 256
TQ1_QS = (QK_K - 4 * QK_K // 64) // 5  # 48
TQ1_QH = QK_K // 64  # 4
TQ1_BLOCK_BYTES = TQ1_QS + TQ1_QH + 2  # 54
TQ2_QS = QK_K // 4  # 64
TQ2_BLOCK_BYTES = TQ2_QS + 2  # 66
_POW3 = torch.tensor([1, 3, 9, 27, 81, 243], dtype=torch.int32)


class CompressedType(Enum):
    """How BitLinear / FusedBitLinear weights are stored before the matmul."""

    # Legacy compact 2-bit layout (4 trits per uint8). Kept as NAIVE for CSV compat.
    NAIVE = 1
    PACKED_2BIT = 1
    FP4 = 2
    INT8 = 3
    FLOAT16 = 4
    # llama.cpp ggml-quants ternary formats (PR #8151).
    TQ2_0 = 5
    TQ1_0 = 6

    @property
    def is_packed(self) -> bool:
        return self in (
            CompressedType.NAIVE,
            CompressedType.TQ1_0,
            CompressedType.TQ2_0,
        )

    @property
    def label(self) -> str:
        if self in (CompressedType.NAIVE, CompressedType.PACKED_2BIT):
            return "PACKED_2BIT"
        return self.name


def _as_ternary_trits(weights: torch.Tensor) -> torch.Tensor:
    """Map {-1,0,1} (or near) to uint8 trits {0,1,2}."""
    return (weights.round().clamp_(-1, 1).to(torch.int16) + 1).to(torch.uint8)


def _pad_flat(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Size, int]:
    flat = weights.reshape(-1).contiguous()
    n = flat.numel()
    pad = (QK_K - (n % QK_K)) % QK_K
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat, weights.shape, n


def _f16_one_bytes(device: torch.device) -> torch.Tensor:
    return torch.tensor([1.0], dtype=torch.float16).view(torch.uint8).to(device)


def pack_tq2_0(ternary_weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,1} weights into TQ2_0 blocks (qs[64] + f16 d)."""
    flat, orig_shape, n = _pad_flat(ternary_weights)
    trits = _as_ternary_trits(flat).to(torch.int32)
    nb = trits.numel() // QK_K
    device = trits.device
    # [nb, 2 halves, 4 trit-slots, 32]
    blocks = trits.view(nb, 2, 4, 32)
    qs = (
        blocks[:, :, 0, :]
        | (blocks[:, :, 1, :] << 2)
        | (blocks[:, :, 2, :] << 4)
        | (blocks[:, :, 3, :] << 6)
    ).to(torch.uint8).reshape(nb, TQ2_QS)
    out = torch.zeros(nb, TQ2_BLOCK_BYTES, dtype=torch.uint8, device=device)
    out[:, :TQ2_QS] = qs
    out[:, TQ2_QS : TQ2_QS + 2] = _f16_one_bytes(device)
    packed = out.reshape(-1)
    packed._mmfree_orig_shape = orig_shape  # type: ignore[attr-defined]
    packed._mmfree_orig_numel = n  # type: ignore[attr-defined]
    packed._mmfree_compressed_type = CompressedType.TQ2_0  # type: ignore[attr-defined]
    return packed


def unpack_tq2_0(
    packed: torch.Tensor,
    dtype: torch.dtype,
    original_shape: Optional[torch.Size] = None,
    original_numel: Optional[int] = None,
) -> torch.Tensor:
    orig_shape = original_shape or getattr(packed, "_mmfree_orig_shape", None)
    orig_numel = original_numel or getattr(packed, "_mmfree_orig_numel", None)
    blocks = packed.view(-1, TQ2_BLOCK_BYTES)
    nb = blocks.shape[0]
    qs = blocks[:, :TQ2_QS].to(torch.int32).view(nb, 2, 32)
    slots = torch.stack([(qs >> (2 * s)) & 3 for s in range(4)], dim=2)  # [nb,2,4,32]
    flat = (slots - 1).reshape(nb * QK_K).to(dtype)
    # Optional per-block scale (usually 1.0 for already-ternary weights).
    d = blocks[:, TQ2_QS : TQ2_QS + 2].contiguous().view(torch.float16).to(torch.float32)
    if not torch.allclose(d, torch.ones_like(d)):
        flat = flat.view(nb, QK_K) * d.view(nb, 1)
        flat = flat.reshape(-1).to(dtype)
    if orig_numel is None:
        orig_numel = flat.numel()
    flat = flat[:orig_numel]
    if orig_shape is not None:
        return flat.view(orig_shape)
    return flat


def _pack_base3_group(trits_group: torch.Tensor) -> torch.Tensor:
    """Pack last-dim groups of 5 trits {0,1,2} into bytes (ceil q*256/243).

    trits_group: [..., 5]
    """
    q = trits_group[..., 0].to(torch.int32)
    for i in range(1, 5):
        q = q * 3 + trits_group[..., i].to(torch.int32)
    return ((q * 256 + 242) // 243).to(torch.uint8)


def _unpack_base3_byte(q_bytes: torch.Tensor, n_trits: int) -> torch.Tensor:
    """Decode n_trits from packed bytes via multiply-by-3^k then (q*3)>>8.

    Returns [..., n_trits] int16 in {-1,0,1} (before scale).
    """
    device = q_bytes.device
    pow3 = _POW3.to(device)
    out = []
    q0 = q_bytes.to(torch.int32)
    for n in range(n_trits):
        q = (q0 * pow3[n]) & 0xFF
        xi = (q * 3) >> 8
        out.append(xi - 1)
    return torch.stack(out, dim=-1).to(torch.int16)


def pack_tq1_0(ternary_weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,1} weights into TQ1_0 blocks (qs[48]+qh[4]+f16 d)."""
    flat, orig_shape, n = _pad_flat(ternary_weights)
    trits = _as_ternary_trits(flat).to(torch.int32)
    nb = trits.numel() // QK_K
    device = trits.device
    blocks = trits.view(nb, QK_K)
    out = torch.zeros(nb, TQ1_BLOCK_BYTES, dtype=torch.uint8, device=device)

    # First 160 elements -> qs[0:32] as 5 x 32
    g0 = blocks[:, 0:160].view(nb, 5, 32).transpose(1, 2)  # [nb,32,5]
    out[:, 0:32] = _pack_base3_group(g0)
    # Next 80 elements -> qs[32:48] as 5 x 16
    g1 = blocks[:, 160:240].view(nb, 5, 16).transpose(1, 2)  # [nb,16,5]
    out[:, 32:48] = _pack_base3_group(g1)
    # Last 16 elements -> qh[4] as 4 x 4, with MSB shift (*3)
    g2 = blocks[:, 240:256].view(nb, 4, 4).transpose(1, 2)  # [nb,4,4]
    q = g2[..., 0]
    for i in range(1, 4):
        q = q * 3 + g2[..., i]
    q = q * 3
    out[:, 48:52] = ((q * 256 + 242) // 243).to(torch.uint8)

    out[:, 52:54] = _f16_one_bytes(device)
    packed = out.reshape(-1)
    packed._mmfree_orig_shape = orig_shape  # type: ignore[attr-defined]
    packed._mmfree_orig_numel = n  # type: ignore[attr-defined]
    packed._mmfree_compressed_type = CompressedType.TQ1_0  # type: ignore[attr-defined]
    return packed


def unpack_tq1_0(
    packed: torch.Tensor,
    dtype: torch.dtype,
    original_shape: Optional[torch.Size] = None,
    original_numel: Optional[int] = None,
) -> torch.Tensor:
    orig_shape = original_shape or getattr(packed, "_mmfree_orig_shape", None)
    orig_numel = original_numel or getattr(packed, "_mmfree_orig_numel", None)
    blocks = packed.view(-1, TQ1_BLOCK_BYTES)
    nb = blocks.shape[0]
    device = packed.device

    qs0 = _unpack_base3_byte(blocks[:, 0:32], 5)  # [nb,32,5]
    part0 = qs0.transpose(1, 2).reshape(nb, 160)
    qs1 = _unpack_base3_byte(blocks[:, 32:48], 5)  # [nb,16,5]
    part1 = qs1.transpose(1, 2).reshape(nb, 80)
    qh = _unpack_base3_byte(blocks[:, 48:52], 4)  # [nb,4,4]
    part2 = qh.transpose(1, 2).reshape(nb, 16)

    flat = torch.cat([part0, part1, part2], dim=1).reshape(nb * QK_K).to(torch.float32)
    d = blocks[:, 52:54].contiguous().view(torch.float16).to(torch.float32)
    if not torch.allclose(d, torch.ones_like(d)):
        flat = flat.view(nb, QK_K) * d.view(nb, 1)
        flat = flat.reshape(-1)
    flat = flat.round().to(dtype)
    if orig_numel is None:
        orig_numel = flat.numel()
    flat = flat[:orig_numel]
    if orig_shape is not None:
        return flat.view(orig_shape)
    return flat


def pack_for_type(ternary_weights: torch.Tensor, compressed_type: CompressedType) -> torch.Tensor:
    if compressed_type == CompressedType.TQ1_0:
        return pack_tq1_0(ternary_weights)
    if compressed_type == CompressedType.TQ2_0:
        return pack_tq2_0(ternary_weights)
    raise ValueError(f"pack_for_type does not handle {compressed_type}")


def unpack_for_type(
    packed: torch.Tensor,
    dtype: torch.dtype,
    compressed_type: CompressedType,
    original_shape: Optional[torch.Size] = None,
    original_numel: Optional[int] = None,
) -> torch.Tensor:
    if compressed_type == CompressedType.TQ1_0:
        return unpack_tq1_0(packed, dtype, original_shape, original_numel)
    if compressed_type == CompressedType.TQ2_0:
        return unpack_tq2_0(packed, dtype, original_shape, original_numel)
    raise ValueError(f"unpack_for_type does not handle {compressed_type}")
