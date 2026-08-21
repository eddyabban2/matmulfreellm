"""Ternary weight packing layouts (llama.cpp PR #8151 + legacy 2-bit).

TQ1_0 ~1.6875 bpw: 5 trits/byte (3^5 < 256) + 4-trit qh + f16 scale / 256.
TQ2_0 ~2.0625 bpw: 4 trits/byte (2 bits each) + f16 scale / 256.
NAIVE / PACKED_2BIT: legacy 4×2-bit/byte pack used by FusedBitLinear today.

Unpack hot path uses:
  - compile-time 256-entry trit lookup tables (parallel gather)
  - torch.compile on the decode cores
  - optional Triton kernels when CUDA + Triton are available
"""
from __future__ import annotations

from enum import Enum
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

QK_K = 256
TQ1_QS = (QK_K - 4 * QK_K // 64) // 5  # 48
TQ1_QH = QK_K // 64  # 4
TQ1_BLOCK_BYTES = TQ1_QS + TQ1_QH + 2  # 54
TQ2_QS = QK_K // 4  # 64
TQ2_BLOCK_BYTES = TQ2_QS + 2  # 66

# ---------------------------------------------------------------------------
# Compile-time lookup tables (built once on import; moved to device lazily).
# ---------------------------------------------------------------------------

def _build_tq2_table() -> torch.Tensor:
    """byte -> 4 trits in {-1,0,1}, little-endian 2-bit fields."""
    rows = []
    for i in range(256):
        rows.append([((i >> (2 * s)) & 3) - 1 for s in range(4)])
    return torch.tensor(rows, dtype=torch.int8)


def _build_tq1_table() -> torch.Tensor:
    """byte -> 5 trits in {-1,0,1} via ggml fixed-point base-3 decode."""
    pow3 = (1, 3, 9, 27, 81)
    rows = []
    for i in range(256):
        row = []
        for n in range(5):
            q = (i * pow3[n]) & 0xFF
            xi = (q * 3) >> 8
            row.append(xi - 1)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int8)


TQ2_UNPACK_TABLE = _build_tq2_table()  # [256, 4]
TQ1_UNPACK_TABLE = _build_tq1_table()  # [256, 5]
_TABLE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _tables_for(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index)
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        hit = (TQ2_UNPACK_TABLE.to(device), TQ1_UNPACK_TABLE.to(device))
        _TABLE_CACHE[key] = hit
    return hit


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


def _block_scales_f32(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode contiguous uint8 pairs as float16 scales -> float32 [nb] (no CPU sync)."""
    # scale_bytes: [nb, 2] uint8
    return scale_bytes.contiguous().view(torch.float16).reshape(-1).to(torch.float32)


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def pack_tq2_0(ternary_weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,1} weights into TQ2_0 blocks (qs[64] + f16 d)."""
    flat, orig_shape, n = _pad_flat(ternary_weights)
    trits = _as_ternary_trits(flat).to(torch.int32)
    nb = trits.numel() // QK_K
    device = trits.device
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
    packed._mmfree_unit_scale = True  # type: ignore[attr-defined]
    return packed


def _pack_base3_group(trits_group: torch.Tensor) -> torch.Tensor:
    """Pack last-dim groups of 5 trits {0,1,2} into bytes (ceil q*256/243)."""
    q = trits_group[..., 0].to(torch.int32)
    for i in range(1, 5):
        q = q * 3 + trits_group[..., i].to(torch.int32)
    return ((q * 256 + 242) // 243).to(torch.uint8)


def pack_tq1_0(ternary_weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,1} weights into TQ1_0 blocks (qs[48]+qh[4]+f16 d)."""
    flat, orig_shape, n = _pad_flat(ternary_weights)
    trits = _as_ternary_trits(flat).to(torch.int32)
    nb = trits.numel() // QK_K
    device = trits.device
    blocks = trits.view(nb, QK_K)
    out = torch.zeros(nb, TQ1_BLOCK_BYTES, dtype=torch.uint8, device=device)

    g0 = blocks[:, 0:160].view(nb, 5, 32).transpose(1, 2)
    out[:, 0:32] = _pack_base3_group(g0)
    g1 = blocks[:, 160:240].view(nb, 5, 16).transpose(1, 2)
    out[:, 32:48] = _pack_base3_group(g1)
    g2 = blocks[:, 240:256].view(nb, 4, 4).transpose(1, 2)
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
    packed._mmfree_unit_scale = True  # type: ignore[attr-defined]
    return packed


# ---------------------------------------------------------------------------
# Unpack cores (table gather + torch.compile). Triton optional acceleration.
# ---------------------------------------------------------------------------

def _unpack_tq2_torch(qs: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """qs [nb, 64] uint8 -> trits [nb, 256] int8 via parallel table gather."""
    nb = qs.shape[0]
    # [nb, 2, 32, 4] -> [nb, 2, 4, 32] -> [nb, 256]
    decoded = table[qs.view(nb, 2, 32).long()]
    return decoded.permute(0, 1, 3, 2).reshape(nb, QK_K)


def _unpack_tq1_torch(
    qs0: torch.Tensor,
    qs1: torch.Tensor,
    qh: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
    """Decode TQ1 block fields -> [nb, 256] int8 via parallel table gather."""
    nb = qs0.shape[0]
    part0 = table[qs0.long()].permute(0, 2, 1).reshape(nb, 160)          # [nb,32,5]
    part1 = table[qs1.long()].permute(0, 2, 1).reshape(nb, 80)           # [nb,16,5]
    part2 = table[qh.long()][..., :4].permute(0, 2, 1).reshape(nb, 16)  # [nb,4,4]
    return torch.cat([part0, part1, part2], dim=1)


class _LazyCompiled:
    """torch.compile on CUDA when the backend works; eager table gather otherwise.

    CPU inductor needs a working C++ toolchain (often missing on Windows), and a
    256-entry gather is already memory-bound / well vectorized in eager PyTorch.
    """

    def __init__(self, fn):
        self._fn = fn
        self._compiled = None
        self._disabled = False

    def __call__(self, *args, **kwargs):
        if self._disabled:
            return self._fn(*args, **kwargs)
        device = args[0].device if args else torch.device("cpu")
        if device.type != "cuda":
            return self._fn(*args, **kwargs)
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._fn, fullgraph=True, dynamic=True)
            except Exception:
                self._disabled = True
                return self._fn(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception:
            self._disabled = True
            self._compiled = None
            return self._fn(*args, **kwargs)


_unpack_tq2_compiled = _LazyCompiled(_unpack_tq2_torch)
_unpack_tq1_compiled = _LazyCompiled(_unpack_tq1_torch)


_TRITON_TQ2 = None


def _try_triton_tq2():
    """Lazy-init Triton TQ2 unpack (one program / block). Returns None if unavailable."""
    global _TRITON_TQ2
    if _TRITON_TQ2 is not None:
        return _TRITON_TQ2 if _TRITON_TQ2 is not False else None
    try:
        import triton
        import triton.language as tl
    except Exception:
        _TRITON_TQ2 = False
        return None

    @triton.jit
    def _tq2_kernel(packed_ptr, out_ptr, nb, BLOCK_BYTES: tl.constexpr):
        bid = tl.program_id(0)
        if bid >= nb:
            return
        # Load 64 qs bytes for this block.
        offs = tl.arange(0, 64)
        qs = tl.load(packed_ptr + bid * BLOCK_BYTES + offs).to(tl.int32)
        # Expand to 256 trits: layout matches torch path (2 halves × 4 slots × 32).
        # half h in {0,1}, slot s in {0,1,2,3}, m in {0..31}
        # byte index = h*32 + m; trit = ((qs >> 2s) & 3) - 1
        # out index = h*128 + s*32 + m
        for h in tl.static_range(2):
            for s in tl.static_range(4):
                m = tl.arange(0, 32)
                byte = tl.load(packed_ptr + bid * BLOCK_BYTES + h * 32 + m).to(tl.int32)
                trit = ((byte >> (2 * s)) & 3) - 1
                tl.store(out_ptr + bid * 256 + h * 128 + s * 32 + m, trit.to(tl.int8))

    def _run(blocks: torch.Tensor) -> torch.Tensor:
        nb = blocks.shape[0]
        out = torch.empty(nb, QK_K, dtype=torch.int8, device=blocks.device)
        _tq2_kernel[(nb,)](blocks, out, nb, TQ2_BLOCK_BYTES, num_warps=4)
        return out

    _TRITON_TQ2 = _run
    return _run


def unpack_tq2_0(
    packed: torch.Tensor,
    dtype: torch.dtype,
    original_shape: Optional[torch.Size] = None,
    original_numel: Optional[int] = None,
    unit_scale: Optional[bool] = None,
) -> torch.Tensor:
    orig_shape = original_shape or getattr(packed, "_mmfree_orig_shape", None)
    orig_numel = original_numel or getattr(packed, "_mmfree_orig_numel", None)
    if unit_scale is None:
        unit_scale = bool(getattr(packed, "_mmfree_unit_scale", False))

    blocks = packed.view(-1, TQ2_BLOCK_BYTES)
    nb = blocks.shape[0]
    tq2_table, _ = _tables_for(packed.device)

    triton_run = None
    if packed.is_cuda and os.environ.get("MMFREE_TRITON_UNPACK", "0") == "1":
        triton_run = _try_triton_tq2()
    if triton_run is not None:
        try:
            flat = triton_run(blocks)
        except Exception:
            global _TRITON_TQ2
            _TRITON_TQ2 = False
            flat = _unpack_tq2_compiled(blocks[:, :TQ2_QS], tq2_table)
    else:
        flat = _unpack_tq2_compiled(blocks[:, :TQ2_QS], tq2_table)

    if not unit_scale:
        d = _block_scales_f32(blocks[:, TQ2_QS : TQ2_QS + 2])
        flat = (flat.to(torch.float32) * d.view(nb, 1)).round()

    flat = flat.reshape(-1).to(dtype)
    if orig_numel is None:
        orig_numel = flat.numel()
    flat = flat[:orig_numel]
    if orig_shape is not None:
        return flat.view(orig_shape)
    return flat


def unpack_tq1_0(
    packed: torch.Tensor,
    dtype: torch.dtype,
    original_shape: Optional[torch.Size] = None,
    original_numel: Optional[int] = None,
    unit_scale: Optional[bool] = None,
) -> torch.Tensor:
    orig_shape = original_shape or getattr(packed, "_mmfree_orig_shape", None)
    orig_numel = original_numel or getattr(packed, "_mmfree_orig_numel", None)
    if unit_scale is None:
        unit_scale = bool(getattr(packed, "_mmfree_unit_scale", False))

    blocks = packed.view(-1, TQ1_BLOCK_BYTES)
    nb = blocks.shape[0]
    _, tq1_table = _tables_for(packed.device)

    flat = _unpack_tq1_compiled(
        blocks[:, 0:32],
        blocks[:, 32:48],
        blocks[:, 48:52],
        tq1_table,
    )

    if not unit_scale:
        d = _block_scales_f32(blocks[:, 52:54])
        flat = (flat.to(torch.float32) * d.view(nb, 1)).round()

    flat = flat.reshape(-1).to(dtype)
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
    unit_scale: Optional[bool] = None,
) -> torch.Tensor:
    if unit_scale is None:
        unit_scale = bool(getattr(packed, "_mmfree_unit_scale", True))
    if compressed_type == CompressedType.TQ1_0:
        return unpack_tq1_0(
            packed, dtype, original_shape, original_numel, unit_scale=unit_scale
        )
    if compressed_type == CompressedType.TQ2_0:
        return unpack_tq2_0(
            packed, dtype, original_shape, original_numel, unit_scale=unit_scale
        )
    raise ValueError(f"unpack_for_type does not handle {compressed_type}")
