"""Round-trip tests for TQ1_0 / TQ2_0 packing (no full mmfreelm import)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "mmfreelm" / "ops" / "ternary_packing.py"


def _load_packing():
    spec = importlib.util.spec_from_file_location("ternary_packing", PACK_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ternary_packing"] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load_packing()


def _random_ternary(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-1, 2, shape, generator=g, dtype=torch.float16)


def test_tq2_roundtrip():
    w = _random_ternary((17, 64))
    packed = tp.pack_tq2_0(w)
    out = tp.unpack_tq2_0(packed, torch.float16, w.shape, w.numel())
    assert torch.equal(out, w), (out.flatten()[:8], w.flatten()[:8])


def test_tq1_roundtrip():
    w = _random_ternary((11, 48))
    packed = tp.pack_tq1_0(w)
    out = tp.unpack_tq1_0(packed, torch.float16, w.shape, w.numel())
    assert torch.equal(out, w), (out.flatten()[:16], w.flatten()[:16])


def test_enum_packed_flags():
    assert tp.CompressedType.NAIVE.is_packed
    assert tp.CompressedType.TQ1_0.is_packed
    assert tp.CompressedType.TQ2_0.is_packed
    assert not tp.CompressedType.FLOAT16.is_packed
    assert tp.CompressedType.NAIVE.label == "PACKED_2BIT"


def test_lookup_tables_match_reference():
    # TQ2 table vs bit extract
    for i in (0, 1, 2, 27, 85, 170, 255):
        for s in range(4):
            assert int(tp.TQ2_UNPACK_TABLE[i, s]) == (((i >> (2 * s)) & 3) - 1)
    # TQ1 table vs ggml decode
    pow3 = (1, 3, 9, 27, 81)
    for i in (0, 17, 81, 128, 242, 255):
        for n in range(5):
            q = (i * pow3[n]) & 0xFF
            xi = (q * 3) >> 8
            assert int(tp.TQ1_UNPACK_TABLE[i, n]) == xi - 1


if __name__ == "__main__":
    test_enum_packed_flags()
    test_lookup_tables_match_reference()
    test_tq2_roundtrip()
    test_tq1_roundtrip()
    # Microbench CPU unpack
    import time
    w = _random_ternary((4096, 4096))
    p2 = tp.pack_tq2_0(w)
    p1 = tp.pack_tq1_0(w)
    for name, packed, fn in (
        ("tq2", p2, tp.unpack_tq2_0),
        ("tq1", p1, tp.unpack_tq1_0),
    ):
        for _ in range(3):
            fn(packed, torch.float16, w.shape, w.numel())
        t0 = time.perf_counter()
        for _ in range(10):
            fn(packed, torch.float16, w.shape, w.numel())
        dt = (time.perf_counter() - t0) / 10
        print(f"{name} unpack avg {dt*1000:.2f} ms")
    print("ternary_packing tests OK")
