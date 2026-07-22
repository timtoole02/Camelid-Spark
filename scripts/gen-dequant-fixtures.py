#!/usr/bin/env py
"""Generate ggml dequant reference fixtures for the runnable lane (Phase 2 / Gate 2).

The reference side is **llama.cpp's own `gguf` Python package** (`gguf.quants`),
the maintained numpy port of ggml's dequantization kernels. For each covered v1
quant we emit a fixture under tests/fixtures/dequant/ holding the exact wire bytes
plus the reference f32 output (stored as u32 bit patterns so the Rust comparison is
bit-exact across JSON).

Sources, per format:
  F32, F16             -> numpy (IEEE-exact, identical to ggml's trivial paths)
  Q8_0, Q4_0           -> gguf.quants.quantize() of deterministic inputs
  Q4_K, Q5_K, Q6_K     -> synthetic but structurally-valid blocks: all integer fields
                          random (seeded); the f16 super-scales are set to a safe
                          finite range so dequant output is finite. A dequant
                          bit-exactness test is independent of byte provenance — the
                          same bytes go through gguf's reference and through our
                          decoder — so synthetic blocks exercise the full superblock
                          arithmetic without needing a multi-GB on-disk K-quant model.
                          (gguf.quants.quantize() does not implement K-quants; the
                          only on-disk K-quant model is too large to memmap on this
                          box — see BACKEND_ASKS RA-2.)

Determinism: fixed seeds; no timestamps. Re-running reproduces byte-identical fixtures.

Usage:  py scripts/gen-dequant-fixtures.py
"""
import importlib.metadata
import json
import os
import numpy as np
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType as Q, GGML_QUANT_SIZES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "tests", "fixtures", "dequant")
GGUF_VERSION = importlib.metadata.version("gguf")

# Byte offsets of the f16 super-scale fields within each K-quant block, so synthetic
# blocks can be sanitized to finite values (everything else is integer -> can't NaN).
KQUANT_F16_OFFSETS = {
    "Q4_K": [0, 2],    # d, dmin at block start (block_q4_K)
    "Q5_K": [0, 2],    # d, dmin at block start (block_q5_K)
    "Q6_K": [208],     # d at block end (block_q6_K: ql[128] qh[64] scales[16] d)
}


def f32_bits(arr):
    u = np.asarray(arr, dtype=np.float32).view(np.uint32)
    return [f"0x{int(v):08x}" for v in u]


def deterministic_input(n, seed):
    rng = np.random.default_rng(seed)
    ramp = (np.arange(n, dtype=np.float32) - n / 2) * 0.037
    noise = rng.standard_normal(n).astype(np.float32) * 0.5
    spikes = np.zeros(n, dtype=np.float32)
    step = max(1, n // 8)
    spikes[::step] = rng.uniform(-4, 4, size=spikes[::step].shape)
    return (ramp + noise + spikes).astype(np.float32)


def write_fixture(name, qtype, block_size, block_bytes, wire, ref_f32, source):
    n_elements = int(ref_f32.size)
    assert wire.dtype == np.uint8
    fixture = {
        "qtype": name,
        "ggml_type_id": int(qtype.value),
        "block_size": int(block_size),
        "block_bytes": int(block_bytes),
        "n_blocks": n_elements // block_size if block_size > 1 else n_elements,
        "n_elements": n_elements,
        "source": source,
        "reference": f"gguf=={GGUF_VERSION} (gguf.quants); F16/F32 via numpy",
        "quant_hex": wire.tobytes().hex(),
        "ref_f32_bits": f32_bits(ref_f32),
    }
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"  wrote {name:5} n_elem={n_elements:5} block_bytes={block_bytes:3} src={source}")
    return fixture


def gen_numpy_float(name, qtype, n, seed):
    x = deterministic_input(n, seed)
    if name == "F32":
        wire, ref = x.tobytes(), x
    elif name == "F16":
        h = x.astype(np.float16)
        wire, ref = h.tobytes(), h.astype(np.float32)
    else:
        raise ValueError(name)
    return write_fixture(name, qtype, 1, GGML_QUANT_SIZES[qtype][1],
                         np.frombuffer(wire, dtype=np.uint8), ref, "numpy")


def gen_gguf_quantize(name, qtype, n, seed):
    x = deterministic_input(n, seed).reshape(1, n)
    q = quantize(x, qtype).reshape(-1).astype(np.uint8)
    ref = dequantize(q.copy(), qtype).reshape(-1)
    bs, tb = GGML_QUANT_SIZES[qtype]
    return write_fixture(name, qtype, bs, tb, q, ref, "gguf.quants.quantize")


def safe_f16_bytes(rng):
    """A finite, moderate-magnitude f16 (either sign) -> 2 LE bytes."""
    mag = rng.uniform(0.002, 8.0)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return np.array([np.float16(sign * mag)], dtype=np.float16).tobytes()


def gen_synthetic_kquant(name, qtype, n_blocks, seed):
    bs, tb = GGML_QUANT_SIZES[qtype]
    offs = KQUANT_F16_OFFSETS[name]
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_blocks):
        b = bytearray(rng.integers(0, 256, size=tb, dtype=np.uint8).tobytes())
        for off in offs:
            b[off:off + 2] = safe_f16_bytes(rng)
        blocks.append(bytes(b))
    wire = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    ref = dequantize(wire.copy(), qtype).reshape(-1)
    assert np.all(np.isfinite(ref)), f"synthetic {name} produced non-finite output"
    return write_fixture(name, qtype, bs, tb, wire, ref, "synthetic:valid-blocks")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"generating dequant fixtures -> {OUT_DIR}")
    manifest = [
        gen_numpy_float("F32", Q.F32, 256, seed=1),
        gen_numpy_float("F16", Q.F16, 256, seed=2),
        gen_gguf_quantize("Q8_0", Q.Q8_0, 256, seed=3),
        gen_gguf_quantize("Q4_0", Q.Q4_0, 256, seed=4),
        gen_synthetic_kquant("Q4_K", Q.Q4_K, n_blocks=8, seed=5),
        gen_synthetic_kquant("Q5_K", Q.Q5_K, n_blocks=8, seed=6),
        gen_synthetic_kquant("Q6_K", Q.Q6_K, n_blocks=8, seed=7),
    ]
    idx = {
        "reference": f"gguf=={GGUF_VERSION}",
        "note": "ref_f32_bits are u32 bit patterns of the reference f32 output",
        "fixtures": [
            {k: m[k] for k in ("qtype", "source", "n_elements", "block_bytes")}
            for m in manifest
        ],
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(idx, fh, indent=1)
    print(f"done: {len(manifest)} fixtures + manifest.json")


if __name__ == "__main__":
    main()
