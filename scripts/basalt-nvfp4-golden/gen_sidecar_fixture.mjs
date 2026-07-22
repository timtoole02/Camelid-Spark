// BASALT Amendment 3 §1.2 + §2.6 — synthetic NVFP4 refusal-trip fixtures.
//
// Provenance: §1.2 pair authored for the Amendment 3 closure commit
// (2026-07-16, S1); §2.6 quartet added by stage S2 (invariant-lane matrix),
// same day, same branch basalt/phase3-cpu-eval. Byte layout follows the GGUF
// v3 reader in src/gguf/reader.rs (magic/version/counts, string = u64 len +
// utf8, tensor descriptor = name/n_dims/dims i64/type i32/offset u64, data
// section aligned to general.alignment default 32). NVFP4 = pin type id 40,
// 64-element/36-byte superblock {d[4] UE4M3, qs[32]} (D-B1). Output is fully
// deterministic: no timestamps, no randomness — re-running MUST reproduce
// byte-identical files (shas pinned in tests/nvfp4_wire_lane_refusals.rs and
// tests/invariant_matrix_binding.rs, plus SHA256SUMS alongside). The S1 pair's
// bytes are FROZEN: the S2 refactor (extraKvs) emits them byte-identically
// (extraKvs=[] keeps metadata_count=1 and the single architecture KV).
//
// S1 pair (§1.2), both real parseable GGUF v3 files < 4 KB:
//   nvfp4_sidecar_trip.gguf   — one NVFP4 tensor + `.scale` + `.input_scale`
//       sidecar tensors: trips the D-B2 sidecar refusal END-TO-END through
//       Gemma4Runtime::load (the check fires right after read_metadata, before
//       any config parsing, so no full gemma4 metadata is needed).
//   nvfp4_nan_sentinel_trip.gguf — one NVFP4 tensor whose wire bytes carry a
//       raw 0x7F UE4M3 scale byte (D17/T5 NaN sentinel) at the correct data
//       offset. Full-file load CANNOT reach the WireQuant sentinel scan (config
//       parsing precedes WireQuant construction and fails first on missing
//       gemma4 keys) — the integration test drives the scan seam directly on
//       these bytes and documents that ordering (honest cells only).
//
// S2 quartet (§2.6, tripped by tests/invariant_matrix_binding.rs):
//   nvfp4_unknown_type_trip.gguf — one tensor with GGML type id 41 (no such
//       type at the pin): trips the PARSE-level fail-closed refusal
//       (src/gguf/reader.rs tensor_nbytes -> "unknown or removed GGML type"),
//       the file-boundary guard shared by every lane (I-unknown-type).
//   nvfp4_k_div_trip.gguf — one NVFP4 tensor with first dim 48 (48 % 64 != 0):
//       trips the PARSE-level divisibility refusal ("not divisible by block
//       size 64") — never a silent pad (I-k-div).
//   nvfp4_sidecar_admit_trip.gguf — the sidecar trio PLUS tokenizer.ggml.model
//       so RUNNABLE ADMISSION reaches the quant axis: trips the D-B2 sidecar
//       reject in runnable::admit end-to-end from file bytes (I-sidecar, L1).
//   nvfp4_pilot_admit.gguf — a BF16-free gemma4+NVFP4 pilot shape with
//       tokenizer.ggml.model: ADMITS on Windows (positive control for the
//       D-B3 carve-out at the file boundary) and trips the §9 platform gate's
//       named TK2 refusal on the ubuntu/macos CI legs (I-plat twin).
//
// Usage: node scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs
// Writes the six .gguf files into tests/fixtures/gguf/ plus
// tests/fixtures/gguf/SHA256SUMS (LF line endings — CRLF breaks verifiers).

import { mkdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

const OUT_DIR = join("tests", "fixtures", "gguf");
const ALIGNMENT = 32; // reader default (no general.alignment KV written)
const GGUF_VERSION = 3;
const T_F32 = 0;
const T_NVFP4 = 40; // pin GGML_TYPE_NVFP4 (D-B1)

// --- little-endian emit helpers over a growable byte array -----------------
class W {
  constructor() {
    this.chunks = [];
    this.len = 0;
  }
  push(buf) {
    this.chunks.push(buf);
    this.len += buf.length;
  }
  u32(v) {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(v);
    this.push(b);
  }
  i32(v) {
    const b = Buffer.alloc(4);
    b.writeInt32LE(v);
    this.push(b);
  }
  u64(v) {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(v));
    this.push(b);
  }
  str(s) {
    const bytes = Buffer.from(s, "utf8");
    this.u64(bytes.length);
    this.push(bytes);
  }
  bytes(arr) {
    this.push(Buffer.from(arr));
  }
  alignTo(a) {
    const pad = (a - (this.len % a)) % a;
    if (pad > 0) this.push(Buffer.alloc(pad));
  }
  out() {
    return Buffer.concat(this.chunks, this.len);
  }
}

// One deterministic 36-byte NVFP4 superblock: 4 UE4M3 sub-scales + 32 qs bytes.
// Safe scales avoid the 0x7F/0xFF sentinels; the NaN variant plants 0x7F at d[0].
function nvfp4Block({ nanSentinel }) {
  const d = nanSentinel ? [0x7f, 0x40, 0x51, 0x66] : [0x38, 0x40, 0x51, 0x66];
  const qs = Array.from({ length: 32 }, (_, j) => (j * 7 + 3) % 256);
  return [...d, ...qs];
}

const f32le = (v) => {
  const b = Buffer.alloc(4);
  b.writeFloatLE(v);
  return b;
};

// tensors: [{ name, dims, type, data: Buffer }]
// extraKvs: [{ key, value }] string KVs appended after general.architecture.
// The S1 fixtures pass none, which keeps their bytes frozen (metadata_count=1).
function buildGguf(tensors, extraKvs = []) {
  const w = new W();
  w.push(Buffer.from("GGUF", "ascii"));
  w.u32(GGUF_VERSION);
  w.u64(tensors.length); // tensor_count (i64)
  w.u64(1 + extraKvs.length); // metadata_count (i64)

  // First honest KV: this is a gemma4-lane fixture (the S1 refusals under test
  // do not depend on it — the sidecar check runs before config parsing). The
  // S2 admission fixtures add tokenizer.ggml.model so runnable::admit's fixed
  // axis order (architecture, tokenizer, THEN quants) reaches the quant axis.
  w.str("general.architecture");
  w.i32(8); // string
  w.str("gemma4");
  for (const kv of extraKvs) {
    w.str(kv.key);
    w.i32(8); // string
    w.str(kv.value);
  }

  // Tensor table with reader-contiguous relative offsets (each tensor's
  // rel_offset = align(prev_end, ALIGNMENT)).
  let rel = 0;
  const placed = [];
  for (const t of tensors) {
    w.str(t.name);
    w.u32(t.dims.length);
    for (const dim of t.dims) w.u64(dim);
    w.i32(t.type);
    w.u64(rel);
    placed.push({ ...t, rel });
    rel = Math.ceil((rel + t.data.length) / ALIGNMENT) * ALIGNMENT;
  }

  // Data section starts at align(header_end, ALIGNMENT). Each tensor's rel
  // offset is ALIGNMENT-aligned and data_start is too, so aligning the absolute
  // position before each push lands every tensor exactly at data_start + rel.
  w.alignTo(ALIGNMENT);
  for (const t of placed) {
    w.alignTo(ALIGNMENT);
    w.push(t.data);
  }
  return w.out();
}

const fixtures = [
  {
    file: "nvfp4_sidecar_trip.gguf",
    tensors: [
      {
        name: "blk.0.ffn_down.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
      // ModelOpt-convention sidecar pair — the D-B2 trip wires.
      { name: "blk.0.ffn_down.weight.scale", dims: [1], type: T_F32, data: f32le(1.0) },
      { name: "blk.0.ffn_down.weight.input_scale", dims: [1], type: T_F32, data: f32le(1.0) },
    ],
  },
  {
    file: "nvfp4_nan_sentinel_trip.gguf",
    tensors: [
      {
        name: "blk.0.ffn_up.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: true })), // d[0] = 0x7F
      },
    ],
  },
  // ---- S2 quartet (Amendment 3 §2.6, invariant-lane matrix) ----------------
  {
    file: "nvfp4_unknown_type_trip.gguf",
    tensors: [
      {
        // Type id 41 does not exist at the pin: read_metadata must refuse at
        // tensor_nbytes with the named unknown-type message. Data bytes are
        // present but never reached (the descriptor refuses first).
        name: "blk.0.mystery.weight",
        dims: [64],
        type: 41,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
    ],
  },
  {
    file: "nvfp4_k_div_trip.gguf",
    tensors: [
      {
        // First dim 48 is not divisible by the NVFP4 superblock size 64:
        // read_metadata must refuse ("not divisible by block size 64"),
        // never silently pad. Data bytes present but never reached.
        name: "blk.0.ffn_down.weight",
        dims: [48],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
    ],
  },
  {
    file: "nvfp4_sidecar_admit_trip.gguf",
    extraKvs: [{ key: "tokenizer.ggml.model", value: "gemma4" }],
    tensors: [
      {
        name: "blk.0.ffn_down.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
      // ModelOpt-convention sidecar pair — the D-B2 trip wires, this time
      // reachable through runnable::admit (tokenizer axis satisfied).
      { name: "blk.0.ffn_down.weight.scale", dims: [1], type: T_F32, data: f32le(1.0) },
      { name: "blk.0.ffn_down.weight.input_scale", dims: [1], type: T_F32, data: f32le(1.0) },
    ],
  },
  {
    file: "nvfp4_pilot_admit.gguf",
    extraKvs: [{ key: "tokenizer.ggml.model", value: "gemma4" }],
    tensors: [
      {
        // BF16-free pilot shape: admits on Windows (D-B3 carve-out, positive
        // control); trips the §9 platform gate off-Windows.
        name: "blk.0.ffn_down.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
    ],
  },
];

mkdirSync(OUT_DIR, { recursive: true });
const sums = [];
for (const f of fixtures) {
  const bytes = buildGguf(f.tensors, f.extraKvs ?? []);
  if (bytes.length >= 4096) throw new Error(`${f.file}: fixture must stay tiny (<4 KB)`);
  writeFileSync(join(OUT_DIR, f.file), bytes);
  const sha = createHash("sha256").update(bytes).digest("hex");
  sums.push(`${sha}  ${f.file}`);
  console.log(`${f.file}: ${bytes.length} B sha256=${sha}`);
}
writeFileSync(join(OUT_DIR, "SHA256SUMS"), sums.join("\n") + "\n"); // LF only
console.log("wrote", join(OUT_DIR, "SHA256SUMS"));
