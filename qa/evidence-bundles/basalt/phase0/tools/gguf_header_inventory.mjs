// BASALT Phase 0 — GGUF header-only tensor inventory parser.
// Provenance: header bytes fetched 2026-07-16 via
//   curl -sL -H 'Range: bytes=0-67108863' \
//     'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf' -o header.bin
// (first 64 MiB of the 8,192,951,456-byte file; LFS sha256
//  a2232a649523c36bf530f1dc3614eb8c800645c4227390381c8b05d4d6eee05a per the HF tree API
//  https://huggingface.co/api/models/unsloth/gemma-4-E4B-it-GGUF/tree/main).
// GGUF v3 layout per llama.cpp pin acd79d603 (build 9632); type ids per
// <llama.cpp>/ggml/include/ggml.h.
//
// Usage: node gguf_header_inventory.mjs <header.bin> <out.json> [meta.json]
//   meta.json optionally supplies { hf_repo, filename, file_size_bytes, lfs_sha256, fp_sibling }.

import { readFileSync, writeFileSync } from "node:fs";

const [, , headerPath, outPath, metaPath] = process.argv;
if (!headerPath || !outPath) {
  console.error("usage: node gguf_header_inventory.mjs <header.bin> <out.json> [meta.json]");
  process.exit(1);
}
const buf = readFileSync(headerPath);
const meta = metaPath ? JSON.parse(readFileSync(metaPath, "utf8")) : {};

let off = 0;
const u32 = () => { const v = buf.readUInt32LE(off); off += 4; return v; };
const i32 = () => { const v = buf.readInt32LE(off); off += 4; return v; };
const u64 = () => { const v = buf.readBigUInt64LE(off); off += 8; return v; };
const i64 = () => { const v = buf.readBigInt64LE(off); off += 8; return v; };
const f32 = () => { const v = buf.readFloatLE(off); off += 4; return v; };
const f64 = () => { const v = buf.readDoubleLE(off); off += 8; return v; };
const u8 = () => { const v = buf.readUInt8(off); off += 1; return v; };
const i8 = () => { const v = buf.readInt8(off); off += 1; return v; };
const u16 = () => { const v = buf.readUInt16LE(off); off += 2; return v; };
const i16 = () => { const v = buf.readInt16LE(off); off += 2; return v; };
const str = () => {
  const len = Number(u64());
  const s = buf.toString("utf8", off, off + len);
  off += len;
  return s;
};

// GGUF metadata value types
function readValue(type, skipArrayElems) {
  switch (type) {
    case 0: return u8();
    case 1: return i8();
    case 2: return u16();
    case 3: return i16();
    case 4: return u32();
    case 5: return i32();
    case 6: return f32();
    case 7: return u8() !== 0;
    case 8: return str();
    case 9: {
      const elemType = u32();
      const count = Number(u64());
      if (skipArrayElems) {
        // Still must walk past the elements.
        const out = { __array: true, elem_type: elemType, count };
        for (let i = 0; i < count; i++) readValue(elemType, true);
        return out;
      }
      const arr = [];
      for (let i = 0; i < count; i++) arr.push(readValue(elemType, false));
      return arr;
    }
    case 10: return u64().toString();
    case 11: return i64().toString();
    case 12: return f64();
    default: throw new Error(`unknown GGUF value type ${type} at offset ${off}`);
  }
}

// ggml type ids -> {name, blockElems, blockBytes} per ggml.h / ggml-common.h at pin acd79d603
const GGML_TYPES = {
  0: { name: "F32", blockElems: 1, blockBytes: 4 },
  1: { name: "F16", blockElems: 1, blockBytes: 2 },
  2: { name: "Q4_0", blockElems: 32, blockBytes: 18 },
  3: { name: "Q4_1", blockElems: 32, blockBytes: 20 },
  6: { name: "Q5_0", blockElems: 32, blockBytes: 22 },
  7: { name: "Q5_1", blockElems: 32, blockBytes: 24 },
  8: { name: "Q8_0", blockElems: 32, blockBytes: 34 },
  9: { name: "Q8_1", blockElems: 32, blockBytes: 36 },
  10: { name: "Q2_K", blockElems: 256, blockBytes: 84 },
  11: { name: "Q3_K", blockElems: 256, blockBytes: 110 },
  12: { name: "Q4_K", blockElems: 256, blockBytes: 144 },
  13: { name: "Q5_K", blockElems: 256, blockBytes: 176 },
  14: { name: "Q6_K", blockElems: 256, blockBytes: 210 },
  15: { name: "Q8_K", blockElems: 256, blockBytes: 292 },
  16: { name: "IQ2_XXS", blockElems: 256, blockBytes: 66 },
  17: { name: "IQ2_XS", blockElems: 256, blockBytes: 74 },
  18: { name: "IQ3_XXS", blockElems: 256, blockBytes: 98 },
  19: { name: "IQ1_S", blockElems: 256, blockBytes: 50 },
  20: { name: "IQ4_NL", blockElems: 32, blockBytes: 18 },
  21: { name: "IQ3_S", blockElems: 256, blockBytes: 110 },
  22: { name: "IQ2_S", blockElems: 256, blockBytes: 82 },
  23: { name: "IQ4_XS", blockElems: 256, blockBytes: 136 },
  24: { name: "I8", blockElems: 1, blockBytes: 1 },
  25: { name: "I16", blockElems: 1, blockBytes: 2 },
  26: { name: "I32", blockElems: 1, blockBytes: 4 },
  27: { name: "I64", blockElems: 1, blockBytes: 8 },
  28: { name: "F64", blockElems: 1, blockBytes: 8 },
  30: { name: "BF16", blockElems: 1, blockBytes: 2 },
  40: { name: "NVFP4", blockElems: 64, blockBytes: 36 }, // QK_NVFP4=64, block_nvfp4=36B (ggml-common.h:211-217)
};

// --- header ---
const magic = u32();
if (magic !== 0x46554747) throw new Error(`bad magic 0x${magic.toString(16)}`);
const version = u32();
const tensorCount = Number(u64());
const kvCount = Number(u64());

// --- KV pairs ---
const interestingKvs = {};
const skippedKvKeys = [];
for (let i = 0; i < kvCount; i++) {
  const key = str();
  const vtype = u32();
  const isTokenizerArray = vtype === 9 && key.startsWith("tokenizer.");
  const val = readValue(vtype, isTokenizerArray);
  if (isTokenizerArray) {
    skippedKvKeys.push(`${key} (array elem_type=${val.elem_type}, count=${val.count})`);
  } else if (key.startsWith("general.") || !key.startsWith("tokenizer.")) {
    // keep general.* and <arch>.* ; keep scalar tokenizer values too? task says skip tokenizer arrays only,
    // but "interesting_kvs" = general.* and <arch>.* — record non-tokenizer keys.
    if (key.startsWith("tokenizer.")) continue;
    interestingKvs[key] = Array.isArray(val) && val.length > 64 ? { __array: true, count: val.length, head: val.slice(0, 8) } : val;
  }
}

// --- tensor infos ---
const tensors = [];
const typeCounts = {};
let sumBytes = 0n;
let sumQuantBytes = 0n; // non-F32 tensors
const unknownTypes = new Set();
for (let i = 0; i < tensorCount; i++) {
  const name = str();
  const nDims = u32();
  const dims = [];
  for (let d = 0; d < nDims; d++) dims.push(Number(u64()));
  const typeId = u32();
  const offset = u64();
  const t = GGML_TYPES[typeId];
  const typeName = t ? t.name : `UNKNOWN_${typeId}`;
  if (!t) unknownTypes.add(typeId);
  const nElems = dims.reduce((a, b) => a * b, 1);
  let bytes = null;
  if (t) {
    if (nElems % t.blockElems !== 0) throw new Error(`tensor ${name}: ${nElems} elems not divisible by block ${t.blockElems}`);
    bytes = BigInt(nElems / t.blockElems) * BigInt(t.blockBytes);
    sumBytes += bytes;
    if (typeName !== "F32") sumQuantBytes += bytes;
  }
  typeCounts[typeName] = (typeCounts[typeName] || 0) + 1;
  const k = dims[0];
  tensors.push({
    name,
    dims,
    type_id: typeId,
    type_name: typeName,
    k,
    k_mod_16: k % 16,
    k_mod_64: k % 64,
    bytes: bytes === null ? null : Number(bytes),
    offset: offset.toString(),
  });
}

const headerEnd = off; // byte offset where tensor data region metadata ends (pre-alignment)

const out = {
  source: {
    hf_repo: meta.hf_repo ?? null,
    filename: meta.filename ?? null,
    file_size_bytes: meta.file_size_bytes ?? null,
    lfs_sha256: meta.lfs_sha256 ?? null,
    header_fetch: {
      url: meta.hf_repo && meta.filename ? `https://huggingface.co/${meta.hf_repo}/resolve/main/${meta.filename}` : null,
      range: "bytes=0-67108863",
      fetched_utc: new Date().toISOString(),
    },
  },
  gguf_version: version,
  tensor_count: tensorCount,
  metadata_kv_count: kvCount,
  arch: interestingKvs["general.architecture"] ?? null,
  interesting_kvs: interestingKvs,
  skipped_tokenizer_arrays: skippedKvKeys,
  fp_sibling: meta.fp_sibling ?? null,
  sanity: {
    header_bytes_parsed: headerEnd,
    sum_tensor_bytes: Number(sumBytes),
    sum_quantized_tensor_bytes_nonF32: Number(sumQuantBytes),
    sum_tensor_bytes_plus_header: Number(sumBytes) + headerEnd,
    file_size_bytes: meta.file_size_bytes ?? null,
    residual_vs_file_size:
      meta.file_size_bytes != null ? meta.file_size_bytes - (Number(sumBytes) + headerEnd) : null,
    by_type_counts: typeCounts,
    types_outside_f32_f16_bf16_q8_0: tensors
      .filter((t) => !["F32", "F16", "BF16", "Q8_0"].includes(t.type_name))
      .map((t) => `${t.name}:${t.type_name}`),
    k_mod_64_violations: tensors.filter((t) => t.k_mod_64 !== 0).map((t) => `${t.name} (K=${t.k})`),
    k_mod_16_violations: tensors.filter((t) => t.k_mod_16 !== 0).map((t) => `${t.name} (K=${t.k})`),
    has_output_weight: tensors.some((t) => t.name === "output.weight"),
    lm_head_tied_inferred: !tensors.some((t) => t.name === "output.weight"),
    unknown_type_ids: [...unknownTypes],
  },
  tensors,
};

writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log(
  `gguf v${version}, arch=${out.arch}, tensors=${tensorCount}, kvs=${kvCount}, header=${headerEnd}B, ` +
    `sum_tensor_bytes=${sumBytes}, types=${JSON.stringify(typeCounts)}`
);
