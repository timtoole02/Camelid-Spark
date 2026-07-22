// BASALT Phase 1 — real_blocks extraction + assembly.
// Pin: llama.cpp acd79d603. GGUF v3 header-walk logic adapted from
// <home>/cam-basalt/qa/evidence-bundles/basalt/phase0/tools/gguf_header_inventory.mjs
//
// Two-phase usage (the numeric truth comes from the pin's compiled C code, not JS):
//   node extract_real_blocks.mjs extract  -> verifies model sha256, walks header,
//        samples NVFP4 (type 40) blocks from >=3 tensors incl. token_embd.weight,
//        writes real_blocks.bin (N*36 raw wire bytes) + real_blocks_meta.json
//   (then: nvfp4_fixture_gen.exe dequant real_blocks.bin <N> real_blocks_expected.txt)
//   node extract_real_blocks.mjs assemble -> merges meta + wire + pin-produced
//        expected values into real_blocks.json
//
// Sampling is deterministic: for tensor with B total blocks and n samples,
// sample j (0-based) takes block index floor(j*B/n).

import { readFileSync, writeFileSync, openSync, readSync, closeSync, statSync } from "node:fs";
import { createHash } from "node:crypto";

const MODEL = "<camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf";
const EXPECTED_SHA = "7337b616141b2436f839b353fb40dc2f77023989316ea7d83624f4f45e2a9146";
const DIR = "<scratchpad>/basalt-p1";
const HEADER_BYTES = 16 * 1024 * 1024; // plenty for a 0.6B header
const TOTAL_SAMPLES = 2048;

const mode = process.argv[2];
if (mode !== "extract" && mode !== "assemble") {
  console.error("usage: node extract_real_blocks.mjs extract|assemble");
  process.exit(1);
}

function walkHeader() {
  const fd = openSync(MODEL, "r");
  const buf = Buffer.alloc(HEADER_BYTES);
  readSync(fd, buf, 0, HEADER_BYTES, 0);
  closeSync(fd);

  let off = 0;
  const u32 = () => { const v = buf.readUInt32LE(off); off += 4; return v; };
  const u64 = () => { const v = buf.readBigUInt64LE(off); off += 8; return v; };
  const str = () => { const len = Number(u64()); const s = buf.toString("utf8", off, off + len); off += len; return s; };
  const SCALAR_SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  function skipValue(type) {
    if (type === 8) { str(); return null; }
    if (type === 9) {
      const elemType = u32(); const count = Number(u64());
      for (let i = 0; i < count; i++) skipValue(elemType);
      return null;
    }
    const sz = SCALAR_SIZE[type];
    if (sz === undefined) throw new Error(`unknown kv type ${type} at ${off}`);
    let v = null;
    if (type === 4) v = buf.readUInt32LE(off);
    if (type === 5) v = buf.readInt32LE(off);
    off += sz;
    return v;
  }

  const magic = u32();
  if (magic !== 0x46554747) throw new Error(`bad magic 0x${magic.toString(16)}`);
  const version = u32();
  const tensorCount = Number(u64());
  const kvCount = Number(u64());

  let alignment = 32; // GGUF default
  for (let i = 0; i < kvCount; i++) {
    const key = str();
    const vtype = u32();
    const v = skipValue(vtype);
    if (key === "general.alignment" && v !== null) alignment = Number(v);
  }

  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = str();
    const nDims = u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(Number(u64()));
    const typeId = u32();
    const offset = Number(u64()); // relative to aligned data-section start
    tensors.push({ name, dims, typeId, offset });
  }
  const headerEnd = off;
  const dataStart = Math.ceil(headerEnd / alignment) * alignment;
  return { version, tensorCount, kvCount, alignment, headerEnd, dataStart, tensors };
}

if (mode === "extract") {
  process.stdout.write("sha256 of model... ");
  const h = createHash("sha256");
  {
    const fd = openSync(MODEL, "r");
    const sz = statSync(MODEL).size;
    const chunk = Buffer.alloc(8 * 1024 * 1024);
    let pos = 0;
    while (pos < sz) {
      const n = readSync(fd, chunk, 0, Math.min(chunk.length, sz - pos), pos);
      h.update(chunk.subarray(0, n));
      pos += n;
    }
    closeSync(fd);
  }
  const sha = h.digest("hex");
  console.log(sha);
  if (sha !== EXPECTED_SHA) { console.error("SHA256 MISMATCH — aborting"); process.exit(1); }

  const hdr = walkHeader();
  const nv = hdr.tensors.filter(t => t.typeId === 40);
  console.log(`gguf v${hdr.version}, tensors=${hdr.tensorCount}, alignment=${hdr.alignment}, headerEnd=${hdr.headerEnd}, dataStart=${hdr.dataStart}, nvfp4_tensors=${nv.length}`);
  if (!nv.some(t => t.name === "token_embd.weight")) { console.error("token_embd.weight is not NVFP4 — aborting"); process.exit(1); }

  // Deterministic tensor choice: token_embd.weight + 5 spread picks from the
  // remaining NVFP4 tensors in header order (first, 1/4, 1/2, 3/4, last).
  const rest = nv.filter(t => t.name !== "token_embd.weight");
  // +1/+2/+3 offsets break the layer-stride aliasing (7 NVFP4 tensors per layer)
  // so the picks cover different tensor roles, not the same role across layers.
  const pickIdx = [...new Set([0, Math.floor(rest.length / 4) + 1, Math.floor(rest.length / 2) + 2, Math.floor(3 * rest.length / 4) + 3, rest.length - 1])];
  const chosen = [nv.find(t => t.name === "token_embd.weight"), ...pickIdx.map(i => rest[i])];
  const per = Math.floor(TOTAL_SAMPLES / chosen.length);
  const counts = chosen.map((_, i) => (i === 0 ? TOTAL_SAMPLES - per * (chosen.length - 1) : per));

  const fd = openSync(MODEL, "r");
  const samples = [];
  const wire = [];
  for (let ci = 0; ci < chosen.length; ci++) {
    const t = chosen[ci];
    const elems = t.dims.reduce((a, b) => a * b, 1);
    if (elems % 64 !== 0) throw new Error(`${t.name}: elems not divisible by 64`);
    const totalBlocks = elems / 64;
    const base = hdr.dataStart + t.offset;
    for (let j = 0; j < counts[ci]; j++) {
      const bi = Math.floor(j * totalBlocks / counts[ci]);
      const b = Buffer.alloc(36);
      readSync(fd, b, 0, 36, base + bi * 36);
      samples.push({ t: ci, b: bi });
      wire.push(b);
    }
  }
  closeSync(fd);

  writeFileSync(`${DIR}/real_blocks.bin`, Buffer.concat(wire));
  writeFileSync(`${DIR}/real_blocks_meta.json`, JSON.stringify({
    model_path: MODEL,
    model_sha256: sha,
    gguf_version: hdr.version,
    alignment: hdr.alignment,
    header_end: hdr.headerEnd,
    data_start: hdr.dataStart,
    tensors: chosen.map((t, i) => ({ index: i, name: t.name, dims: t.dims, type_id: t.typeId, rel_offset: t.offset, n_samples: counts[i], total_blocks: t.dims.reduce((a, b) => a * b, 1) / 64 })),
    samples,
  }, null, 1));
  console.log(`wrote ${samples.length} blocks from ${chosen.length} tensors: ${chosen.map((t, i) => `${t.name}(${counts[i]})`).join(", ")}`);
}

if (mode === "assemble") {
  const meta = JSON.parse(readFileSync(`${DIR}/real_blocks_meta.json`, "utf8"));
  const bin = readFileSync(`${DIR}/real_blocks.bin`);
  const expected = readFileSync(`${DIR}/real_blocks_expected.txt`, "utf8").trim().split("\n");
  if (expected.length !== meta.samples.length) throw new Error(`expected ${meta.samples.length} lines, got ${expected.length}`);

  // provenance mirrors the harness's (same pin, same route); generator lists both tools.
  const prov = {
    pin_sha: "acd79d603",
    generator: "extract_real_blocks.mjs + nvfp4_fixture_gen.c",
    route: "linked-libs",
    route_detail: "wire bytes read verbatim from the GGUF file; every expected value produced by dequantize_row_nvfp4 in the pin-built ggml-base.dll via nvfp4_fixture_gen.exe dequant mode; this JS only slices bytes and formats JSON",
    compiler: "MSVC cl 19.44.35228 x64 (_MSC_FULL_VER=194435228)",
    date: new Date().toISOString().slice(0, 10),
    prng: "none (blocks sampled deterministically: block index = floor(j*total_blocks/n_samples))",
    seed: 0,
  };
  const blocks = meta.samples.map((s, i) => ({
    t: s.t,
    b: s.b,
    w: bin.subarray(i * 36, i * 36 + 36).toString("base64"),
    e: expected[i],
  }));
  const out = {
    provenance: prov,
    desc: "blocks read straight out of the NVFP4 GGUF: t = index into tensors[], b = block index within that tensor (block = 64 elements, 36 wire bytes: d[4] UE4M3 scales then qs[32]); w = wire bytes base64; e = 64 expected f32 from the pin's dequantize_row_nvfp4 (concatenated lowercase %08x hex of IEEE-754 u32 bits, element order 0..63)",
    model: {
      path: meta.model_path,
      sha256: meta.model_sha256,
      gguf_version: meta.gguf_version,
      alignment: meta.alignment,
      header_end: meta.header_end,
      data_start: meta.data_start,
    },
    tensors: meta.tensors,
    blocks,
  };
  // compact but line-per-block for diffability
  const json = "{\n" +
    `"provenance": ${JSON.stringify(prov, null, 1)},\n` +
    `"desc": ${JSON.stringify(out.desc)},\n` +
    `"model": ${JSON.stringify(out.model, null, 1)},\n` +
    `"tensors": ${JSON.stringify(out.tensors, null, 1)},\n` +
    `"blocks": [\n` +
    blocks.map(b => JSON.stringify(b)).join(",\n") +
    "\n]\n}\n";
  writeFileSync(`${DIR}/out1/real_blocks.json`, json);
  console.log(`real_blocks.json: ${blocks.length} blocks, ${json.length} bytes`);
}
