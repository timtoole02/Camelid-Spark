// BASALT Phase 1 — fixture internal-consistency cross-check.
// Verifies (independently of the C harness, in JS f32 arithmetic):
//   1. decode_table[scale][code] === fround(kvalues[code] * ue4m3_table[scale]) bit-exactly
//      (exact: |kvalue| <= 12 fits 4 bits, scale has 24-bit mantissa -> product exact in
//       f64, single rounding to f32 == C float multiply)
//   2. nibble probes: expected == packing-rule + decode-table reconstruction
//   3. every random_blocks / encode_vectors / real_blocks expected output ==
//      decode-table reconstruction from the wire bytes
//   4. rt-* encode vectors round-trip (dequant bits == input bits) — informational
// Usage: node crosscheck.mjs <fixtures-dir>

import { readFileSync } from "node:fs";

const DIR = process.argv[2] ?? "out1";
const KV = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

const hex2f = (h) => {
  const b = new ArrayBuffer(4);
  new DataView(b).setUint32(0, parseInt(h, 16));
  return new DataView(b).getFloat32(0);
};
const f2hex = (f) => {
  const b = new ArrayBuffer(4);
  new DataView(b).setFloat32(0, Math.fround(f));
  return new DataView(b).getUint32(0).toString(16).padStart(8, "0");
};

const ue = JSON.parse(readFileSync(`${DIR}/ue4m3_table.json`, "utf8"));
const dt = JSON.parse(readFileSync(`${DIR}/decode_table.json`, "utf8"));
let fails = [];
const check = (cond, msg) => { if (!cond) fails.push(msg); };

// --- 1. decode table vs closed form ---
check(ue.table.length === 256, "ue4m3 table length");
check(ue.table[0x00] === "00000000", "ue4m3[0x00] must be +0.0");
check(ue.table[0x7f] === "00000000", "ue4m3[0x7f] must be +0.0");
let n1 = 0;
for (let s = 0; s < 256; s++) {
  const d = hex2f(ue.table[s]);
  for (let c = 0; c < 16; c++) {
    const want = f2hex(KV[c] * d);
    if (dt.entries[s][c] !== want) fails.push(`decode_table[${s}][${c}]=${dt.entries[s][c]} != ${want}`);
    else n1++;
  }
}
console.log(`1. decode_table closed-form: ${n1}/4096 bit-exact`);

// --- helper: reconstruct 64 outputs from 36 wire bytes via decode table ---
function reconstruct(wire) {
  const out = new Array(64);
  for (let s = 0; s < 4; s++) {
    const scale = wire[s];
    for (let j = 0; j < 8; j++) {
      const byte = wire[4 + s * 8 + j];
      out[s * 16 + j]     = dt.entries[scale][byte & 0x0f];
      out[s * 16 + 8 + j] = dt.entries[scale][byte >> 4];
    }
  }
  return out.join("");
}

// --- 2. nibble probes ---
for (const p of dt.nibble_probes) {
  const wire = Buffer.from(p.wire, "base64");
  check(wire.length === 36, `${p.name} wire length`);
  check(reconstruct(wire) === p.expected, `${p.name} packing mismatch`);
}
console.log(`2. nibble probes: ${dt.nibble_probes.length} checked`);

// --- 3. wire->expected reconstruction across all block fixtures ---
function checkBlocks(file, list, wKey, eKey) {
  let n = 0;
  for (const b of list) {
    const wire = Buffer.from(b[wKey], "base64");
    if (wire.length !== 36) { fails.push(`${file}: bad wire length`); continue; }
    const rec = reconstruct(wire);
    if (rec !== b[eKey]) fails.push(`${file} block tag/idx=${b.tag ?? `${b.t}:${b.b}`}: expected mismatch`);
    else n++;
  }
  console.log(`3. ${file}: ${n}/${list.length} blocks reconstruct bit-exact`);
}
const rb = JSON.parse(readFileSync(`${DIR}/random_blocks.json`, "utf8"));
checkBlocks("random_blocks.json", rb.blocks, "w", "e");
const ev = JSON.parse(readFileSync(`${DIR}/encode_vectors.json`, "utf8"));
checkBlocks("encode_vectors.json", ev.vectors, "wire", "dequant");
const rl = JSON.parse(readFileSync(`${DIR}/real_blocks.json`, "utf8"));
checkBlocks("real_blocks.json", rl.blocks, "w", "e");

// --- counts ---
check(rb.blocks.length >= 10000, `random_blocks count ${rb.blocks.length} < 10000`);
check(rl.blocks.length >= 2000, `real_blocks count ${rl.blocks.length} < 2000`);
const tset = new Set(rl.blocks.map((b) => b.t));
check(tset.size >= 3, `real_blocks tensors ${tset.size} < 3`);
check(rl.tensors.some((t) => t.name === "token_embd.weight"), "token_embd.weight missing");

// --- 4. round-trip info on rt-* vectors ---
let rtOk = 0, rtTot = 0;
for (const v of ev.vectors.filter((v) => v.tag.startsWith("rt-"))) {
  rtTot++;
  if (v.input.join("") === v.dequant) rtOk++;
  else fails.push(`INFO-ONLY? rt vector ${v.tag} not exact round-trip`);
}
console.log(`4. rt-* round-trips exact: ${rtOk}/${rtTot}`);

// --- tag inventory ---
const tags = {};
for (const b of rb.blocks) tags[b.tag] = (tags[b.tag] || 0) + 1;
console.log("random_blocks tags:", JSON.stringify(tags));

if (fails.length) {
  console.error(`CROSS-CHECK FAIL (${fails.length}):`);
  for (const f of fails.slice(0, 20)) console.error("  " + f);
  process.exit(1);
}
console.log("CROSS-CHECK PASS");
