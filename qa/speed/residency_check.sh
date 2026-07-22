#!/usr/bin/env bash
# Q4_K 8B-on-6GB residency + decode validation. Loads Qwen3-8B-Q4_K_M on the
# CUDA resident path (which now admits Q4_K) and confirms ALL layers fit VRAM-
# resident (no host offload) + measures decode tok/s. Run when the model is
# present and the Q4_K kernel build is done. ONE camelid at a time (kill strays).
set -u
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-../models/Qwen3-8B-Q4_K_M.gguf}"  # override with MODEL=/path/to.gguf
PROMPT="${1:-Explain, step by step, how a transformer decoder generates text one token at a time, and why decode is memory-bandwidth bound.}"
if [ ! -f "$MODEL" ]; then echo "model not present: $MODEL"; exit 1; fi
echo "=== 8B-Q4_K residency + decode (CUDA resident path) ==="
CAMELID_CUDA_RESIDENT_DECODE=1 ./target/release/camelid.exe bench-generate "$MODEL" \
  --prompt "$PROMPT" --max-tokens 64 --temperature 0 --iterations 3 --warmup 2>res.err > res.json
node -e '
const fs=require("fs");
const rows=fs.readFileSync("res.json","utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
if(!rows.length){console.log("no output — see res.err");process.exit(1);}
const tps=rows.map(r=>r.tokens_per_second).sort((a,b)=>a-b);
const med=tps[Math.floor(tps.length/2)];
const o=rows[0].offload||{};
const peakGB=(rows[0].peak_memory_bytes/1e9).toFixed(2);
const bytesTok=4.7e9, bw=273e9;            // ~weight stream/token; STREAM bw
console.log(`layers_resident=${o.layers_resident} layers_offloaded=${o.layers_offloaded} free_vram=${o.free_vram_bytes?(o.free_vram_bytes/1e9).toFixed(2)+"GB":"?"} peak=${peakGB}GB`);
console.log(`decode median ${med.toFixed(2)} t/s  (~${(med*bytesTok/bw*100).toFixed(0)}% of ~${(bw/bytesTok).toFixed(0)} t/s Q4_K roofline)`);
console.log(o.layers_offloaded===0 ? "RESIDENCY: PASS — 8B fully VRAM-resident on the 6GB card (zero host offload)" : `RESIDENCY: PARTIAL — ${o.layers_offloaded}/${(o.layers_resident||0)+(o.layers_offloaded||0)} layers offloaded`);
'
echo "(compare: 8B-Q8_0 would NOT fit — needs ~16/36 resident + 20 host-streamed)"
