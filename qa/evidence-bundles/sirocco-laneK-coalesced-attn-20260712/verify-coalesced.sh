#!/usr/bin/env bash
# Verify the coalesced-default-ON build (branch sirocco-laneK-coalesced).
# Self-contained: compares the new binary's DEFAULT (coalesced on) vs its =0 kill switch (off).
set -u
cd ~/Camelid
SC="~/AppData/Local/Temp/claude/C--Users-timto/9729b42a-cbdc-4f3e-a6e4-7e25a1557fb2/scratchpad"
M=models/Llama-3.2-1B-Instruct-Q8_0.gguf
toks() { CAMELID_LOG=error env "$@" ./target/release/camelid.exe bench-generate "$M" --prompt-file "$1x" --max-tokens 64 --temperature 0 2>/dev/null; }

echo "### 1. GPU-resident load + parity gate (must NOT fall to CPU) ###"
CAMELID_LOG=info ./target/release/camelid.exe bench-generate "$M" --prompt "Hi" --max-tokens 4 --temperature 0 2>&1 | grep -iE "resident|parity|FAILED|CPU|gpu-runnable" | head

echo "### 2. ctx~0 NO-REGRESSION (default vs =0; must be ~equal, split-K inactive) ###"
run0() { CAMELID_LOG=error env "$@" ./target/release/camelid.exe bench-generate "$M" --prompt "Hi" --max-tokens 256 --temperature 0 --warmup 2>/dev/null | node -e 'const j=JSON.parse(require("fs").readFileSync(0,"utf8"));process.stdout.write(j.tokens_per_second.toFixed(2))'; }
echo "ctx~0 default(on)=$(run0)  off(=0)=$(run0 CAMELID_ATTN_COALESCED=0)"; sleep 6

echo "### 3. long-ctx SPEEDUP (default on vs =0), varied prompt ###"
runL() { CAMELID_LOG=error env "$@" ./target/release/camelid.exe bench-generate "$M" --prompt-file "$SC/varied2k.txt" --max-tokens 64 --temperature 0 --warmup 2>/dev/null | node -e 'const j=JSON.parse(require("fs").readFileSync(0,"utf8"));process.stdout.write(j.tokens_per_second.toFixed(2)+"@"+j.prompt_tokens)'; }
for r in 1 2; do echo "r$r on=$(runL)  off=$(runL CAMELID_ATTN_COALESCED=0)"; sleep 6; done

echo "### 4. TOKEN-IDENTITY (default on vs off) long-ctx ###"
CAMELID_LOG=error ./target/release/camelid.exe bench-generate "$M" --prompt-file "$SC/varied2k.txt" --max-tokens 64 --temperature 0 2>/dev/null | node -e 'const j=JSON.parse(require("fs").readFileSync(0,"utf8"));process.stdout.write(JSON.stringify(j.output_token_ids))' > "$SC/vc-on.txt"
CAMELID_ATTN_COALESCED=0 CAMELID_LOG=error ./target/release/camelid.exe bench-generate "$M" --prompt-file "$SC/varied2k.txt" --max-tokens 64 --temperature 0 2>/dev/null | node -e 'const j=JSON.parse(require("fs").readFileSync(0,"utf8"));process.stdout.write(JSON.stringify(j.output_token_ids))' > "$SC/vc-off.txt"
diff -q "$SC/vc-on.txt" "$SC/vc-off.txt" && echo "TOKEN-IDENTICAL (default-on vs kill-switch)" || echo "DIVERGED"
