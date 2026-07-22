#!/usr/bin/env bash
# spec-verify-parity.sh — Apple-Silicon twin of spec-verify-parity.ps1.
#
# SPEC-VERIFY LOSSLESSNESS GATE: greedy speculative decode must be token-identical
# to this build's own plain greedy decode, on every spec-friendly workload column.
# The denominator is always Camelid's own plain greedy stream — no llama.cpp
# comparison.
#
# POST-PHASE-3 REALITY  (see docs/perf-deep-dive/WIN2METAL_PHASE3_PLAN.md §5,
#                        docs/perf-deep-dive/WIN2METAL_RECON.md §A1/§A2)
#   The bit-exact Metal batched speculative verify (`verify_batch`, C0–C3) is now
#   wired into `serve` (C4): with CAMELID_SPEC_DECODE=ngram + CAMELID_SPEC_GPU=1 the
#   macOS resident-decode engine stays ON under speculation and the GPU verify fires
#   live, accepting the longest-confirmed draft prefix per round. Every emitted token
#   is the target's own greedy argmax given its accepted prefix, so the speculative
#   stream is byte-identical to the non-speculative greedy baseline.
#
#   This is the lane that was previously unreachable from the harness:
#     - §A2 reachability — `bench-speculative` runs the CPU forward path on Mac (the
#       resident engine is not engaged by that CLI), so it could never exercise the
#       Metal verify. The Metal verify only fires under `serve` with the resident
#       stack + CAMELID_SPEC_GPU=1, which is exactly what the Metal lane below drives.
#     - §A1 divergence — that CPU chunk verify (forward_greedy_verify_chunk) is NOT
#       byte-identical to single-token decode on this M4: whenever a CPU verify round
#       fires the stream diverges from plain greedy. That is a pre-existing CPU-path
#       defect, OUT OF Phase 3 scope, and is NOT the thing this gate measures.
#
# LANES
#   metal-resident (THE PHASE 3 GATE — governs the exit code):
#       Drives `camelid serve` on :18100 twice on the same model:
#         spec     = CAMELID_SPEC_DECODE=ngram CAMELID_SPEC_GPU=1 + resident stack on
#                    (default greedy request → speculation engages → verify_batch fires)
#         baseline = resident stack on, speculation OFF
#                    (default greedy request → plain single-token greedy decode)
#       Both requests are default greedy (NO temperature/seed/top_p/top_k — passing any
#       of those silently disables speculation, see api/mod.rs spec branch gate).
#       For each spec-friendly column it asserts the speculative generated-token stream
#       is BYTE-IDENTICAL (SHA-256 over the token ids AND over the decoded text) to the
#       baseline, AND that the Metal verify actually fired (CAMELID_SPEC_VERIFY_TRACE
#       `[metal-spec-verify]` rounds > 0 — a column that silently passed via the CPU
#       fallback is a FAILURE, not a pass). LOSSLESS + fired on every gate column → 0.
#       Any divergence, or a gate column that never exercised verify_batch → non-zero.
#
#   metal-tree (THE PHASE 4 GATE — also governs the exit code):
#       REACHABILITY (honest): the TREE verify (verify_tree_gpu → verify_tree_metal →
#       verify_batch_tree) is NOT reachable from `serve`. The server speculative loop
#       (api/mod.rs) only ever calls the LINEAR verify_drafts_gpu; CAMELID_SPEC_TREE is
#       consulted ONLY by `bench-speculative` (main.rs generate_run_speculative), which
#       re-enables the resident paths so the Metal resident engine engages. So this lane
#       drives `bench-speculative` with CAMELID_SPEC_TREE=1 + the SAME resident fast stack
#       the serve lane uses (CAMELID_SPEC_TREE_GATE=0 so a full tree is drawn every round and
#       the GPU verify is forced to fire — losslessness is the verify's job either way).
#       bench-speculative computes its OWN lossless verdict (the speculative token stream vs
#       THIS build's plain greedy decode, byte-identical) and reports gpu/cpu verify rounds.
#       GATE: every tree column must be LOSSLESS *and* the GPU tree verify must have fired
#       (gpu_verify_rounds>0) — a silent CPU fallback or a no-spec-round pass is a FAILURE.
#       The receipt records the max tree fan-out observed (fan-out>1 ⇒ genuine multi-branch
#       tree verify + branching KV compaction exercised, not just the single-branch anchor).
#
#   bench-speculative reference (INFORMATIONAL ONLY — does NOT gate):
#       The original bench-speculative linear/tree lanes, kept verbatim. recon §A1/§A2
#       described these (at base 28f224b) as the CPU chunk-verify fallback that DIVERGES
#       the moment a verify round fires. HONEST POST-PHASE-3 UPDATE: that is no longer
#       what they do on this build. The C4 reroute of `verify_drafts_gpu` to the Metal
#       seam (src/inference.rs) is reached by `bench-speculative` too — `generate_run_
#       speculative` re-enables the resident paths so the verify engages — so on this
#       Phase 3 binary the linear lane runs through the SAME bit-exact Metal verify
#       (gpu_verify_rounds>0, cpu_verify_rounds=0) and reports LOSSLESS; the tree lane
#       fires no spec rounds (trivially lossless, as §A1 already noted). The §A1 CPU-
#       chunk divergence now reproduces ONLY when the GPU verify path is unavailable
#       (the pre-Phase-3 base, or a forced-CPU config). The per-run verify path is
#       derived live from the round counters, not asserted. Still NON-gating either way.
#       Set SKIP_CPU_BENCH=1 to skip; CPU_BENCH_TIMEOUT (default 180s) bounds each run.
#
# A `camelid.spec-verify/v1` receipt is written to qa/speed/receipts/ per run.
#
# Usage:
#   qa/speed/spec-verify-parity.sh
#   BIN=/path/to/camelid MODEL=/path/to.gguf qa/speed/spec-verify-parity.sh
#   SKIP_CPU_BENCH=1 qa/speed/spec-verify-parity.sh      # Metal gate only
#
# Exit code is governed by the Metal resident (linear) lane AND the Phase 4 tree lane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- pins (all overridable by env) -----------------------------------------
BIN="${BIN:-/Volumes/Untitled/camelid-target/release/camelid}"
MODEL="${MODEL:-/Volumes/Untitled/models/Llama-3.2-1B-Instruct-Q8_0.gguf}"
PROMPTS_JSON="${PROMPTS_JSON:-$SCRIPT_DIR/prompts.json}"
RECEIPTS_DIR="${RECEIPTS_DIR:-$SCRIPT_DIR/receipts}"
DRAFTER="${DRAFTER:-ngram}"          # greedy n-gram drafter (no draft model needed)
PORT="${PORT:-18100}"
ADDR="127.0.0.1:${PORT}"
SKIP_CPU_BENCH="${SKIP_CPU_BENCH:-0}"
CPU_BENCH_TIMEOUT="${CPU_BENCH_TIMEOUT:-180}"

# Columns that drive the Metal verify gate: spec-friendly (the n-gram drafter finds
# matches so verify rounds fire) + the >512-context longctx column (forces the Metal
# split-K verify path). These are the columns where verify_batch is expected to fire.
GATE_COLUMNS="${GATE_COLUMNS:-code_completion structured_json repetitive_extraction longctx_splitk}"

# Columns that drive the Phase 4 TREE verify gate. These are the spec-friendly columns where
# the suffix drafter finds recurrence and proposes branching (fan-out>1) trees, so the GPU tree
# verify (verify_tree_gpu→verify_tree_metal→verify_batch_tree) fires every round under the
# ungated full-tree policy. Each must be LOSSLESS with gpu_verify_rounds>0 (see header LANES).
TREE_COLUMNS="${TREE_COLUMNS:-repetitive_extraction code_completion structured_json}"
TREE_TIMEOUT="${TREE_TIMEOUT:-240}"   # hard per-column ceiling for the tree bench-speculative run

# --- preflight --------------------------------------------------------------
[ -x "$BIN" ]           || { echo "[spec-verify-parity] bin not found/executable: $BIN" >&2; exit 2; }
[ -f "$MODEL" ]         || { echo "[spec-verify-parity] model not found: $MODEL" >&2; exit 2; }
[ -f "$PROMPTS_JSON" ]  || { echo "[spec-verify-parity] prompts not found: $PROMPTS_JSON" >&2; exit 2; }
command -v node >/dev/null || { echo "[spec-verify-parity] node is required (JSON parsing)" >&2; exit 2; }
command -v curl >/dev/null || { echo "[spec-verify-parity] curl is required (HTTP API)" >&2; exit 2; }
mkdir -p "$RECEIPTS_DIR"

BIN_VERSION="$("$BIN" --version 2>/dev/null | head -n1 | tr -d '\r')"

# --- teardown ---------------------------------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spec_verify_parity.XXXXXX")"
SERVE_PID=""
kill_serve() {
  if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 0.25; done
    kill -9 "$SERVE_PID" 2>/dev/null || true
  fi
  SERVE_PID=""
  # belt-and-braces: free the port regardless of how serve exited
  lsof -ti "tcp:${PORT}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}
cleanup() { kill_serve; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

# --- materialize each column's prompt verbatim, emit "id<TAB>n_gen" lines ----
node -e '
  const fs = require("fs");
  const pack = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  const dir = process.argv[2];
  const out = [];
  for (const c of pack.columns) {
    fs.writeFileSync(`${dir}/${c.id}.txt`, String(c.prompt));   // exact bytes, no template
    out.push(`${c.id}\t${c.n_gen}`);
  }
  process.stdout.write(out.join("\n") + "\n");
' "$PROMPTS_JSON" "$WORK" > "$WORK/_columns.tsv"

echo "[spec-verify-parity] bin=$BIN ($BIN_VERSION)"
echo "[spec-verify-parity] model=$(basename "$MODEL")  drafter=$DRAFTER  addr=$ADDR"
echo "[spec-verify-parity] gate columns: $GATE_COLUMNS"
echo

# ---------------------------------------------------------------------------
# Serve helpers
# ---------------------------------------------------------------------------
# Start `camelid serve` in the background with the resident stack on. $1 = log file,
# $2 = "spec" (speculation + GPU verify + verify trace) or "baseline" (no speculation).
start_serve() {
  local logf="$1" mode="$2"
  kill_serve
  : > "$logf"
  local -a env=(
    CAMELID_METAL_RESIDENT_DECODE=1
    CAMELID_METAL_WIRE=1
    CAMELID_METAL_WIRE_NSG8=1
    CAMELID_METAL_F32Y=1
    CAMELID_METAL_NOCOPY=1
    CAMELID_NO_OPEN=1
  )
  if [ "$mode" = "spec" ]; then
    env+=(
      CAMELID_SPEC_DECODE="$DRAFTER"
      CAMELID_SPEC_GPU=1
      CAMELID_SPEC_VERIFY_TRACE=1
    )
  fi
  env "${env[@]}" "$BIN" serve --addr "$ADDR" --model "$MODEL" --no-open \
    >"$logf" 2>&1 &
  SERVE_PID=$!
}

# Block until serve prints its ready marker (or it dies / times out). 120s ceiling.
wait_ready() {
  local logf="$1" i
  for i in $(seq 1 120); do
    if grep -q "Camelid is ready" "$logf" 2>/dev/null; then return 0; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "[spec-verify-parity] serve exited before ready; tail:" >&2
      tail -n 20 "$logf" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "[spec-verify-parity] serve did not become ready within 120s" >&2
  tail -n 20 "$logf" >&2 || true
  return 1
}

# Count cumulative `[metal-spec-verify]` trace lines in a serve log. grep -c prints "0"
# and exits 1 on zero matches, so neutralize the exit and guarantee a single integer.
count_traces() {
  local n
  n="$(grep -c '\[metal-spec-verify\]' "$1" 2>/dev/null || true)"
  printf '%s' "${n:-0}"
}

# Fetch the loaded model id from /v1/models (the API keys requests on it).
fetch_model_id() {
  curl -s "http://${ADDR}/v1/models" \
    | node -e 'const d=JSON.parse(require("fs").readFileSync(0,"utf8"));process.stdout.write((d.data&&d.data[0]&&d.data[0].id)||"")'
}

# POST one default-greedy completion (NO sampling params -> spec engages when enabled).
# $1 = model id, $2 = prompt file, $3 = max_tokens, $4 = output JSON file.
do_completion() {
  local mid="$1" pf="$2" ngen="$3" outf="$4"
  local body
  body="$(node -e '
    const fs=require("fs");
    const prompt=fs.readFileSync(process.argv[2],"utf8");
    process.stdout.write(JSON.stringify({
      model: process.argv[1],
      prompt,
      max_tokens: parseInt(process.argv[3],10),
    }));
  ' "$mid" "$pf" "$ngen")"
  curl -s --max-time 300 "http://${ADDR}/v1/completions" \
    -H 'Content-Type: application/json' -d "$body" > "$outf"
}

# ---------------------------------------------------------------------------
# Metal resident lane (THE GATE)
# ---------------------------------------------------------------------------
echo "== Metal resident speculative-verify lane (Phase 3 GATE) =="

# Pass 1: speculation ON. Collect spec output + per-column verify-trace evidence.
echo "[metal-lane] starting spec serve (CAMELID_SPEC_DECODE=$DRAFTER CAMELID_SPEC_GPU=1, resident stack on)…"
SPEC_LOG="$WORK/serve.spec.log"
start_serve "$SPEC_LOG" spec
wait_ready "$SPEC_LOG" || { echo "[metal-lane] spec serve failed to start" >&2; exit 3; }
MODEL_ID="$(fetch_model_id)"
[ -n "$MODEL_ID" ] || { echo "[metal-lane] could not resolve model id from /v1/models" >&2; exit 3; }
echo "[metal-lane] model id: $MODEL_ID  (spec serve ready)"

for col in $GATE_COLUMNS; do
  ngen="$(awk -F'\t' -v c="$col" '$1==c{print $2}' "$WORK/_columns.tsv")"
  [ -n "$ngen" ] || { echo "[metal-lane] WARN: column '$col' not in prompts.json — skipping" >&2; continue; }
  # snapshot cumulative trace count, run the spec request, capture this column's traces
  before="$(count_traces "$SPEC_LOG")"
  do_completion "$MODEL_ID" "$WORK/${col}.txt" "$ngen" "$WORK/${col}.spec.json"
  after="$(count_traces "$SPEC_LOG")"
  delta=$(( after - before ))
  if [ "$delta" -gt 0 ]; then
    grep '\[metal-spec-verify\]' "$SPEC_LOG" | tail -n "$delta" > "$WORK/${col}.trace.txt"
  else
    : > "$WORK/${col}.trace.txt"
  fi
  echo "[metal-lane] spec   $col: verify_batch rounds=$delta"
done
kill_serve

# Pass 2: speculation OFF. Same model, same resident stack, plain greedy baseline.
echo "[metal-lane] starting baseline serve (speculation OFF, resident stack on)…"
BASE_LOG="$WORK/serve.base.log"
start_serve "$BASE_LOG" baseline
wait_ready "$BASE_LOG" || { echo "[metal-lane] baseline serve failed to start" >&2; exit 3; }
BASE_MODEL_ID="$(fetch_model_id)"
echo "[metal-lane] baseline serve ready (model id: $BASE_MODEL_ID)"
for col in $GATE_COLUMNS; do
  ngen="$(awk -F'\t' -v c="$col" '$1==c{print $2}' "$WORK/_columns.tsv")"
  [ -n "$ngen" ] || continue
  do_completion "$BASE_MODEL_ID" "$WORK/${col}.txt" "$ngen" "$WORK/${col}.base.json"
  echo "[metal-lane] base   $col: done"
done
# Sanity: the baseline server must NOT have fired any Metal spec verify.
base_traces="$(count_traces "$BASE_LOG")"
kill_serve
echo "[metal-lane] baseline server verify traces=$base_traces (expected 0)"
echo

# ---------------------------------------------------------------------------
# Metal resident TREE speculative-verify lane (THE PHASE 4 GATE)
# ---------------------------------------------------------------------------
# verify_tree_gpu → verify_tree_metal → verify_batch_tree is NOT reachable from `serve`
# (api/mod.rs's speculative loop only ever calls the LINEAR verify_drafts_gpu; CAMELID_SPEC_TREE
# is consulted only by `bench-speculative`/generate_run_speculative, which re-enables the
# resident paths). So this lane drives `bench-speculative` with CAMELID_SPEC_TREE=1 + the SAME
# resident fast stack the serve lane uses. bench-speculative computes its OWN lossless verdict
# (spec stream vs this build's plain greedy decode, byte-identical) and reports gpu/cpu verify
# rounds; this lane GATES on LOSSLESS && gpu_verify_rounds>0 per column.
echo "== Metal resident TREE speculative-verify lane (Phase 4 GATE) =="
TREE_TSV="$WORK/tree_lane.tsv"
: > "$TREE_TSV"

# Run one tree column through bench-speculative with the resident stack + CAMELID_SPEC_TREE=1,
# under a hard timeout (stock macOS has no GNU `timeout`). The single JSON record lands in $4;
# stderr (with the [metal-tree-verify] trace) lands in $5.
run_tree_lane() {
  local id="$1" pf="$2" ngen="$3" outf="$4" errf="$5"
  env \
    CAMELID_METAL_RESIDENT_DECODE=1 \
    CAMELID_METAL_WIRE=1 \
    CAMELID_METAL_WIRE_NSG8=1 \
    CAMELID_METAL_F32Y=1 \
    CAMELID_METAL_NOCOPY=1 \
    CAMELID_NO_OPEN=1 \
    CAMELID_SPEC_TREE=1 \
    CAMELID_SPEC_TREE_GATE=0 \
    CAMELID_SPEC_VERIFY_TRACE=1 \
    "$BIN" bench-speculative "$MODEL" \
      --drafter "$DRAFTER" --workload "$id" --prompt-file "$pf" \
      --max-tokens "$ngen" --warmup >"$outf.raw" 2>"$errf" &
  local bpid=$!
  ( sleep "$TREE_TIMEOUT"; pkill -9 -P "$bpid" 2>/dev/null; kill -9 "$bpid" 2>/dev/null ) &
  local watcher=$!
  wait "$bpid" 2>/dev/null || true
  kill "$watcher" 2>/dev/null || true
  grep -E '^[[:space:]]*\{' "$outf.raw" 2>/dev/null | tail -n1 > "$outf" || true
}

for col in $TREE_COLUMNS; do
  ngen="$(awk -F'\t' -v c="$col" '$1==c{print $2}' "$WORK/_columns.tsv")"
  [ -n "$ngen" ] || { echo "[tree-lane] WARN: column '$col' not in prompts.json — skipping" >&2; continue; }
  run_tree_lane "$col" "$WORK/${col}.txt" "$ngen" "$WORK/${col}.tree.out" "$WORK/${col}.tree.err"
  json="$(cat "$WORK/${col}.tree.out" 2>/dev/null || true)"
  # Max tree fan-out observed this column (from the [metal-tree-verify] trace: max_fanout=N).
  maxfan="$(grep -o 'max_fanout=[0-9]*' "$WORK/${col}.tree.err" 2>/dev/null | sed 's/max_fanout=//' | sort -n | tail -n1)"
  [ -n "$maxfan" ] || maxfan=0
  ttraces="$(grep -c '\[metal-tree-verify\]' "$WORK/${col}.tree.err" 2>/dev/null || true)"; ttraces="${ttraces:-0}"
  if [ -z "$json" ]; then
    echo "[tree-lane] $col: NO RECORD (timed out at ${TREE_TIMEOUT}s or no output) -> FAIL"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$col" "norecord" "-1" "0" "0" "$maxfan" "$ttraces" >> "$TREE_TSV"
    continue
  fi
  parsed="$(node -e '
    const r=JSON.parse(process.argv[1]);
    const div=r.first_divergent_generated_token_index;
    const verdict=(r.lossless===true && div<0)?"LOSSLESS":"DIVERGE";
    process.stdout.write([verdict,div,r.gpu_verify_rounds||0,r.cpu_verify_rounds||0].join("\t"));
  ' "$json")"
  IFS=$'\t' read -r verdict div gpu_rounds cpu_rounds <<<"$parsed"
  echo "[tree-lane] $col: $verdict (gpu_verify_rounds=$gpu_rounds cpu_verify_rounds=$cpu_rounds div=$div max_fanout=$maxfan)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$col" "$verdict" "$div" "$gpu_rounds" "$cpu_rounds" "$maxfan" "$ttraces" >> "$TREE_TSV"
done
echo

# ---------------------------------------------------------------------------
# Optional informational CPU-fallback bench lane (§A1) — never gates
# ---------------------------------------------------------------------------
CPU_BENCH_TSV="$WORK/cpu_bench.tsv"
: > "$CPU_BENCH_TSV"
if [ "$SKIP_CPU_BENCH" != "1" ]; then
  echo "== bench-speculative reference lane (INFORMATIONAL ONLY — does NOT gate; cf. §A1/§A2) =="
  # Run one bench-speculative lane in the background with a hard per-run timeout
  # (stock macOS has no GNU coreutils `timeout`). bench-speculative is the direct
  # child so the watcher reliably kills it; the single JSON record lands in $outf.
  run_cpu_bench_lane() {
    local id="$1" pf="$2" ngen="$3" lane="$4" outf="$5" tree=""
    [ "$lane" = "tree" ] && tree="1"
    CAMELID_SPEC_TREE="$tree" "$BIN" bench-speculative "$MODEL" \
      --drafter "$DRAFTER" --workload "$id" --prompt-file "$pf" \
      --max-tokens "$ngen" --warmup >"$outf.raw" 2>"$WORK/${id}.${lane}.err" &
    local bpid=$!
    ( sleep "$CPU_BENCH_TIMEOUT"; pkill -9 -P "$bpid" 2>/dev/null; kill -9 "$bpid" 2>/dev/null ) &
    local watcher=$!
    wait "$bpid" 2>/dev/null || true
    kill "$watcher" 2>/dev/null || true
    grep -E '^[[:space:]]*\{' "$outf.raw" 2>/dev/null | tail -n1 > "$outf" || true
  }
  while IFS=$'\t' read -r id ngen; do
    [ -n "$id" ] || continue
    pf="$WORK/${id}.txt"
    for lane in linear tree; do
      run_cpu_bench_lane "$id" "$pf" "$ngen" "$lane" "$WORK/${id}.${lane}.out"
      json="$(cat "$WORK/${id}.${lane}.out" 2>/dev/null || true)"
      if [ -z "$json" ]; then
        printf '  %-22s %-6s : (no record — timed out at %ss or no spec round) [informational]\n' "$id" "$lane" "$CPU_BENCH_TIMEOUT"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$lane" "norecord" "-1" "0" "0" "norecord" >> "$CPU_BENCH_TSV"
        continue
      fi
      parsed="$(node -e '
        const r=JSON.parse(process.argv[1]);
        const div=r.first_divergent_generated_token_index;
        const ok=(r.lossless===true && div<0);
        const pct=Math.round((r.accept_rate||0)*100);
        process.stdout.write([ok?"LOSSLESS":"DIVERGE",div,pct,r.gpu_verify_rounds||0,r.cpu_verify_rounds||0].join("\t"));
      ' "$json")"
      IFS=$'\t' read -r verdict div pct gpu_rounds cpu_rounds <<<"$parsed"
      # Derive the verify path honestly from the round counters (NOT asserted).
      if [ "$gpu_rounds" -gt 0 ]; then vpath="metal-verify"
      elif [ "$cpu_rounds" -gt 0 ]; then vpath="cpu-chunk"
      else vpath="no-spec-round"; fi
      printf '  %-22s %-6s : %-8s via %-13s (div_idx=%s accept=%s%% gpu_rounds=%s cpu_rounds=%s) [informational]\n' \
        "$id" "$lane" "$verdict" "$vpath" "$div" "$pct" "$gpu_rounds" "$cpu_rounds"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$lane" "$verdict" "$div" "$cpu_rounds" "$gpu_rounds" "$vpath" >> "$CPU_BENCH_TSV"
    done
  done < "$WORK/_columns.tsv"
  echo "  (informational, NOT gating. 'via' is derived live from the round counters: on this"
  echo "   Phase 3 build the linear lane runs the bit-exact Metal verify too — cpu-chunk would"
  echo "   be the §A1 divergent path, seen only when the GPU verify is unavailable.)"
  echo
else
  echo "== CPU-fallback bench lane skipped (SKIP_CPU_BENCH=1) =="
  echo
fi

# ---------------------------------------------------------------------------
# Verdict + receipt (Metal lane governs the exit code)
# ---------------------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHORTSHA="$(printf '%s' "$BIN_VERSION" | sed -n 's/.*-g\([0-9a-f]\{7,\}\).*/\1/p')"
[ -n "$SHORTSHA" ] || SHORTSHA="unknown"
RECEIPT="$RECEIPTS_DIR/spec-verify-${STAMP}-g${SHORTSHA}.json"
RECEIPT_LATEST="$RECEIPTS_DIR/spec-verify-latest.json"

META="$(node -e '
  process.stdout.write(JSON.stringify({
    binary_version: process.argv[1],
    binary_path:    process.argv[2],
    short_sha:      process.argv[3],
    model_id:       process.argv[4],
    model_path:     process.argv[5],
    model_file:     require("path").basename(process.argv[5]),
    host:           process.argv[6],
    arch:           process.argv[7],
    os:             process.argv[8],
    addr:           process.argv[9],
    drafter:        process.argv[10],
    baseline_server_traces: parseInt(process.argv[11],10),
    generated_at:   new Date().toISOString(),
  }));
' "$BIN_VERSION" "$BIN" "$SHORTSHA" "$MODEL_ID" "$MODEL" \
  "$(hostname -s)" "$(uname -m)" "$(uname -sr)" "$ADDR" "$DRAFTER" "$base_traces")"

# build_receipt.js: compares spec vs baseline token streams per gate column, folds in
# the verify-trace evidence + the informational CPU bench, writes the receipt, prints a
# human verdict, and EXITS 0 (all gate columns lossless AND verify_batch fired) or 1.
cat > "$WORK/build_receipt.js" <<'NODE'
const fs = require("fs");
const crypto = require("crypto");
const [, , promptsJson, work, gateCols, metaJson, receiptPath, latestPath, cpuTsv, treeTsv] = process.argv;
const pack = JSON.parse(fs.readFileSync(promptsJson, "utf8"));
const ngenOf = Object.fromEntries(pack.columns.map(c => [c.id, c.n_gen]));
const meta = JSON.parse(metaJson);
const cols = gateCols.trim().split(/\s+/).filter(Boolean);

const sha = buf => crypto.createHash("sha256").update(buf).digest("hex");
const tokSha = ids => sha(Buffer.from(new Uint32Array(ids).buffer));
const tokens = j => (j && j.camelid && j.camelid.generated_token_ids) || null;
const text = j => (j && j.choices && j.choices[0] && j.choices[0].text) || "";

function parseTrace(file) {
  let raw = "";
  try { raw = fs.readFileSync(file, "utf8"); } catch (_) { raw = ""; }
  const rounds = [];
  for (const line of raw.split("\n")) {
    const m = line.match(/base=(\d+)\s+k=(\d+)\s+accepted=(\d+)\s+emitted_len=(\d+)/);
    if (m) rounds.push({ base_position: +m[1], k: +m[2], accepted: +m[3], emitted_len: +m[4] });
  }
  return rounds;
}

const columns = [];
let allLossless = true, allFired = true;
for (const id of cols) {
  let spec = null, base = null;
  try { spec = JSON.parse(fs.readFileSync(`${work}/${id}.spec.json`, "utf8")); } catch (_) {}
  try { base = JSON.parse(fs.readFileSync(`${work}/${id}.base.json`, "utf8")); } catch (_) {}
  const st = tokens(spec), bt = tokens(base);
  const rounds = parseTrace(`${work}/${id}.trace.txt`);
  const fired = rounds.length;
  const acceptedTotal = rounds.reduce((a, r) => a + r.accepted, 0);
  const maxK = rounds.reduce((a, r) => Math.max(a, r.k), 0);
  const haveBoth = Array.isArray(st) && Array.isArray(bt);
  const specSha = haveBoth ? tokSha(st) : null;
  const baseSha = haveBoth ? tokSha(bt) : null;
  const specTextSha = spec ? sha(Buffer.from(text(spec))) : null;
  const baseTextSha = base ? sha(Buffer.from(text(base))) : null;
  const lossless = haveBoth && specSha === baseSha && specTextSha === baseTextSha;
  const firedOk = fired > 0;
  if (!lossless) allLossless = false;
  if (!firedOk) allFired = false;
  columns.push({
    id, n_gen: ngenOf[id] ?? null,
    spec_token_count: st ? st.length : null,
    baseline_token_count: bt ? bt.length : null,
    spec_tokens_sha256: specSha,
    baseline_tokens_sha256: baseSha,
    spec_text_sha256: specTextSha,
    baseline_text_sha256: baseTextSha,
    lossless, bit_identical: lossless,
    verify_batch_fired: firedOk,
    verify_rounds: fired,
    drafts_accepted_total: acceptedTotal,
    max_k: maxK,
    rounds, // per-round base_position / k / accepted / emitted_len (plan §5)
    gate_pass: lossless && firedOk,
    spec_error: spec && spec.error ? spec.error : undefined,
    baseline_error: base && base.error ? base.error : undefined,
  });
}

// informational bench-speculative reference summary (never gates)
let cpu = { status: "skipped" };
try {
  const lines = fs.readFileSync(cpuTsv, "utf8").trim().split("\n").filter(Boolean);
  if (lines.length) {
    const rows = lines.map(l => {
      const [id, lane, verdict, div, cpu_rounds, gpu_rounds, verify_path] = l.split("\t");
      return {
        id, lane, verdict,
        first_divergent_index: +div,
        cpu_verify_rounds: +cpu_rounds,
        gpu_verify_rounds: +gpu_rounds,
        verify_path,
      };
    });
    cpu = {
      status: "informational",
      gating: false,
      note: "bench-speculative reference lane (NOT gating). recon §A1/§A2 described this " +
            "(at base 28f224b) as the CPU chunk-verify fallback that diverges once a verify " +
            "round fires. HONEST POST-PHASE-3 UPDATE: on this Phase 3 build the linear lane " +
            "reaches the SAME bit-exact Metal verify (gpu_verify_rounds>0, cpu_verify_rounds=0) " +
            "and is lossless; the §A1 CPU-chunk divergence reproduces only when the GPU verify " +
            "path is unavailable. verify_path is derived live from the round counters.",
      lossless: rows.filter(r => r.verdict === "LOSSLESS").length,
      diverged: rows.filter(r => r.verdict === "DIVERGE").length,
      via_metal_verify: rows.filter(r => r.verify_path === "metal-verify").length,
      via_cpu_chunk: rows.filter(r => r.verify_path === "cpu-chunk").length,
      no_spec_round: rows.filter(r => r.verify_path === "no-spec-round").length,
      rows,
    };
  }
} catch (_) {}

// Phase 4 TREE lane (GATING). Driven via bench-speculative CAMELID_SPEC_TREE=1 — the only
// reachable path for verify_tree_gpu→verify_tree_metal→verify_batch_tree (serve has no tree
// branch). Each column must be LOSSLESS (spec stream byte-identical to this build's plain
// greedy, computed inside bench-speculative) AND the GPU tree verify must have fired
// (gpu_verify_rounds>0). A silent CPU fallback or a no-spec-round pass is a FAILURE.
let tree = { status: "absent", gating: true, verdict: "FAIL" };
let treeGatePass = false;
try {
  const lines = fs.readFileSync(treeTsv, "utf8").trim().split("\n").filter(Boolean);
  if (lines.length) {
    const rows = lines.map(l => {
      const [id, verdict, div, gpu_rounds, cpu_rounds, max_fanout, tree_traces] = l.split("\t");
      const lossless = verdict === "LOSSLESS";
      const fired = (+gpu_rounds) > 0;
      return {
        id,
        verdict,
        lossless,
        first_divergent_index: +div,
        gpu_verify_rounds: +gpu_rounds,
        cpu_verify_rounds: +cpu_rounds,
        max_tree_fanout: +max_fanout,
        tree_verify_traces: +tree_traces,   // includes the unmeasured warmup run's traces
        gpu_tree_verify_fired: fired,
        multi_branch_fanout: (+max_fanout) > 1,
        gate_pass: lossless && fired,
      };
    });
    const treeAllLossless = rows.every(r => r.lossless);
    const treeAllFired = rows.every(r => r.gpu_tree_verify_fired);
    const maxFanout = rows.reduce((a, r) => Math.max(a, r.max_tree_fanout), 0);
    treeGatePass = treeAllLossless && treeAllFired;
    tree = {
      status: "gating",
      gating: true,
      verdict: treeGatePass ? "LOSSLESS" : "FAIL",
      reachability: "verify_tree_gpu is NOT reachable from `serve`: the server speculative loop " +
        "(api/mod.rs) only calls the LINEAR verify_drafts_gpu, and CAMELID_SPEC_TREE is consulted " +
        "only by `bench-speculative` (main.rs generate_run_speculative). This lane therefore drives " +
        "bench-speculative with CAMELID_SPEC_TREE=1 (CAMELID_SPEC_TREE_GATE=0, full tree every round) " +
        "+ the resident fast stack; bench-speculative computes the lossless verdict against this " +
        "build's own plain greedy stream.",
      spec_env: {
        CAMELID_SPEC_TREE: "1",
        CAMELID_SPEC_TREE_GATE: "0",
        CAMELID_SPEC_GPU: "(n/a — bench-speculative engages the resident path directly)",
        CAMELID_METAL_RESIDENT_DECODE: "1",
        CAMELID_METAL_WIRE: "1",
        CAMELID_METAL_WIRE_NSG8: "1",
        CAMELID_METAL_F32Y: "1",
        CAMELID_METAL_NOCOPY: "1",
      },
      all_columns_lossless: treeAllLossless,
      all_columns_gpu_tree_verify_fired: treeAllFired,
      max_tree_fanout_observed: maxFanout,
      multi_branch_fanout_fired: maxFanout > 1,   // fan-out>1 ⇒ branching verify + KV compaction exercised
      columns: rows,
    };
  }
} catch (_) {}

const gatePass = allLossless && allFired && treeGatePass;
const totalRounds = columns.reduce((a, c) => a + c.verify_rounds, 0);
const receipt = {
  schema: "camelid.spec-verify/v1",
  generated_at: meta.generated_at,
  lane: "metal-resident-speculative-verify",
  host: `${meta.host} (${meta.arch}, ${meta.os})`,
  binary_version: meta.binary_version,
  binary_path: meta.binary_path,
  binary_short_sha: meta.short_sha,
  model: { id: meta.model_id, file: meta.model_file, path: meta.model_path },
  serve: {
    addr: meta.addr,
    drafter: meta.drafter,
    spec_env: {
      CAMELID_SPEC_DECODE: meta.drafter,
      CAMELID_SPEC_GPU: "1",
      CAMELID_SPEC_VERIFY_TRACE: "1",
      CAMELID_METAL_RESIDENT_DECODE: "1",
      CAMELID_METAL_WIRE: "1",
      CAMELID_METAL_WIRE_NSG8: "1",
      CAMELID_METAL_F32Y: "1",
      CAMELID_METAL_NOCOPY: "1",
    },
    baseline_request: "default greedy, speculation OFF",
    spec_request: "default greedy (no sampling params), speculation ON",
    baseline_server_verify_traces: meta.baseline_server_traces,
  },
  overall_verdict: gatePass ? "LOSSLESS" : "FAIL",   // governs the exit code: linear AND tree lanes
  gate: {
    // Linear lane (Phase 3 metal-resident verify_batch via serve).
    verdict: (allLossless && allFired) ? "LOSSLESS" : "FAIL",
    all_columns_lossless: allLossless,
    all_columns_verify_fired: allFired,
    total_verify_rounds: totalRounds,
    columns: cols.length,
  },
  tree,   // Phase 4 tree lane (verify_tree_metal/verify_batch_tree via bench-speculative)
  columns,
  cpu_fallback_bench: cpu,
};

fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + "\n");
fs.writeFileSync(latestPath, JSON.stringify(receipt, null, 2) + "\n");

// human verdict to stderr
console.error("== Metal resident lane verdict ==");
for (const c of columns) {
  const id = c.id.padEnd(22);
  const verdict = c.lossless ? "LOSSLESS" : "DIVERGE ";
  const fired = c.verify_batch_fired ? `verify_batch fired (rounds=${c.verify_rounds}, accepted=${c.drafts_accepted_total}, max_k=${c.max_k})`
                                     : "verify_batch DID NOT FIRE (would be a silent CPU fallback)";
  console.error(`  ${id} ${verdict}  ${fired}`);
  if (c.lossless) {
    console.error(`    spec==baseline tokens sha256=${(c.spec_tokens_sha256||"").slice(0,16)}… (${c.spec_token_count} tok)`);
  } else {
    console.error(`    spec tokens sha=${(c.spec_tokens_sha256||"null")} baseline=${(c.baseline_tokens_sha256||"null")}`);
    if (c.spec_error) console.error(`    spec_error: ${JSON.stringify(c.spec_error).slice(0,160)}`);
    if (c.baseline_error) console.error(`    baseline_error: ${JSON.stringify(c.baseline_error).slice(0,160)}`);
  }
}
console.error("");
console.error("== Metal resident TREE lane verdict (Phase 4) ==");
if (tree.status !== "gating") {
  console.error("  TREE LANE PRODUCED NO RECORDS — FAIL (verify_tree_metal never observed)");
} else {
  for (const r of tree.columns) {
    const id = r.id.padEnd(22);
    const verdict = r.lossless ? "LOSSLESS" : "DIVERGE ";
    const fired = r.gpu_tree_verify_fired
      ? `tree verify fired (gpu_rounds=${r.gpu_verify_rounds}, cpu_rounds=${r.cpu_verify_rounds}, max_fanout=${r.max_tree_fanout}${r.multi_branch_fanout ? " MULTI-BRANCH" : " single-branch"})`
      : "tree verify DID NOT FIRE on GPU (silent CPU fallback / no spec round)";
    console.error(`  ${id} ${verdict}  ${fired}`);
  }
  console.error(`  max tree fan-out observed across columns: ${tree.max_tree_fanout_observed}` +
                (tree.multi_branch_fanout_fired ? " (genuine multi-branch fan-out exercised)" : " (single-branch only — verify_tree_metal still exercised & lossless)"));
}
console.error("");
console.error(`receipt: ${receiptPath}`);
process.exit(gatePass ? 0 : 1);
NODE

set +e
node "$WORK/build_receipt.js" \
  "$PROMPTS_JSON" "$WORK" "$GATE_COLUMNS" "$META" "$RECEIPT" "$RECEIPT_LATEST" "$CPU_BENCH_TSV" "$TREE_TSV"
GATE_RC=$?
set -e

echo
if [ "$GATE_RC" -eq 0 ]; then
  # Honest fan-out claim: only assert "multi-branch" if the receipt recorded fan-out>1 this run
  # (the drafter may produce only single-branch trees on a given run; the multi-branch +
  # KV-compaction proof is the unit gate metal_tree_verify_bit_identical, always exercised there).
  TREE_FANOUT="$(node -e 'try{const r=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"));process.stdout.write(r.tree&&r.tree.multi_branch_fanout_fired?"multi":"single")}catch(e){process.stdout.write("single")}' "$RECEIPT_LATEST" 2>/dev/null || echo single)"
  if [ "$TREE_FANOUT" = "multi" ]; then
    TREE_NOTE="with genuine multi-branch fan-out this run"
  else
    TREE_NOTE="single-branch trees this run — multi-branch+compaction proven by the unit gate metal_tree_verify_bit_identical"
  fi
  echo "PASS: Metal resident speculative-verify is LOSSLESS (byte-identical to plain greedy) on"
  echo "      every gate column for BOTH lanes — the Phase 3 LINEAR verify_batch (via serve) and"
  echo "      the Phase 4 TREE verify_batch_tree (via bench-speculative) — and the GPU verify"
  echo "      demonstrably fired on each (linear: verify_batch; tree: verify_tree_metal,"
  echo "      $TREE_NOTE). CPU-fallback bench lane (if run) is informational"
  echo "      only (§A1) and did NOT affect this result."
else
  echo "FAIL: a speculative-verify gate lane did not pass — a gate column either diverged from"
  echo "      plain greedy or never exercised the GPU verify (linear verify_batch or the Phase 4"
  echo "      tree verify_tree_metal). See the per-lane verdict above."
fi
exit "$GATE_RC"
