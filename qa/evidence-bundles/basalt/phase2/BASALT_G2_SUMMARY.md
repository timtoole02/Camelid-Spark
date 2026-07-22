# BASALT Gate G2 — Phase 2 summary (GGUF load and cross-engine interop)

Status: FINAL — all Phase 2 lanes complete 2026-07-16.
Branch: `basalt/phase2-load-interop` (off main 939710de = post-G1). Bundle:
`qa/evidence-bundles/basalt/phase2/`. Protocol: `basalt_eval_protocol.md` incl. Amendment 2.

## 1. Produced rows (pin llama-quantize, Q8_0-baseline source per Amendment 2)

| row | file | sha256 | size (B) | deterministic ×2 | dry-run verified |
|---|---|---|---|---|---|
| NVFP4-mm | gemma-4-E4B-it-NVFP4-mm.gguf | `eb293344972e2b292a043b8e7649b9788dca915b034e5c2721cfc531cf9863d9` | 6,058,607,776 | YES (sha-identical ×2) | 294 overrides ✔, embd q8_0 mmap-copy ✔ |
| Q4K-mm | gemma-4-E4B-it-Q4K-mm.gguf | `d306fa7753ec80eda4ab389f1b3a06273cc2291ebe977993a767587c055b2031` | 6,058,607,776 | YES | 294 q4_K, byte-count identical to NVFP4-mm (both 4.5 bpw) |
| Q4_K_M-df | gemma-4-E4B-it-Q4_K_M-df.gguf | `4f7f288ad56b9c64ddab0bdb25af1b14757f9470181252885f496ebbf18b25b5` | 6,180,443,296 | YES | mixture: 336 q4_K / 42 q6_K (21 attn_v + 21 ffn_down, the alternating promotion) / 2 q8_0 embd (pinned) |
| Q4_K_M-im | gemma-4-E4B-it-Q4_K_M-im.gguf | `3f5a71ecf3bbd5ca7c7d6ee318208fe0ffd743935a2e3ad71b933afe809fde53` | 6,180,443,296 | YES | same types as -df, values differ (imatrix, 342 entries loaded clean) |
| NVFP4-all | — | — | — | **BLOCKED-HOST** (Amendment 2; ~22.5 GB staging receipt) | — |

Baseline: gemma-4-E4B-it-Q8_0.gguf `a2232a64…` (verified at download and before each leg).

## 2. Measured size table (replaces recon §7 projection)

- NVFP4-mm measured file: **6,058,607,776 B**, reconciled EXACTLY against the Phase 0
  projection: 6,204,494,296 (projected tensor bytes) − 161,710,080 (84 `inp_gate`/`proj`
  F32→Q8_0, 2,621,440→696,320 B each — the projection had kept them F32) − 2,352
  (projected sidecar tensors that `llama-quantize` does not produce) + 15,824,717 (header)
  + 1,195 (alignment padding — the same residual as Phase 0's byte-closure check)
  = 6,058,607,776. Zero unexplained bytes.
- All rows measured (table §1). Peak free RAM never below 7.89 GB across all legs (4 GB floor); ~24.5 GB written (+ equal deleted determinism temps).
- 6 GB residency: unchanged verdict for full residency (weights alone ~5.6 GiB resident);
  matmul-only GPU set unchanged ~2.44 GB (Phase 4 design space).

## 3. Type census + keep-list, observed (NVFP4-mm)

F32 339 / Q8_0 86 / BF16 1 / NVFP4 294 (=7 families × 42 layers) — CONFIRMED three ways:
spot-check header walk, quantize dry-run/type logs, and engine-side `camelid inspect`
(`pilot_inspect_receipt.md`). Q8_0 decomposes exactly: 2 kept embeddings + 84 disclosed
`inp_gate`/`proj` F32→Q8_0 conversions. No foreign types. ftype KV = 39.

## 4. Amendment 2 copy-through — proven at byte level

`token_embd.weight` (713,031,680 B) and `per_layer_token_embd.weight` (2,994,733,056 B)
byte-range hashes IDENTICAL between baseline and NVFP4-mm; 3 sampled norms identical;
`inp_gate`/`proj` differ exactly as disclosed (F32→Q8_0). Receipt:
`kept_tensor_identity.json` (bundle).

## 5. Cross-engine dequant spot-check (pin ↔ Camelid)

Pin side: 5,120 blocks from 10 tensors (all 7 matmul families, depths blk.0–blk.41),
dequanted via the pin DLL (route linked-libs, same MSVC provenance as the Phase 1
fixtures), run twice byte-identical, and every block reconstructs bit-exactly from the
committed decode-table fixture. Fixture (committed as
`tests/fixtures/dequant/nvfp4_e4b_spotcheck.json`, `nvfp4_real_blocks.json` schema).
Camelid side: DONE — `tests/nvfp4_e4b_spotcheck.rs` consumes that fixture: all 5,120
blocks bit-exact through BOTH paths (nvfp4_wire_block_dequant per block;
decode_nvfp4_tensor over concatenated groups), family-coverage asserted, fixture pinned
to the receipted pilot sha. Full suite 1205/0 at assembly (+2 tests in the review-fix
commit: the real-pilot BF16 admission shape and the always-on reader id/layout pin).

## 6. Engine wiring (refusal-point move + admission scoping) — commit 63eeba5f

Landed: enum `NVFP4` (id 40, layout 64/36, Debug name = receipt label); ftype **39** map
entry; dequant dispatch → Phase 1 fail-closed decoder (0x7F AND 0xFF sentinel refusal
unit-tested); D-B2 sidecar fail-closed at admission (fires only when NVFP4 present;
verified no false-positive on the pilot's real `layer_output_scale.weight` names);
seam-split comment in place. **Design decision for Tim's G2 review**: gemma4 is
deliberately not a runnable-lane architecture, so D-B3 was implemented as a mirror-image
carve-out — gemma4 passes the architecture axis **iff the file carries ≥1 NVFP4 tensor**;
gemma4-without-NVFP4 keeps today's refusal verbatim (supported-lane Q8_0 row unaffected);
NVFP4 outside gemma4 rejects `axis=quant` with a message naming D-B3. Admit-then-fail is
excluded: smoke stays oracle-gated, the serve bridge does not route gemma4.
Refusal-move receipts (bundle `refusal_move_receipt.md`): BEFORE = phase-0 parse refusal
(`Unknown(40)`, exit 1); AFTER = qwen3 file inspect exit 0 (census 197 NVFP4/113 F32,
ftype 39) + runnable-smoke exit 1 with the D-B3 pilot-scope message. Full suite 1204/0,
clippy `-D warnings` + fmt clean, privacy 0 findings, checksums + public scrub pass.
Disclosed process slip: one clippy invocation overlapped a quantize-leg start
(compile-only, no harm; gate-then-run pattern reinstated).

## 6b. Lane clarity (pilot vs runnable lane)

The E4B pilot contains one BF16 tensor (`per_layer_model_proj`), and the runnable lane
rejects BF16 (MUSTER M-B5 precedent) — so the pilot's runnable-lane admission receipt may
refuse on BF16 grounds independent of NVFP4. This is correct and disclosed: the pilot's
execution home is the gemma4 supported-lane runtime (Phase 3 adds its NVFP4 WireFormat);
D-B3's runnable-lane NVFP4 coverage is receipted via the qwen3 refusal-move captures
(pilot-scope refusal) and admission unit tests. Observed outcome, recorded verbatim in the
engine receipts: runnable-smoke on BOTH real E4B NVFP4 files (ours and the wild one)
refuses on the BF16 tensor, exactly as predicted — the honest pre-Phase-3 state.

## 7. Wild-file interop (optional leg, unknown provenance, no claims)

`FreedomAISVR/Gemma-4-E4B-it-NVFP4-GGUF` (sha `ea8cac5b…`, 5,185,929,952 B, long-tail
uploader, 2026-05-18): header census = **378 NVFP4** (the 294 matmuls PLUS the 84
`inp_gate`/`proj`), **2 Q6_K embeddings** (re-staged on a big-RAM host — independent
confirmation the BLOCKED-HOST limit is this host, not the format), 1 BF16, 339 F32,
**zero sidecar tensors** (llama-quantize convention, not ModelOpt — the D-B2 refusal is
NOT exercised by this file; its coverage remains synthetic-fixture until a sidecar-bearing
fixture exists). Camelid outcome (engine receipts): **parses and refuses cleanly** —
recorded verbatim, unknown provenance, no support claims either way.

## 8. G2 checklist (conductor §5)

- [x] pilot NVFP4 GGUF loads clean (inspect exit 0 with full census; admission outcomes recorded verbatim incl. the disclosed BF16 runnable-lane refusal — §6b)
- [x→bundle] sha256 + tensor-type inventory receipts (rows as produced)
- [x] measured size table (§1/§2 of this summary; all four rows measured + deterministic)
- [x] sampled-tensor dequant parity receipts (pin-side extraction + Camelid-side test, both bit-exact — §5)
- Phase 2b (native quantizer): NOT exercised (D-B5: deprioritized)
- STOP → ping Tim (this gate is his review point)

## 9. Disclosed deviations & provenance gotchas (quantize legs)

- inp_gate/proj (84 F32 tensors) → q8_0 in the -mm rows but **q4_K in the Q4_K_M rows** (the
  mixture's default; only embeddings were pinned). Within-pair identical, so the GATED
  comparison and the df-vs-im comparison are clean; mm-vs-Q4_K_M practical comparisons
  carry this small disclosed confound.
- **All four rows inherit `quantize.imatrix.*` KV keys from the upstream Q8_0 source** —
  including Q4_K_M-df, which had NO imatrix applied here. Receipt readers must not infer
  imatrix use from those keys; the authoritative statement is the command lines in the
  quantize logs (this bundle).
- NVFP4-mm and Q4K-mm are byte-count identical — never distinguish rows by size.
- Review-fix disclosures (post-assembly adversarial review, 8 confirmed findings applied):
  the engine commit had appended an NVFP4 parenthetical to the GENERIC uncovered-quant
  refusal message for all architectures — reverted, the generic message is again
  byte-identical to pre-BASALT main (the D-B3 scope text lives only in the
  NVFP4-specific rejection); the real-pilot admission shape (NVFP4 + BF16 → refuses on
  BF16) is now pinned by an in-tree unit test, matching the receipts here; the
  `log_sha256` block inside `quantize-quantize_legs_summary.json` hashes the
  pre-sanitization scratchpad originals — the authoritative hashes for the committed
  (renamed, path-sanitized) logs are this bundle's `SHA256SUMS`.
