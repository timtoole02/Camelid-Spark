# Wild-file interop pre-receipt (header-level; camelid-side inspect pending)

File: models/wild-FreedomAISVR-gemma-4-E4B-it-NVFP4.gguf
Upstream: FreedomAISVR/Gemma-4-E4B-it-NVFP4-GGUF (long-tail uploader, 2026-05-18)
sha256: ea8cac5b184e19c09583fa2df691e15db1e0fb990be8d8a7ea8123ea50dad8cf
size: 5,185,929,952 B — downloaded 2026-07-16, unknown provenance, NO support claims.

Header census (walker: scratchpad basalt-p2-spotcheck/walk_wild.mjs, GGUF v3, 720 tensors, 49 KVs):
- NVFP4 (id 40): 378 = the 294 matmul weights PLUS the 84 inp_gate/proj
- Q6_K (id 14): 2 = token_embd + per_layer_token_embd (re-staged on a big-RAM host —
  the exact operation that is BLOCKED-HOST here; independent confirmation the limit is
  this host, not the format)
- BF16: 1 (per_layer_model_proj), F32: 339 (norms etc.)
- Sidecar (.scale/.input_scale) tensors: ZERO -> not a ModelOpt conversion; made via the
  same llama-quantize override path as our rows. The D-B2 sidecar refusal is NOT exercised
  by this file (no wild sidecar-bearing GGUF encountered yet; the cosmicproc safetensors
  checkpoint remains the known sidecar source if that receipt is ever needed).

Expected Camelid behavior (engine-side receipt pending): parses; admission = gemma4 arch +
covered quants {NVFP4(gemma4-scoped), Q6_K, F32, BF16... note BF16 coverage per runnable
lane rules} -> record verbatim outcome either way; smoke refuses (combo not oracle-qualified).
