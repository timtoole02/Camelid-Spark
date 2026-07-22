# SIROCCO Phase P — M0b: token-parity reality (KILL gate)

**Result: PASSED.** The flash online-softmax reassociation is **token-identical** to the byte-exact baseline across 6 diverse prompts (incl. near-tie stressors), ctx 6.5k–13.6k, 64/64 greedy tokens each. Flash prefill attention is **token-parity-safe** — the phase proceeds to M1.

## Why this is the decisive gate

Flash tiling reassociates the softmax (online running-max rescale instead of the two-pass global-max→exp→chunked-reduce). Byte-identity is **capacity-blocked** at 8k (would need O(prefix) shared state), so flash prefill is inherently **token-parity**. The campaign's #8 (dp4a q6k) died exactly here — a per-layer f32-association change flipped tokens on 8/8 prompts by compounding across 16 layers. Prefill is *stricter*: it perturbs the persisted KV cache, so error compounds per-layer **and** per-position. M0b answers, for the cost of one env-gated edit (no tiling), whether flash's reassociation survives.

## Method (`m0b-probe.diff`)

A `CAMELID_FLASH_PROBE` compile-gate (`-DFLASH_PROBE`, added to the nvrtc options only when the env is set) selects an **online-rescale** branch in `attention_batched`: a single pass over the raw scores maintaining running `(m, l, acc)` — the *exact* float reassociation a tiled flash kernel introduces — with **zero tiling**. It early-returns, so probe-off is byte-identical to `main`. `bench-generate` uses `attention_batched` only in prefill (no drafter), so probe-on vs probe-off isolates the **prefill-attention reassociation's** effect on the generated tokens.

Oracle: for each prompt, generate 64 greedy tokens with the probe ON vs OFF and diff the token ids. Any single flip on any prompt = KILL (and, since byte-identity is capacity-blocked, no shippable flash variant → abandon Phase P flash).

## Result (`m0b-oracle-result.txt`)

| prompt | content | ctx | tokens |
|---|---|---|---|
| varied6k | narrative | 6602 | 64/64 ✅ |
| promptB | technical | 10232 | 64/64 ✅ |
| promptC | dialogue | 12211 | 64/64 ✅ |
| mp-rep | **near-tie repetition** | 6503 | 64/64 ✅ |
| mp-code | code | 10032 | 64/64 ✅ |
| mp-num | numeric/list | 13651 | 64/64 ✅ |

**6/6 token-identical.** The online-softmax reassociation does not flip a greedy token across the 16 prefill layers, even on the near-tie repetitive prompt engineered to have the closest logit margins.

## Caveat / scope

M0b probes the **numerical reassociation** (online rescale), not the tiling. The real M1 kernel adds a **tiling-specific** reduction order (key-splits + cross-tile combine); the same oracle must be re-run against the real kernel at M1. But M0b removes the phase's single biggest risk — that flash prefill is fundamentally token-parity-dead like #8. It is not.

## Files
- `m0b-probe.diff` — the compile-gated online-rescale probe (the reproducible instrument).
- `m0b-oracle-result.txt` — the 6-prompt oracle run.
