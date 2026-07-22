# Camelid Parity Proof

Last updated: 2026-05-13

Camelid's strongest technical claim is not that it can emit tokens.

It is that, for specific exact GGUF rows and bounded validation packs, Camelid matches llama.cpp token-for-token and text-for-text — and that the repo points to the artifacts that prove it.

This file makes that proof easy to audit.

## What parity means here

Camelid uses **bounded exact-row parity** language.

In this repo, parity means the cited artifact records agreement with llama.cpp for the checked lane on:

- prompt token IDs
- generated token IDs
- generated text

Parity does **not** automatically imply:

- broad family support
- model-native or larger context support
- arbitrary template support
- production throughput
- portability
- neighboring quant or size support

Everything stays exact-row and artifact-scoped.

## Headline parity story

Today Camelid can honestly say:

- **TinyLlama 1.1B Chat Q8_0** is the trusted current gate with parity, template-shape, bounded context, API/WebUI, and RSS/perf evidence.
- **Llama 3.2 1B Instruct Q8_0** has bounded exact-row parity through checked 512/1024/2048/4096/8192 packs.
- **Llama 3.2 3B Instruct Q8_0** has bounded exact-row parity through the anchored checked 512/1024/2048/4096/8192 raw-decode ladder on the current canonical GGUF (token+text identical at 50 greedy tokens, fully GPU-resident on the 6 GiB reference card); the May packs and broader 50-token parity remain historical evidence for the prior upload.
- **Llama 3 8B Instruct Q8_0** has bounded exact-row parity for compact smoke, broader 50-token validation, and checked 512/1024/2048 packs where cited artifacts exist.

That is the public proof surface. Nothing adjacent inherits support.

## Parity map by exact row

| Exact row | Strongest public parity headline | Start here |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | Full current gate with broader five-prompt / 50-token parity plus bounded context/template evidence | `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json` |
| Llama 3.2 1B Instruct Q8_0 | Exact-row bounded parity through 8192, with 4096/8192 tied to cited source/runtime heads | `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json` |
| Llama 3.2 3B Instruct Q8_0 | Exact-row anchored bounded raw-decode parity through the 512/1024/2048/4096/8192 ladder on the current canonical GGUF | `qa/evidence-bundles/llama32-3b-context-512-8192-anchored-20260710T2119-head-6527a770/manifest.json` |
| Llama 3 8B Instruct Q8_0 | Exact-row bounded 1024/2048 parity plus broader 50-token parity on the named source/runtime head | `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` |

## The proof ladder

### TinyLlama 1.1B Chat Q8_0

Primary artifacts:

- `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json`

Why it matters:

- this is Camelid's current fully trusted gate
- it proves the repo is not just “sometimes right” on a single demo prompt

### Llama 3.2 1B Instruct Q8_0

Primary artifacts:

- `qa/evidence-bundles/llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`

What the latest 8192 artifact proves:

- prompt-token parity at `7650` prompt tokens
- generated token parity for `[34,2735,35,12,18831]`
- generated text parity for `CMLD-819`

The 4096 artifact also passed on source/runtime head `470388f8165b` with `3755` prompt tokens and generated text `CMLD-409`.

Why it matters:

- the earlier 2048 red box was real
- the RoPE frequency-factor fix closed it with a clean public artifact
- later 4096/8192 compact-template recall packs strengthen only this exact 1B row, not neighboring rows or model-native/larger context beyond checked packs

### Llama 3.2 3B Instruct Q8_0

Primary context artifact:

- `qa/evidence-bundles/llama32-3b-context-512-8192-anchored-20260710T2119-head-6527a770/manifest.json`

Historical artifacts, measured on the prior upload (sha256 `b5607b50…`) and retained as evidence for that file only:

- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`

What the anchored ladder artifact proves:

- raw-decode greedy parity on all five 512/1024/2048/4096/8192 buckets (`408`/`885`/`1893`/`3978`/`8049` llama3 prompt tokens)
- 50/50 generated tokens and generated text identical to llama.cpp `acd79d603` on every bucket
- the whole ladder ran fully GPU-resident on the 6 GiB reference card against the current canonical GGUF (sha256 `f34112a1…`) — the same file the June 2026 capability receipts pin
- this is raw-decode parity evidence only, not chat-template context coverage

Why it matters:

- this is the row closest to the current product target
- the checked-context claim now rests on a single file identity instead of inheriting May evidence from a prior upload
- it shows Camelid is not hand-waving about 3B correctness before talking about UX/perf work

### Llama 3 8B Instruct Q8_0

Primary artifacts:

- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`
- `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json`

What the current 1024/2048 artifact proves:

- bounded exact-row parity for `CMLD-102` and `CMLD-204`
- prompt tokens, generated token IDs, and generated text all match on the named source/runtime head

Why it matters:

- it answers the old “8B is just groundwork” story with durable bounded evidence
- it also shows Camelid can be exact-row honest without pretending broad 8B support

## Why this deserves to be a headline

Most local inference projects prove they can produce text.

Camelid proves something rarer:

- the exact row is named
- the support boundary is explicit
- the API/UI/docs all point at the same truth
- the parity artifacts are committed and auditable

That combination is the engineering trust story.

## Auditor quick path

If you want the shortest serious review path:

1. Read `COMPATIBILITY.md`
2. Read `STATUS.md`
3. Open the row-specific manifest linked above
4. Verify the row, source head, prompt-token parity, generated-token parity, and generated-text parity
5. Confirm the claim in the README does not exceed that artifact

## Claim boundary

This file highlights parity proof only.

It does not widen Camelid's support contract beyond the exact bounded claims in `COMPATIBILITY.md` and `STATUS.md`.