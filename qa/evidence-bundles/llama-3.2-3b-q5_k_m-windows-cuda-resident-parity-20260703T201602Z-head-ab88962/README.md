# Llama-3.2-3B-Instruct Q5_K_M parity evidence

This bundle certifies the exact `Llama-3.2-3B-Instruct-Q5_K_M.gguf` row for Camelid GPU-resident decode against pinned llama.cpp `acd79d603`.

- Harness: `scripts/raw-decode-parity.mjs`
- Comparator: llama.cpp `/completion`, CPU `-ngl 0`
- Camelid route: GPU-resident CUDA decode, Q5_K/Q6_K wire tensors (`q5k_gemv` + `q6k_gemv`)
- Token counts: 1, 5, 50
- Result: `all_pass=true` in `parity.json`
- Model SHA256: `0b94ccd04d908304cec5246a3d942b64417a423bc5c6d47c73bc557e590b5194`

Note: the harness `proof_chain` string predates Q5_K and mentions q4/q6; this bundle manifest is the controlling row metadata for this Q5_K/Q6_K run.
