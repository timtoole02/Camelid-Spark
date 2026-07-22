# 2026-07-05 - Llama 3.2 3B Q5_K_M CUDA API smoke and WebUI guard

Scope: exact `Llama-3.2-3B-Instruct-Q5_K_M.gguf` local Windows CUDA smoke only. This note records that the runtime can load and answer API requests for this exact file when NVRTC is staged, and that the WebUI correctly remains blocked by the current support contract. It does not promote Q5_K_M support, broad K-quant support, WebUI unlock, production throughput, portability, context length, or neighboring rows.

## Preconditions

The local run required two host/setup prerequisites before Q5_K_M evidence could be collected:

- Windows CLI startup needed the stack-reserve build fix tracked separately in PR #380 or an equivalent local build. Without that, `camelid.exe --version`, `camelid.exe --help`, and `camelid.exe serve --help` overflowed the default Windows stack before model/runtime code ran.
- NVRTC had to be staged beside `camelid.exe`. The local run used the NVIDIA `nvidia-cuda-nvrtc-cu12` 12.9 Windows wheel as the source for `nvrtc64_120_0.dll`, `nvrtc64_120_0.alt.dll`, and `nvrtc-builtins64_129.dll`. The CUDA driver still came from the installed NVIDIA display driver.

Host class: Windows x86_64, NVIDIA RTX laptop GPU with about 8 GiB VRAM. No hostname, user home, SSH command, key path, or private validation host detail is part of this note.

## Result

PASS for local CUDA API smoke:

- backend selected the NVIDIA CUDA path;
- all 28 Llama 3.2 3B Q5_K_M layers were resident in VRAM;
- `/api/models/load` succeeded for the exact Q5_K_M GGUF;
- `/v1/models` listed the loaded model;
- `/api/capabilities` remained unchanged and did not promote Q5_K_M;
- `/v1/completions` generated one token for `hello`;
- `/v1/chat/completions` generated one token for `hello`;
- generation timing summary was written by `scripts/model-promotion-smoke-bundle.mjs`.

Observed API outputs:

- completion prompt tokens: `2`; generated token ids: `[11]`; text: `,`;
- chat prompt tokens: `11`; generated token ids: `[9906]`; text: `Hello`.

PASS for guarded WebUI smoke:

- frontend returned HTTP 200;
- backend health reported `loaded_now=true` and `generation_ready=true` for the Q5_K_M smoke model;
- frontend matched the existing `llama32_3b_instruct_q8_0` compatibility row;
- frontend detected the quant mismatch (`Q5_K_M` loaded against a Q8_0 support row);
- `contract_supported=false`;
- `expect-webui-chat blocked` passed;
- chat completion was skipped by the WebUI smoke because the active model is blocked by the WebUI chat gate.

## Rerun recipe

Use placeholders rather than local absolute paths in public docs:

```powershell
# Build a Windows camelid.exe with the CLI stack-reserve fix available.
cargo build --bin camelid --quiet --jobs 1

# Stage NVRTC beside the executable. The repo release path uses scripts/package-windows-cuda.ps1
# when a CUDA Toolkit bin directory is available. A local validation run may also stage the
# matching NVIDIA NVRTC redistributable pair beside target/debug/camelid.exe.

# Start the backend with the exact Q5_K_M GGUF.
target\debug\camelid.exe serve --model <path-to-Llama-3.2-3B-Instruct-Q5_K_M.gguf> --no-open

# Capture API-only smoke.
node scripts/model-promotion-smoke-bundle.mjs `
  --api http://127.0.0.1:8181 `
  --model <path-to-Llama-3.2-3B-Instruct-Q5_K_M.gguf> `
  --model-id llama32-3b-q5km-cuda-smoke `
  --out-dir target/q5km-cuda-api-smoke-YYYYMMDD `
  --message hello `
  --max-tokens 1 `
  --temperature 0 `
  --skip-frontend

# Build and serve the frontend in a second terminal.
npm --prefix frontend install
npm --prefix frontend run build
npm --prefix frontend run preview -- --host 127.0.0.1 --port 4175

# Verify the WebUI guard, not WebUI unlock.
node frontend/scripts/smoke.mjs `
  --api http://127.0.0.1:8181 `
  --frontend http://127.0.0.1:4175 `
  --expect-compatibility-row llama32_3b_instruct_q8_0 `
  --expect-compatibility-status supported_exact_row_smoke `
  --expect-contract-supported false `
  --expect-webui-chat blocked
```

## Claim boundary

This is local smoke evidence only. It proves that the exact Q5_K_M GGUF can load and answer API completion/chat requests on a Windows CUDA host when NVRTC is available, and that the frontend support contract still blocks chat because Q5_K_M must not inherit the Q8_0 support row.

It does not add or justify an exact Q5_K_M `/api/capabilities` support row by itself. Support-contract recognition still needs an explicit maintainer decision, scrubbed durable evidence, and synchronized compatibility/status/API/frontend wording before any WebUI unlock or support promotion.