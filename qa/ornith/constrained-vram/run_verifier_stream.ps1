$ErrorActionPreference = 'Continue'
$here = '$env:USERPROFILE\cam-ornith\qa\ornith\constrained-vram'
$env:CAMELID_ORNITH_GGUF = "$env:USERPROFILE/Camelid/models/ornith-1.0-9b-Q8_0.gguf"
$env:CAMELID_STREAM_FILE = Join-Path $here 'stream_tokens.json'
Remove-Item Env:CAMELID_QWEN35_CUDA -ErrorAction SilentlyContinue
$exe = '$env:USERPROFILE\cam-ornith\target\release\deps\camelid-ef6991ec346b41c9.exe'
# 1) Q8_0 verifier argmax stream (CPU int8 path), 12,092 positions.
& $exe --ignored --nocapture --exact runnable::smoke::tests::ornith_qwen35_argmax_stream `
    2> (Join-Path $here 'argmax_verifier.log') > (Join-Path $here 'argmax_verifier.out')
# 2) Q8_0 batched verify cost.
& $exe --ignored --nocapture --exact runnable::smoke::tests::ornith_qwen35_verify_cost `
    2> (Join-Path $here 'verify_cost.log') > (Join-Path $here 'verify_cost.out')
# 3) Q6_K verify cost (generic dequant path — expected slow; the comparison datum).
$env:CAMELID_ORNITH_GGUF = "$env:USERPROFILE/Camelid/models/ornith-1.0-9b-Q6_K.gguf"
& $exe --ignored --nocapture --exact runnable::smoke::tests::ornith_qwen35_verify_cost `
    2> (Join-Path $here 'verify_cost_q6k.log') > (Join-Path $here 'verify_cost_q6k.out')
'DONE' | Out-File (Join-Path $here 'verifier_stage.done') -Encoding utf8
