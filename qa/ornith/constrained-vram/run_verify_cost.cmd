@echo off
set HERE=%USERPROFILE%\cam-ornith\qa\ornith\constrained-vram
set CAMELID_STREAM_FILE=%HERE%\stream_tokens.json
set CAMELID_VERIFY_PREFIX=120
set EXE=%USERPROFILE%\cam-ornith\target\release\deps\camelid-ef6991ec346b41c9.exe

set CAMELID_ORNITH_GGUF=%USERPROFILE%/Camelid/models/ornith-1.0-9b-Q8_0.gguf
"%EXE%" --ignored --nocapture --exact runnable::smoke::tests::ornith_qwen35_verify_cost > "%HERE%\verify_cost.out" 2> "%HERE%\verify_cost.log"

set CAMELID_ORNITH_GGUF=%USERPROFILE%/Camelid/models/ornith-1.0-9b-Q6_K.gguf
"%EXE%" --ignored --nocapture --exact runnable::smoke::tests::ornith_qwen35_verify_cost > "%HERE%\verify_cost_q6k.out" 2> "%HERE%\verify_cost_q6k.log"

echo done > "%HERE%\verify_cost.done"
