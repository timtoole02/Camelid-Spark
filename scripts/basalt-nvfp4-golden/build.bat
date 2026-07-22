@echo off
rem BASALT Phase 1 harness build — pin llama.cpp acd79d603, route linked-libs
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /O2 /std:c11 /MD /DGGML_SHARED /I <llama.cpp>\ggml\include /I <llama.cpp>\ggml\src nvfp4_fixture_gen.c /link /LIBPATH:<llama.cpp>\build\ggml\src ggml-base.lib
if errorlevel 1 exit /b 1
cl 2>&1 | findstr /i version
