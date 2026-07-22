#requires -version 5
<#
  SPEED_CAMPAIGN.md Phase 1 — honest llama.cpp baseline (best flags, not weak defaults).

  Emits camelid.speed-receipt/v1 cells under qa/evidence-bundles/llamacpp-baseline-<utc>-...
  Two lanes:
    raw  : llama-bench  -fa 1, device pinned, >=N repetitions, median +/- stddev   (config row 4)
    spec : llama-speculative  target + draft, llama.cpp's own lossless spec decode  (config row 5)

  Every number dereferences to a receipt; the receipt carries machine/gpu/clocks/commit/flags/cmd
  so a hostile reviewer with the receipt + the two commits can re-run and land within stddev.

  FLAG ASSUMPTIONS for the pinned commit live in the $LB / $SP hashtables below — validated against
  `llama-bench --help` / `llama-speculative --help` before the first real run (see -Validate).
#>
param(
  [string]$BinDir   = "$env:USERPROFILE\llama.cpp\build\bin",
  [string]$Target   = "$env:USERPROFILE\models\Qwen3-4B-Q8_0.gguf",
  [string]$Draft    = "$env:USERPROFILE\camelid-dltest\models\Qwen3-0.6B-Q8_0.gguf",
  [string]$Prompts  = "$env:USERPROFILE\Camelid\qa\speed\prompts.json",
  [int]$Reps        = 5,         # llama-bench repetitions (after warmup); raise for tighter stddev
  [int]$DraftMax    = 8,         # matches Camelid MAX_VERIFY_K = 8
  [int]$DraftMin    = 1,
  [int]$NgenSpec    = 128,
  [string]$OutRoot  = "$env:USERPROFILE\Camelid\qa\evidence-bundles",
  [int]$PinClock    = 0,         # if >0, attempt nvidia-smi -lgc <PinClock> (needs admin); 0 = leave clocks free
  [switch]$Validate,             # only print the binaries' --help + env, do not benchmark
  [switch]$RawOnly,
  [switch]$SpecOnly
)
$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"   # pin the device for every child process

$llamaBench = Join-Path $BinDir "llama-bench.exe"
$llamaSpec  = Join-Path $BinDir "llama-speculative.exe"
foreach ($b in @($llamaBench,$llamaSpec)) { if (-not (Test-Path $b)) { throw "missing binary: $b (build llama.cpp first)" } }

# ---- environment capture (once) -------------------------------------------------
function Get-Env {
  $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.current.sm,power.limit --format=csv,noheader,nounits 2>$null
  $p = ($smi -split ",").Trim()
  $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
  $logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
  $ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  $nvcc = (& nvcc --version | Select-String "release").ToString().Trim()
  $commit = (& git -C "$env:USERPROFILE\llama.cpp" rev-parse HEAD).Trim()
  # Brief Phase 1: record the matmul backend state so a reviewer knows which CUDA GEMM
  # path was exercised. These env vars force MMQ (quantized int8 tensor-core kernels) or
  # cuBLAS; unset means llama.cpp's heuristic (the documented default) chose at init.
  $forceMmq    = if ($null -ne $env:GGML_CUDA_FORCE_MMQ)    { $env:GGML_CUDA_FORCE_MMQ }    else { "unset (default heuristic)" }
  $forceCublas = if ($null -ne $env:GGML_CUDA_FORCE_CUBLAS) { $env:GGML_CUDA_FORCE_CUBLAS } else { "unset (default heuristic)" }
  [PSCustomObject]@{
    gpu_name = $p[0]; driver = $p[1]; vram_total_mib = $p[2]
    clocks_max_sm_mhz = $p[3]; clocks_cur_sm_mhz = $p[4]; power_limit_w = $p[5]
    cpu = $cpu; logical_cpus = $logical; host_ram_gb = $ramGb
    nvcc = $nvcc; os = "windows " + [System.Environment]::OSVersion.Version.ToString()
    sm_clock_policy = $script:clockPolicy
    ggml_cuda_force_mmq = $forceMmq; ggml_cuda_force_cublas = $forceCublas
    llamacpp_commit = $commit
    llamacpp_build_flags = "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release (Ninja, VS2022 BuildTools 14.44, CUDA 12.9)"
  }
}

function Sha256Short($path) {
  (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

# Current SM clock as an int (MHz), admin-free. Used to BRACKET each timed cell
# (sample immediately before + after the child) so the receipt records the real
# clock band the work ran under instead of a misleading post-spindown snapshot.
function Get-SmClock {
  $v = & nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>$null | Select-Object -First 1
  if ($v -match '(\d+)') { [int]$matches[1] } else { $null }
}

# Attempt to lock the graphics clock (reproducibility hardening). Returns a policy
# string for the receipt. Locking needs admin on Windows; on failure we record that
# honestly and fall back to free-boost + per-cell bracketing (the campaign forbids
# pretending a knob was set when it was not).
$script:clockPinned = $false
function Set-ClockPin([int]$mhz) {
  if ($mhz -le 0) { return "free (laptop boost; per-cell pre/post clock bracket recorded)" }
  $r = Invoke-Native "nvidia-smi" @("-lgc","$mhz,$mhz")
  if ($r.exit -eq 0) {
    $script:clockPinned = $true
    return "pinned $mhz MHz (nvidia-smi -lgc), bracket-verified per cell"
  }
  $why = ($r.stdout + $r.stderr) -replace "\s+"," " | ForEach-Object { $_.Trim() }
  Write-Warning "clock pin to $mhz MHz FAILED (need admin?): $why -- falling back to free boost"
  return "PIN_ATTEMPTED_FAILED ($mhz MHz; not elevated?) -> free boost; per-cell bracket recorded"
}

# Native-exe runner. PS 5.1 turns a native command's stderr into NativeCommandError
# records (and $?=$false) under -ErrorAction Stop even on exit 0, so we never redirect
# native stderr in-process; we run via Start-Process to files. No arg contains spaces.
function Invoke-Native($exe, [string[]]$argList) {
  $o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
  $p = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $o -RedirectStandardError $e
  $stdout = (Get-Content $o -Raw); $stderr = (Get-Content $e -Raw)
  Remove-Item $o,$e -ErrorAction SilentlyContinue
  [PSCustomObject]@{ stdout = $stdout; stderr = $stderr; exit = $p.ExitCode }
}

function Stat-Median([double[]]$xs) {
  if ($xs.Count -eq 0) { return $null }
  $s = $xs | Sort-Object
  $n = $s.Count
  if ($n % 2 -eq 1) { return [math]::Round($s[[int](($n-1)/2)],3) }
  return [math]::Round((($s[$n/2-1] + $s[$n/2]) / 2),3)
}
function Stat-Std([double[]]$xs) {
  if ($xs.Count -lt 2) { return 0.0 }
  $m = ($xs | Measure-Object -Average).Average
  $v = (($xs | ForEach-Object { ($_ - $m) * ($_ - $m) } | Measure-Object -Sum).Sum) / ($xs.Count - 1)
  [math]::Round([math]::Sqrt($v),3)
}

if ($Validate) {
  "==== llama-bench --help ===="; & $llamaBench --help 2>&1 | Select-Object -First 60
  ""; "==== llama-speculative --help ===="; & $llamaSpec --help 2>&1 | Select-Object -First 60
  ""; "==== environment ===="; Get-Env | Format-List
  return
}

$script:clockPolicy = Set-ClockPin $PinClock
"clock policy: $script:clockPolicy"
$envBlock = Get-Env
$utc = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$shortCommit = $envBlock.llamacpp_commit.Substring(0,7)
$bundle = Join-Path $OutRoot "llamacpp-baseline-qwen3-0.6b-4b-q8-$utc-llamacpp-$shortCommit"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null
"bundle: $bundle"

$targetSha = Sha256Short $Target
$draftSha  = Sha256Short $Draft
$cells = @()

function New-Receipt($lane, $cfg, $workload, $model, $fields) {
  $base = [ordered]@{
    schema = "camelid.speed-receipt/v1"
    lane = $lane; config = $cfg; workload = $workload
    generated_utc = $utc
    machine = $envBlock
    model = $model
    engine = "llama.cpp"
    llamacpp_commit = $envBlock.llamacpp_commit
    repetitions = $Reps
    warmup_discarded = $true
  }
  foreach ($k in $fields.Keys) { $base[$k] = $fields[$k] }
  $base
}

# ---- RAW DECODE lane (llama-bench, -fa 1) ---------------------------------------
# pp/tg are token-count driven and content-independent, so one sweep per model covers
# the raw lane; per-workload content variation only matters for the spec lane below.
function Run-Raw($name, $path, $sha) {
  $pp = 512; $tg = $NgenSpec
  $jsonOut = Join-Path $bundle "rawbench-$name.json"
  $argv = @("-m",$path,"-p","$pp","-n","$tg","-fa","1","-ngl","99","-r","$Reps","-o","json")
  "  raw: $llamaBench $($argv -join ' ')"
  $clkPre = Get-SmClock
  $raw = (Invoke-Native $llamaBench $argv).stdout
  $clkPost = Get-SmClock
  $raw | Out-File -Encoding utf8 $jsonOut
  $parsed = $raw | ConvertFrom-Json
  foreach ($row in $parsed) {
    $kind = if ($row.n_prompt -gt 0 -and $row.n_gen -eq 0) { "prefill_pp" } else { "decode_tg" }
    $rec = New-Receipt "raw" "llamacpp_normal_decode" $kind `
      ([ordered]@{ id=$name; path=$path; sha256=$sha; quant="Q8_0"; arch="qwen3" }) `
      ([ordered]@{
        n_prompt = $row.n_prompt; n_gen = $row.n_gen
        avg_tps = [math]::Round($row.avg_ts,3); stddev_tps = [math]::Round($row.stddev_ts,3)
        samples_tps = $row.samples_ts
        flash_attn = $row.flash_attn; n_gpu_layers = $row.n_gpu_layers
        gpu_clock_bracket_mhz = @($clkPre, $clkPost)   # sampled pre/post the timed child (admin-free drift evidence)
        sm_clock_policy = $script:clockPolicy
        cmd = "$llamaBench " + ($argv -join ' ')
        lossless_note = "llama.cpp greedy raw decode; its own non-spec reference (D1)."
      })
    $script:cells += $rec
  }
}

# ---- SPEC DECODE lane (llama-speculative) ---------------------------------------
# Content-dependent (acceptance rate varies by workload) -> run per matrix column.
# llama-speculative has no -r flag, so we loop: 1 warmup (discarded) + $Reps timed.
# Parser matches this pinned commit's output:
#   "decoded N tokens in X seconds, speed: Y t/s"   <- decode throughput (reliable)
#   "n_drafted = N" / "n_accept = N" / "accept = P%"
# (the common_perf_print "total time" is the known-broken negative-clock artifact on
#  Windows and is deliberately NOT used.)
function Parse-Spec($txt) {
  $dec = if ($txt -match "decoded\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds, speed:\s+([0-9.]+)\s+t/s") {
    [PSCustomObject]@{ tps=[double]$matches[3]; n=[int]$matches[1]; sec=[double]$matches[2] } } else { $null }
  $enc = if ($txt -match "encoded\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds, speed:\s+([0-9.]+)\s+t/s") {
    [double]$matches[3] } else { $null }
  [PSCustomObject]@{
    decode_tps = if ($dec) { $dec.tps } else { $null }
    decoded_n  = if ($dec) { $dec.n } else { $null }
    prefill_tps = $enc
    n_drafted = if ($txt -match "n_drafted\s*=\s*([0-9]+)") { [int]$matches[1] } else { $null }
    n_accept  = if ($txt -match "n_accept\s*=\s*([0-9]+)")  { [int]$matches[1] } else { $null }
    accept_pct = if ($txt -match "accept\s*=\s*([0-9.]+)%") { [double]$matches[1] } else { $null }
  }
}
# One llama-speculative invocation for a column. Returns parsed stats + clock bracket
# + raw log. The rep/interleave loop lives in the main section so columns can be
# round-robined across reps (thermal-position fair) instead of run consecutively.
function Invoke-SpecOnce($col) {
  $promptFile = Join-Path $env:TEMP ("specprompt_" + $col.id + ".txt")
  $col.prompt | Out-File -Encoding ascii -NoNewline $promptFile
  $argv = @("-m",$Target,"-md",$Draft,"-f",$promptFile,"-n","$NgenSpec",
            "--spec-draft-n-max","$DraftMax","--spec-draft-n-min","$DraftMin",
            "-ngl","99","-ngld","99","--top-k","1","--temp","0","-c","2048")
  $clkPre = Get-SmClock
  $res = Invoke-Native $llamaSpec $argv
  $clkPost = Get-SmClock
  $combined = "$($res.stdout)`n$($res.stderr)"   # stats print on stderr (info log)
  [PSCustomObject]@{ parsed = (Parse-Spec $combined); combined = $combined; argv = $argv; clkPre = $clkPre; clkPost = $clkPost }
}

$pp = (Get-Content $Prompts -Raw | ConvertFrom-Json)

try {
  if (-not $SpecOnly) {
    "== RAW DECODE lane (llama-bench -fa 1) =="
    Run-Raw "qwen3-4b"   $Target $targetSha
    Run-Raw "qwen3-0.6b" $Draft  $draftSha
  }
  if (-not $RawOnly) {
    "== SPEC DECODE lane (llama-speculative, target+draft) =="
    "   INTERLEAVED column order across reps (each column sampled across the whole thermal"
    "   timeline, so drift can't bias whichever column runs last); warmup pass discarded."
    $cols = @($pp.columns)
    # warmup: one untimed pass over every column to reach GPU steady state (discarded)
    foreach ($col in $cols) { [void](Invoke-SpecOnce $col) }
    # accumulate per-column samples across $Reps interleaved (round-robin) passes
    $acc = [ordered]@{}
    foreach ($col in $cols) { $acc[$col.id] = [ordered]@{ samples=@(); clocks=@(); first=$null; argv=$null } }
    for ($rep = 1; $rep -le $Reps; $rep++) {
      foreach ($col in $cols) {
        $r = Invoke-SpecOnce $col
        $a = $acc[$col.id]
        $a.argv = $r.argv
        if ($r.parsed.decode_tps) { $a.samples += $r.parsed.decode_tps }
        $a.clocks += @($r.clkPre, $r.clkPost)
        if ($rep -eq 1) {
          $a.first = $r.parsed
          $r.combined | Out-File -Encoding utf8 (Join-Path $bundle ("specbench-" + $col.id + ".log"))
        }
      }
      "  interleaved rep $rep/$Reps complete"
    }
    foreach ($col in $cols) {
      $a = $acc[$col.id]
      $clk = @($a.clocks | Where-Object { $null -ne $_ })
      $rec = New-Receipt "spec" "llamacpp_speculative_decode" $col.id `
        ([ordered]@{
          target = [ordered]@{ id="qwen3-4b"; path=$Target; sha256=$targetSha; quant="Q8_0" }
          draft  = [ordered]@{ id="qwen3-0.6b"; path=$Draft; sha256=$draftSha; quant="Q8_0" }
        }) `
        ([ordered]@{
          n_gen = $NgenSpec; spec_draft_n_max = $DraftMax; spec_draft_n_min = $DraftMin
          accept_rate_pct = $a.first.accept_pct; n_drafted = $a.first.n_drafted; n_accept = $a.first.n_accept
          decode_tps_median = (Stat-Median $a.samples); decode_tps_stddev = (Stat-Std $a.samples)
          decode_tps_samples = $a.samples
          prefill_tps = $a.first.prefill_tps
          sampling = "greedy (top-k 1, temp 0) - lossless vs llama.cpp own greedy (D1)"
          sampling_order = "interleaved across $Reps reps (thermal-position fair)"
          gpu_clock_bracket_mhz = @( ($clk | Measure-Object -Minimum).Minimum, ($clk | Measure-Object -Maximum).Maximum )
          sm_clock_policy = $script:clockPolicy
          cmd = "$llamaSpec " + ($a.argv -join ' ')
          raw_stats_log = ("specbench-" + $col.id + ".log")
        })
      $script:cells += $rec
    }
  }
}
finally {
  if ($script:clockPinned) { [void](Invoke-Native "nvidia-smi" @("-rgc")); "clocks reset (nvidia-smi -rgc)" }
}

$manifest = [ordered]@{
  schema = "camelid.speed-baseline-bundle/v1"
  campaign = "SPEED_CAMPAIGN.md Phase 1 - llama.cpp honest baseline"
  generated_utc = $utc
  engine = "llama.cpp"
  llamacpp_commit = $envBlock.llamacpp_commit
  model_pair = "qwen3-0.6b-q8 (draft) -> qwen3-4b-q8 (target)"
  machine = $envBlock
  cell_count = $cells.Count
  cells = $cells
}
$manifestPath = Join-Path $bundle "manifest.json"
$manifest | ConvertTo-Json -Depth 12 | Out-File -Encoding utf8 $manifestPath
""
"wrote $($cells.Count) cells -> $manifestPath"
