#requires -version 5
<#
  SPEED_CAMPAIGN.md Phase 2 — matched Camelid baseline (1:1 with Phase 1 llama.cpp).

  SAME model pair / quant / prompts / n_gen / greedy sampling as the llama.cpp baseline,
  so rows are directly comparable. Emits the SAME schema (camelid.speed-receipt/v1,
  engine="camelid") under qa/evidence-bundles/camelid-baseline-<utc>-...

  Lanes:
    raw  : camelid bench-generate, plain greedy GPU-resident decode        (config row 1)
    spec : CAMELID_SPEC_NGRAM=5, lossless n-gram/prompt-lookup SELF-spec    (NOT a config-row-2 match)

  IMPORTANT — spec-lane mechanism: this is n-gram (prompt-lookup) self-speculation, which
  needs zero extra VRAM (no draft model) and so fits the 6 GB card. It only accelerates
  literally-repeated output and is NOT mechanism-comparable to llama.cpp's 0.6B->4B
  draft-MODEL spec (Phase 1 config 5). The matrix's config 2 (serialized draft-model spec)
  and flagship config 3 (concurrent CPU-draft/GPU-verify) are NOT built yet; do not present
  these n-gram rows head-to-head against llama.cpp spec as a win/loss.

  Lossless invariant (D1): the spec lane's token-id stream MUST byte-match Camelid's own
  plain-greedy decode (config 1) for the same prompt. The harness checks this per workload
  and records lossless_match; a mismatch is a correctness failure, not a faster number.
#>
param(
  [string]$Bin      = "$env:USERPROFILE\Camelid\target\release\camelid.exe",
  [string]$Target   = "$env:USERPROFILE\models\Qwen3-4B-Q8_0.gguf",
  [string]$Draft    = "$env:USERPROFILE\camelid-dltest\models\Qwen3-0.6B-Q8_0.gguf",
  [string]$Prompts  = "$env:USERPROFILE\Camelid\qa\speed\prompts.json",
  [int]$Reps        = 5,
  [int]$Ngen        = 128,
  [string]$OutRoot  = "$env:USERPROFILE\Camelid\qa\evidence-bundles",
  [switch]$Probe,                # single GPU-engagement probe, no bundle
  [switch]$RawOnly,
  [switch]$SpecOnly
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Bin)) { throw "missing camelid.exe: $Bin (build with: cargo build --release)" }

function Invoke-Native($exe, [string[]]$argList, [hashtable]$envset) {
  $applied = @{}
  if ($envset) { foreach ($k in $envset.Keys) { $applied[$k] = [Environment]::GetEnvironmentVariable($k); Set-Item -Path "env:$k" -Value $envset[$k] } }
  $o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
  $p = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $o -RedirectStandardError $e
  $stdout = (Get-Content $o -Raw); $stderr = (Get-Content $e -Raw)
  Remove-Item $o,$e -ErrorAction SilentlyContinue
  if ($envset) { foreach ($k in $envset.Keys) { if ($null -eq $applied[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } else { Set-Item -Path "env:$k" -Value $applied[$k] } } }
  [PSCustomObject]@{ stdout = $stdout; stderr = $stderr; exit = $p.ExitCode }
}
function Stat-Median([double[]]$xs){ if($xs.Count -eq 0){return $null}; $s=$xs|Sort-Object; $n=$s.Count; if($n%2-eq1){[math]::Round($s[[int](($n-1)/2)],3)}else{[math]::Round((($s[$n/2-1]+$s[$n/2])/2),3)} }
function Stat-Std([double[]]$xs){ if($xs.Count -lt 2){return 0.0}; $m=($xs|Measure-Object -Average).Average; $v=(($xs|ForEach-Object{($_-$m)*($_-$m)}|Measure-Object -Sum).Sum)/($xs.Count-1); [math]::Round([math]::Sqrt($v),3) }
function Sha256Short($path){ (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() }
function IdsSha($idsArray){ $s = ($idsArray -join ","); $h=(New-Object Security.Cryptography.SHA256Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($s)); ([BitConverter]::ToString($h)).Replace('-','').Substring(0,16) }

# Parse the JSON lines bench-generate writes to stdout (one per measured iteration).
function Parse-Bench($stdout) {
  $recs = @()
  foreach ($line in ($stdout -split "`n")) {
    $t = $line.Trim()
    if ($t.StartsWith("{") -and $t -match "tokens_per_second") {
      try { $recs += ($t | ConvertFrom-Json) } catch {}
    }
  }
  $recs
}

function Get-Env {
  $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.current.sm --format=csv,noheader,nounits 2>$null
  $p = ($smi -split ",").Trim()
  $commit = (& git -C "$env:USERPROFILE\Camelid" rev-parse HEAD).Trim()
  $dirty  = [bool]((& git -C "$env:USERPROFILE\Camelid" status --porcelain) )
  [PSCustomObject]@{
    gpu_name=$p[0]; driver=$p[1]; vram_total_mib=$p[2]; clocks_max_sm_mhz=$p[3]; clocks_cur_sm_mhz=$p[4]
    cpu=(Get-CimInstance Win32_Processor|Select-Object -First 1).Name
    logical_cpus=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    host_ram_gb=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)
    nvcc=((& nvcc --version|Select-String "release").ToString().Trim())
    os="windows " + [Environment]::OSVersion.Version.ToString()
    camelid_commit=$commit; camelid_worktree_dirty=$dirty
    camelid_build="cargo build --release (Windows: cuda cfg on by default, cuda_resident_q8_runtime)"
  }
}

if ($Probe) {
  "== GPU-engagement probe: Qwen3-4B, 64 tok greedy =="
  $r = Invoke-Native $Bin @("bench-generate",$Target,"--prompt","Explain what a CPU cache is in two sentences.","--max-tokens","64","--warmup") @{ CAMELID_LOG="info" }
  $recs = Parse-Bench $r.stdout
  if ($recs.Count -gt 0) { "decode tok/s = {0:N2} | gen {1} tok | ttft {2:N1} ms" -f $recs[0].tokens_per_second, $recs[0].generated_tokens, $recs[0].ttft_ms }
  "--- stderr signal (look for cuda_resident / GPU / HardwareProfile) ---"
  ($r.stderr -split "`n" | Select-String -Pattern "cuda|resident|GPU|CUDA|backend|Hardware|offload" | Select-Object -First 12) -join "`n"
  return
}

$envBlock = Get-Env
$utc = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$bundle = Join-Path $OutRoot "camelid-baseline-qwen3-4b-q8-$utc-head-$($envBlock.camelid_commit.Substring(0,7))"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null
"bundle: $bundle"
$targetSha = Sha256Short $Target
$cells = @()
$baselineIds = @{}   # workload -> ids sha for lossless cross-check

function New-Receipt($lane,$cfg,$workload,$model,$fields){
  $base=[ordered]@{ schema="camelid.speed-receipt/v1"; lane=$lane; config=$cfg; workload=$workload
    generated_utc=$utc; machine=$envBlock; model=$model; engine="camelid"
    camelid_commit=$envBlock.camelid_commit; repetitions=$Reps; warmup_discarded=$true }
  foreach($k in $fields.Keys){ $base[$k]=$fields[$k] }; $base
}

function Run-Lane($col, $lane, $cfg, $envset) {
  $pf = Join-Path $env:TEMP ("camprompt_" + $col.id + ".txt")
  $col.prompt | Out-File -Encoding ascii -NoNewline $pf
  $argv = @("bench-generate",$Target,"--prompt-file",$pf,"--max-tokens","$Ngen","--temperature","0","--iterations","$Reps","--warmup")
  $r = Invoke-Native $Bin $argv $envset
  ($r.stdout + "`n--STDERR--`n" + $r.stderr) | Out-File -Encoding utf8 (Join-Path $bundle ("$lane-" + $col.id + ".log"))
  $recs = Parse-Bench $r.stdout
  $tps = @($recs | ForEach-Object { [double]$_.tokens_per_second } | Where-Object { $_ -gt 0 })
  $ids = if ($recs.Count -gt 0) { $recs[0].output_token_ids } else { @() }
  $idsSha = if ($ids.Count -gt 0) { IdsSha $ids } else { "NA" }
  [PSCustomObject]@{ tps=$tps; idsSha=$idsSha; genTokens=$(if($recs.Count){$recs[0].generated_tokens}else{0}); ttft=$(if($recs.Count){$recs[0].ttft_ms}else{$null}) }
}

$pp = (Get-Content $Prompts -Raw | ConvertFrom-Json)

if (-not $SpecOnly) {
  "== RAW DECODE lane (camelid bench-generate, plain greedy, GPU-resident) =="
  foreach ($col in $pp.columns) {
    "  raw[$($col.id)]"
    $res = Run-Lane $col "raw" "camelid_normal_decode" $null
    $baselineIds[$col.id] = $res.idsSha
    $cells += New-Receipt "raw" "camelid_normal_decode" $col.id `
      ([ordered]@{ id="qwen3-4b"; path=$Target; sha256=$targetSha; quant="Q8_0"; arch="qwen3" }) `
      ([ordered]@{ n_gen=$Ngen; decode_tps_median=(Stat-Median $res.tps); decode_tps_stddev=(Stat-Std $res.tps)
        decode_tps_samples=$res.tps; generated_tokens=$res.genTokens; ttft_ms=$res.ttft
        output_ids_sha256=$res.idsSha
        sampling="greedy (temp 0) - Camelid plain greedy; its OWN non-spec reference (D1)"
        gpu_clock_at_run_mhz=(Get-Env).clocks_cur_sm_mhz })
  }
}
if (-not $RawOnly) {
  "== SPEC DECODE lane (n-gram/prompt-lookup SELF-spec, lossless; NOT draft-model spec) =="
  foreach ($col in $pp.columns) {
    "  spec[$($col.id)]"
    # bench-generate reads CAMELID_SPEC_NGRAM (main.rs); the GPU batched verify is automatic
    # when the model is GPU-resident. CAMELID_SPEC_GPU is an API-server-only flag (api/mod.rs)
    # and a no-op here, so it is deliberately NOT set.
    $res = Run-Lane $col "spec" "camelid_ngram_self_spec" @{ CAMELID_SPEC_NGRAM="5" }
    $lossless = if ($baselineIds.ContainsKey($col.id)) { $res.idsSha -eq $baselineIds[$col.id] } else { $null }
    $cells += New-Receipt "spec" "camelid_ngram_self_spec" $col.id `
      ([ordered]@{ id="qwen3-4b"; path=$Target; sha256=$targetSha; quant="Q8_0"; arch="qwen3" }) `
      ([ordered]@{ n_gen=$Ngen; decode_tps_median=(Stat-Median $res.tps); decode_tps_stddev=(Stat-Std $res.tps)
        decode_tps_samples=$res.tps; generated_tokens=$res.genTokens
        output_ids_sha256=$res.idsSha; lossless_baseline_sha256=($baselineIds[$col.id])
        lossless_match=$lossless
        spec_mechanism="ngram_prompt_lookup_self_spec (no draft model; CAMELID_SPEC_NGRAM)"
        comparable_to_llamacpp_spec=$false
        comparability_note="n-gram self-spec only accelerates literally-repeated output; it is NOT mechanism-comparable to llama.cpp's 0.6B->4B DRAFT-MODEL spec (Phase 1 config 5). Do not table the decode_tps head-to-head as a spec-vs-spec win/loss. The campaign's draft-model lanes are config 2 (serialized) and the flagship config 3 (concurrent), neither built yet."
        sampling="greedy (temp 0) GPU n-gram self-spec - MUST byte-match config-1 baseline ids (D1)"
        gpu_clock_at_run_mhz=(Get-Env).clocks_cur_sm_mhz })
  }
}

$manifest=[ordered]@{ schema="camelid.speed-baseline-bundle/v1"
  campaign="SPEED_CAMPAIGN.md Phase 2 - Camelid matched baseline"
  generated_utc=$utc; engine="camelid"; camelid_commit=$envBlock.camelid_commit
  model_pair="qwen3-4b-q8 (target; NO draft model - n-gram/prompt-lookup self-spec)"
  matched_to="Phase 1 llama.cpp baseline ONLY on the RAW lane (same model/quant/prompts/n_gen/greedy). The spec lane is n-gram self-spec and is NOT mechanism-matched to llama.cpp's draft-model spec."
  machine=$envBlock; cell_count=$cells.Count; cells=$cells }
$mp = Join-Path $bundle "manifest.json"
$manifest | ConvertTo-Json -Depth 12 | Out-File -Encoding utf8 $mp
""; "wrote $($cells.Count) cells -> $mp"
$bad = $cells | Where-Object { $_.lane -eq "spec" -and $_.lossless_match -eq $false }
if ($bad) { "WARNING: $($bad.Count) spec cell(s) FAILED the lossless check - investigate before counting as baseline" }
