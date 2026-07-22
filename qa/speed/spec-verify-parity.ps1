<#
.SYNOPSIS
  GPU speculative-verify losslessness gate: spec output must be token-identical to plain greedy.

.DESCRIPTION
  Verifies that the GPU speculative-verify kernels (attention_batched for the linear
  verify_drafts_gpu lane, attention_tree_batched for the CAMELID_SPEC_TREE tree lane) produce
  output bit-identical to single-token attention_decode, so every emitted token is the target's
  own greedy argmax.

  Before the weighted-V G-group fix, the verify kernels reduced the weighted-V with a single
  sequential sum (G=1) while attention_decode splits it into G=ceil(pc/head_dim) reassociated
  groups. Once context exceeded ~head_dim tokens those low bits disagreed and a near-tie token
  could flip (observed: creative_writing diverged from plain greedy at generated token 113 on
  the tree lane). This harness runs the full prompt pack on BOTH verify lanes and asserts the
  intra-Camelid lossless gate (spec stream == this run's own plain greedy stream) on EVERY
  column, including the spec-hostile mandatory-report columns.

  No llama.cpp comparison; the denominator is always this build's own plain greedy decode.
#>
[CmdletBinding()]
param(
  [string]$Bin = '',
  [string]$Model = '$env:USERPROFILE\models\Llama-3.2-3B-Instruct-Q8_0.gguf',
  [string]$PromptsJson = ''
)
$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { (Get-Location).Path }
if (-not $Bin)         { $Bin = Join-Path $ScriptDir '..\..\target\release\camelid.exe' }
if (-not $PromptsJson) { $PromptsJson = Join-Path $ScriptDir 'prompts.json' }
$Bin = (Resolve-Path $Bin).Path
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$utf8 = New-Object System.Text.UTF8Encoding($false)
$pack = Get-Content $PromptsJson -Raw | ConvertFrom-Json
$tmpDir = Join-Path $env:TEMP 'spec_verify_parity'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

# Run one bench on a given verify lane. No native stderr redirect (PS5.1 NativeCommandError trap);
# $out captures stdout (the JSON record) only.
function RunLane($promptFile, $id, $nGen, $lane) {
  if ($lane -eq 'tree') { $env:CAMELID_SPEC_TREE = '1' } else { Remove-Item Env:CAMELID_SPEC_TREE -ErrorAction SilentlyContinue }
  $out = & $Bin bench-speculative $Model --drafter ngram --workload $id --prompt-file $promptFile --max-tokens $nGen --warmup
  Remove-Item Env:CAMELID_SPEC_TREE -ErrorAction SilentlyContinue
  ($out | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1)
}

Write-Host ("[spec-verify-parity] bin={0}" -f $Bin)
Write-Host ("[spec-verify-parity] model={0}  columns={1}" -f (Split-Path $Model -Leaf), $pack.columns.Count)

$rows = @()
foreach ($col in $pack.columns) {
  $id = $col.id; $nGen = [int]$col.n_gen
  $pf = Join-Path $tmpDir ("{0}.txt" -f $id)
  [System.IO.File]::WriteAllText($pf, [string]$col.prompt, $utf8)
  foreach ($lane in @('linear', 'tree')) {
    $line = RunLane $pf $id $nGen $lane
    if (-not $line) { Write-Host ("  {0,-22} {1,-6}: NO JSON" -f $id, $lane) -ForegroundColor Red; continue }
    $r = $line | ConvertFrom-Json
    $div = [int]$r.first_divergent_generated_token_index
    $ok = $r.lossless -and ($div -lt 0)
    Write-Host ("  {0,-22} {1,-6}: {2}  (div_idx={3}, accept={4:P0})" -f $id, $lane, $(if ($ok) { 'LOSSLESS' } else { 'DIVERGE' }), $div, $r.accept_rate) `
      -ForegroundColor $(if ($ok) { 'Green' } else { 'Red' })
    $rows += [pscustomobject]@{ workload = $id; lane = $lane; lossless = [bool]$ok; div_idx = $div }
  }
}

Write-Host ""
$bad = $rows | Where-Object { -not $_.lossless }
if ($bad) {
  Write-Host ("FAIL: {0} (workload, lane) pair(s) diverged from plain greedy:" -f $bad.Count) -ForegroundColor Red
  $bad | ForEach-Object { Write-Host ("  {0} / {1}  @ token {2}" -f $_.workload, $_.lane, $_.div_idx) -ForegroundColor Red }
  exit 1
} else {
  Write-Host ("PASS: GPU spec-verify is token-identical to plain greedy on all {0} (workload x lane) pairs." -f $rows.Count) -ForegroundColor Green
}
