#requires -version 5
<#
  Render a camelid.speed-baseline-bundle/v1 manifest into a human-readable markdown
  matrix + README. Engine-agnostic: works on the llama.cpp Phase 1 bundle and (once it
  emits the same schema) the Camelid Phase 2 bundle. Reads only the receipt; writes no
  numbers of its own (every figure dereferences to a cell).
#>
param(
  [Parameter(Mandatory)][string]$Bundle
)
$ErrorActionPreference = "Stop"
$m = Get-Content (Join-Path $Bundle "manifest.json") -Raw | ConvertFrom-Json
$mc = $m.machine

$raw  = $m.cells | Where-Object lane -eq "raw"
$spec = $m.cells | Where-Object lane -eq "spec"

$sb = New-Object System.Text.StringBuilder
function W($s){ [void]$sb.AppendLine($s) }

W "# $($m.campaign)"
W ""
W "Engine: **$($m.engine)** @ commit ``$($m.llamacpp_commit)``  "
W "Model pair: $($m.model_pair)  "
W "Generated (UTC): $($m.generated_utc)  "
W "Cells: $($m.cell_count)"
W ""
W "## Machine"
W ""
W "| field | value |"
W "|---|---|"
W "| GPU | $($mc.gpu_name) |"
W "| Driver | $($mc.driver) |"
W "| VRAM | $($mc.vram_total_mib) MiB |"
W "| SM clock (max / idle snapshot) | $($mc.clocks_max_sm_mhz) / $($mc.clocks_cur_sm_mhz) MHz |"
W "| SM clock policy | $($mc.sm_clock_policy) |"
W "| CPU | $($mc.cpu) ($($mc.logical_cpus) logical) |"
W "| Host RAM | $($mc.host_ram_gb) GB |"
W "| CUDA | $($mc.nvcc) |"
W "| OS | $($mc.os) |"
W "| llama.cpp build | $($mc.llamacpp_build_flags) |"
W ""
W "## Raw decode lane (llama-bench, -fa 1)"
W ""
W "| model | prefill pp (t/s) | decode tg (t/s) |"
W "|---|---|---|"
$byModel = $raw | Group-Object { $_.model.id }
foreach ($g in $byModel) {
  $pp = $g.Group | Where-Object workload -eq "prefill_pp" | Select-Object -First 1
  $tg = $g.Group | Where-Object workload -eq "decode_tg"  | Select-Object -First 1
  $ppS = if ($pp) { "{0} +/- {1}  (n_prompt={2})" -f $pp.avg_tps, $pp.stddev_tps, $pp.n_prompt } else { "-" }
  $tgS = if ($tg) { "{0} +/- {1}  (n_gen={2})"    -f $tg.avg_tps, $tg.stddev_tps, $tg.n_gen } else { "-" }
  W "| $($g.Name) | $ppS | $tgS |"
}
W ""
W "## Speculative decode lane (llama-speculative, target+draft, greedy/lossless)"
W ""
W "| workload | decode t/s (median +/- sd) | accept % | n_drafted | n_accept |"
W "|---|---|---|---|---|"
foreach ($c in $spec) {
  $tps = if ($null -ne $c.decode_tps_median) { "{0} +/- {1}" -f $c.decode_tps_median, $c.decode_tps_stddev } else { "PARSE_FAIL" }
  W "| $($c.workload) | $tps | $($c.accept_rate_pct) | $($c.n_drafted) | $($c.n_accept) |"
}
W ""
# spec vs raw-target speedup (both lossless greedy) -- the honest headline
$tgTarget = ($raw | Where-Object { $_.model.id -like "*4b*" -and $_.workload -eq "decode_tg" } | Select-Object -First 1).avg_tps
if ($tgTarget) {
  W "## llama.cpp spec vs llama.cpp raw-target decode (intra-engine speedup)"
  W ""
  W "Target raw decode baseline: **$tgTarget t/s**. This is llama.cpp-vs-llama.cpp only;"
  W "the cross-engine comparison against Camelid is filled in once Phase 2 lands."
  W ""
  W "| workload | spec t/s | raw-target t/s | spec/raw |"
  W "|---|---|---|---|"
  foreach ($c in $spec) {
    $r = if ($c.decode_tps_median -and $tgTarget) { [math]::Round($c.decode_tps_median / $tgTarget, 2).ToString() + "x" } else { "-" }
    W "| $($c.workload) | $($c.decode_tps_median) | $tgTarget | $r |"
  }
  W ""
}
W "## Reproduce"
W ""
W '```'
W "git -C $env:USERPROFILE\llama.cpp checkout $($m.llamacpp_commit)"
W "# rebuild (Ninja + CUDA arch 86), then:"
W "pwsh qa/speed/llamacpp-baseline.ps1 -Reps $($m.cells[0].repetitions)"
W '```'
W ""
W "**Caveats.** SM clock policy: ``$($mc.sm_clock_policy)``. On this laptop, boost/thermal drift"
W "is real, so the spec lane runs its columns **interleaved** (round-robin across reps), letting each"
W "column sample the whole thermal timeline instead of penalizing whichever runs last; each cell also"
W "records a pre/post ``gpu_clock_bracket_mhz``. Warmup discarded; median +/- stddev over"
W "$($m.cells[0].repetitions) reps. Per D1/D2 (SPEED_CAMPAIGN.md): both lanes are lossless greedy"
W "(each matches its OWN engine's non-spec greedy); this is a SPEED comparison at matched settings,"
W "not a claim that llama.cpp and Camelid emit identical token streams."

$out = Join-Path $Bundle "README.md"
$sb.ToString() | Out-File -Encoding utf8 $out
Write-Host "wrote $out"
Write-Host ""
$sb.ToString()
