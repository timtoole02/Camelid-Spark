param(
  [string]$Bin = ".\target\release\camelid.exe",
  [string]$Label = "run",
  [int]$Threads = 8,
  [int]$MaxTokens = 48
)
# Canonical CPU parity+speed harness. Reference sha1 (scalar baseline) = B3ECD15506DA.
$m = (Resolve-Path ".\models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf").Path
$bin = (Resolve-Path $Bin).Path
$prompt = "Once upon a time, in a small village by the sea, there lived a young fisherman who"
$log = "$env:TEMP\cpubench_$Label.txt"
$threadArg = if ($Threads -gt 0) { ' --threads ' + $Threads } else { '' }  # 0 = use binary default pool
$argline = 'bench-generate "' + $m + '" --prompt "' + $prompt + '" --max-tokens ' + $MaxTokens + ' --deterministic' + $threadArg
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $bin -ArgumentList $argline -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$p.WaitForExit(); $sw.Stop()
$cpu = $p.TotalProcessorTime.TotalSeconds; $wall = $sw.Elapsed.TotalSeconds
$j = (Get-Content $log | Where-Object { $_ -match 'tokens_per_second' }) -join ''
$tps = if ($j -match '"tokens_per_second":([0-9.]+)') { [math]::Round([double]$matches[1], 2) } else { 'NA' }
$ids = if ($j -match '"output_token_ids":\[([0-9,]+)\]') { $matches[1] } else { 'NA' }
$sha = (New-Object Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($ids))
$h = ([BitConverter]::ToString($sha)).Replace('-', '').Substring(0, 12)
$cores = if ($wall -gt 0) { [math]::Round($cpu / $wall, 2) } else { 0 }
"{0,-24} tok/s={1,-7} cores={2,-5} ids_sha1={3}" -f $Label, $tps, $cores, $h
