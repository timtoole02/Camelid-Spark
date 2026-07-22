$ErrorActionPreference = 'Continue'
$bin = '$env:USERPROFILE\llama.cpp\build\bin\llama-cli.exe'
$models = '$env:USERPROFILE\Camelid\models'
$here = '$env:USERPROFILE\cam-ornith\qa\ornith\constrained-vram'
$out = Join-Path $here 'residency_16k_part2.txt'
"# 16K residency measurements $(Get-Date -Format o)" | Out-File -FilePath $out -Encoding utf8
foreach ($q in @('Q3_K_M','IQ4_XS')) {
    "== $q ==" | Out-File -FilePath $out -Append -Encoding utf8
    $log = Join-Path $here "res16k_$q.log"
    $p = Start-Process -FilePath $bin -ArgumentList '-m',"$models\ornith-1.0-9b-$q.gguf",'-c','16384','-ngl','99','-p','"What is the capital of France?"','-n','8','--temp','0','--seed','0' -WindowStyle Hidden -RedirectStandardInput (Join-Path $here 'empty_stdin.txt') -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -PassThru
    $peak = 0
    while (-not $p.HasExited) {
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($used -gt $peak) { $peak = $used }
        Start-Sleep -Seconds 1
    }
    $headroom = 6144 - $peak
    "peak_vram_mib=$peak headroom_vs_6144=$headroom exit=$($p.ExitCode)" | Out-File -FilePath $out -Append -Encoding utf8
    $errText = Get-Content ($log + '.err') -ErrorAction SilentlyContinue | Select-String -Pattern 'out of memory|failed to allocate' | Select-Object -First 2
    if ($errText) { $errText | Out-File -FilePath $out -Append -Encoding utf8 }
}
"done" | Out-File -FilePath $out -Append -Encoding utf8
