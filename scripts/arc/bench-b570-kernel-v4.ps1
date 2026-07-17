# bench-b570-kernel-v4.ps1 — bench do kernel v4 (único perfil no tree)
# Uso:
#   powershell -File scripts/arc/bench-b570-kernel-v4.ps1
#   powershell -File scripts/arc/bench-b570-kernel-v4.ps1 -Phase matrix
param(
    [ValidateSet("bench","matrix")]
    [string]$Phase = "bench",
    [string]$Bin     = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp\build-arc\bin",
    [string]$Model17 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
    [string]$Model27 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Bonsai-27B-Q1_0.gguf",
    [string]$OutDir  = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp\benches\arc-b570",
    [int]$Reps = 5
)

$bench = Join-Path $Bin "llama-bench.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log   = Join-Path $OutDir "kernel-v4-$Phase-$stamp.txt"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# v4 is always on for Xe2 — clear any legacy env leftovers
Remove-Item Env:GGML_VK_B570_KERNEL -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_B570_MMVQ_VEC -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_B570_FUSE_RMSNORM_MUL -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_B570_MMVQ_XL -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_ARC_FA_LEGACY -ErrorAction SilentlyContinue

function Run-Bench($model, $tag, $reps, $extra) {
    Write-Host "== $tag ==" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -fa on -p 512 -n 128 -r $reps -b 512 -ub 256 @extra 2>&1 |
        Tee-Object -FilePath $log -Append
}

switch ($Phase) {
"bench" {
    Run-Bench $Model17 "1.7B v4" $Reps @()
    Run-Bench $Model27 "27B  v4" $Reps @()
}
"matrix" {
    # Runtime knobs only (no kernel-version A/B — v4 is the only profile)
    foreach ($gq in @("0","1")) {
        foreach ($poll in @("0","25","50","100")) {
            foreach ($ub in @("64","128","256","512")) {
                if ($gq -eq "1") { $env:GGML_VK_ALLOW_GRAPHICS_QUEUE = "1" }
                else { Remove-Item Env:GGML_VK_ALLOW_GRAPHICS_QUEUE -ErrorAction SilentlyContinue }
                Run-Bench $Model27 "27B gq=$gq poll=$poll ub=$ub" 3 @("--poll",$poll,"-ub",$ub)
            }
        }
    }
}
}

Write-Host "`nLog: $log" -ForegroundColor Green
