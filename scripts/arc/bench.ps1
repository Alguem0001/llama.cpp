# A/B llama-bench for Intel Arc
# Usage:
#   powershell -File scripts/arc/bench.ps1 -Model path\to\model.gguf
#   powershell -File scripts/arc/bench.ps1 -Model ... -LegacyFa
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$BinDir = "",
    [int]$Prompt = 512,
    [int]$Gen = 128,
    [int]$Reps = 3,
    [switch]$LegacyFa,
    [switch]$NoFlashAttn
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if (-not $BinDir) {
    foreach ($c in @(
        (Join-Path $Root "build-arc\bin"),
        (Join-Path $Root "build-arc\bin\Release"),
        (Join-Path $Root "build\bin"),
        (Join-Path $Root "build\bin\Release")
    )) {
        if (Test-Path (Join-Path $c "llama-bench.exe")) { $BinDir = $c; break }
    }
}
if (-not $BinDir) { throw "llama-bench.exe not found. Run scripts/arc/build-vulkan.ps1 first." }

$bench = Join-Path $BinDir "llama-bench.exe"
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

if ($LegacyFa) {
    $env:GGML_VK_ARC_FA_LEGACY = "1"
    Write-Host "GGML_VK_ARC_FA_LEGACY=1 (upstream Intel FA)" -ForegroundColor Yellow
} else {
    Remove-Item Env:GGML_VK_ARC_FA_LEGACY -ErrorAction SilentlyContinue
    Write-Host "Xe2 FA experiment active (default on this branch)" -ForegroundColor Cyan
}

$fa = if ($NoFlashAttn) { "off" } else { "on" }
$args = @(
    "-m", $Model,
    "-ngl", "99",
    "-fa", $fa,
    "-p", "$Prompt",
    "-n", "$Gen",
    "-r", "$Reps"
)

Write-Host "Running: $bench $($args -join ' ')" -ForegroundColor Cyan
& $bench @args
