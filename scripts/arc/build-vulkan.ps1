# Build llama.cpp Vulkan optimized for Intel Arc (Windows)
# Usage: powershell -File scripts/arc/build-vulkan.ps1
param(
    [string]$BuildDir = "build-arc",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item $BuildDir -Recurse -Force
}

$generator = @()
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $generator = @("-G", "Ninja")
}

Write-Host "==> Configure $BuildDir (Vulkan + native CPU)" -ForegroundColor Cyan
cmake -B $BuildDir @generator `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=ON `
    -DGGML_NATIVE=ON `
    -DGGML_BACKEND_DL=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_SERVER=ON

Write-Host "==> Build" -ForegroundColor Cyan
cmake --build $BuildDir --config Release -j

$bin = Join-Path $Root "$BuildDir\bin"
if (-not (Test-Path (Join-Path $bin "llama-server.exe"))) {
    $bin = Join-Path $Root "$BuildDir\bin\Release"
}

Write-Host ""
Write-Host "OK. Binaries:" -ForegroundColor Green
Get-ChildItem $bin -Filter "llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object { "  $($_.FullName)" }
Write-Host ""
Write-Host "Quick list devices:" -ForegroundColor Cyan
Write-Host "  & `"$bin\llama-cli.exe`" --list-devices"
