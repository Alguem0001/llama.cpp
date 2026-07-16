# Dual-build helper: SYCL (oneAPI) for Intel Arc
# Requires: Intel oneAPI Base Toolkit (icx/icpx + DPC++).
# Usage: powershell -File scripts/arc/build-sycl.ps1
param(
    [string]$BuildDir = "build-sycl",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$candidates = @(
    "${env:ONEAPI_ROOT}\setvars.bat",
    "C:\Program Files (x86)\Intel\oneAPI\setvars.bat",
    "C:\Program Files\Intel\oneAPI\setvars.bat"
)
$setvars = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $setvars) {
    Write-Host @"
oneAPI not found.

Install:
  winget install Intel.OneAPI.BaseToolkit

Then open "Intel oneAPI command prompt" or re-run this script.
"@ -ForegroundColor Yellow
    exit 2
}

if ($Clean -and (Test-Path $BuildDir)) { Remove-Item $BuildDir -Recurse -Force }

$bat = @"
@echo off
call "$setvars" --force
cd /d "$Root"
REM Windows: use icx for both (MSVC-like). Linux would use icpx for CXX.
set "VS2022INSTALLDIR=$env:VS2022INSTALLDIR"
cmake -B $BuildDir -G Ninja ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_SYCL=ON ^
  -DGGML_SYCL_F16=ON ^
  -DGGML_NATIVE=ON ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF
if errorlevel 1 exit /b 1
cmake --build $BuildDir -j --target llama-server llama-bench
"@
# Inject VS path for setvars
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
$env:VS2022INSTALLDIR = $vs
$tmp = Join-Path $env:TEMP "build-sycl-llama.bat"
$bat = $bat -replace 'call "\$setvars" --force', "set `"VS2022INSTALLDIR=$vs`"`r`ncall `"$setvars`" --force"
$bat | Set-Content $tmp -Encoding ASCII
cmd /c $tmp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "SYCL build OK: $BuildDir\bin" -ForegroundColor Green
Write-Host "Compare vs Vulkan:"
Write-Host "  build-arc\bin\llama-bench.exe  vs  $BuildDir\bin\llama-bench.exe"
