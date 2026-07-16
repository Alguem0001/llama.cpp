# Arc-tuned llama-server preset
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Mmproj = "",
    [string]$BinDir = "",
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [int]$Ngl = 99,
    [switch]$Tools,
    [switch]$LegacyFa
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
        if (Test-Path (Join-Path $c "llama-server.exe")) { $BinDir = $c; break }
    }
}
if (-not $BinDir) { throw "llama-server.exe not found" }

if ($LegacyFa) { $env:GGML_VK_ARC_FA_LEGACY = "1" }
else { Remove-Item Env:GGML_VK_ARC_FA_LEGACY -ErrorAction SilentlyContinue }

# Persistent shader cache helps after first run on Windows
$env:GGML_VK_FORCE_MAX_ALLOCATION_SIZE = $env:GGML_VK_FORCE_MAX_ALLOCATION_SIZE

$exe = Join-Path $BinDir "llama-server.exe"
$args = @(
    "-m", $Model,
    "-ngl", "$Ngl",
    "-c", "$Ctx",
    "-fa", "on",
    "-b", "512",
    "-ub", "256",
    "--host", $HostAddr,
    "--port", "$Port",
    "--jinja"
)
if ($Tools) { $args += @("--tools", "all") }
if ($Mmproj -and (Test-Path $Mmproj)) { $args += @("--mmproj", $Mmproj) }

Write-Host "cwd=$BinDir" -ForegroundColor DarkGray
Write-Host "& $exe $($args -join ' ')" -ForegroundColor Cyan
Set-Location $BinDir
& $exe @args
