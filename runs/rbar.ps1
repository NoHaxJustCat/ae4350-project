# Fresh R-bar run. Network/device/optimizer come from libs/constants.py.

param(
    [string]$RunTag = "rbar_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 500000,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.35
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "rbar | run-tag=$RunTag | steps=$TotalTimesteps | noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

& ./.conda/python.exe -u scripts/train.py `
    --scenario rbar `
    --total-timesteps $TotalTimesteps `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --run-tag $RunTag
