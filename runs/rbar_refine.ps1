# Refine an existing R-bar model at full distance.
# -TotalTimesteps is the ABSOLUTE target including the resumed model's steps.

param(
    [Parameter(Mandatory = $true)][string]$ResumeFrom,
    [string]$RunTag = "rbar_refine_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 800000,
    [double]$NoiseStart = 0.01,
    [double]$NoiseEnd = 0.002,
    [double]$NoiseDecayFrac = 0.5
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "rbar refine | run-tag=$RunTag | steps=$TotalTimesteps | from $ResumeFrom"
Write-Host "  noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

& ./.conda/python.exe -u scripts/train.py `
    --scenario rbar `
    --total-timesteps $TotalTimesteps `
    --resume-from $ResumeFrom `
    --curriculum-start-distance 1000 `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --run-tag $RunTag
