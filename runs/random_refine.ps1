# Refine an existing model on "random", widening the direction sector.
#
# Seeding from a V-bar specialist is the intended use: the "random" dv_ref is
# 3*pi/4 * dv_opt precisely so its actuator scale matches the vbar scenario's
# at 0 deg, letting a vbar model transfer in unchanged. Its usable envelope is
# only ~+-0.12 deg though, so start the sector tiny.
#
# -TotalTimesteps is the ABSOLUTE target including the resumed model's steps.

param(
    [Parameter(Mandatory = $true)][string]$ResumeFrom,
    [string]$RunTag = "random_refine_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 2300000,
    [double]$AngleStart = 0.1,
    [double]$AngleIncrement = 1.0,
    [double]$NoiseStart = 0.05,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.5
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "random refine | run-tag=$RunTag | steps=$TotalTimesteps | from $ResumeFrom"
Write-Host "  sector +-$AngleStart deg, +$AngleIncrement per cleared window | noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

& ./.conda/python.exe -u scripts/train.py `
    --scenario random `
    --total-timesteps $TotalTimesteps `
    --resume-from $ResumeFrom `
    --curriculum-start-distance 1000 `
    --angle-curriculum `
    --angle-curriculum-start $AngleStart `
    --angle-curriculum-increment $AngleIncrement `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --run-tag $RunTag
