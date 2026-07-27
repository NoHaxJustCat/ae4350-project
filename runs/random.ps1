# Fresh "random" (all-angles) run, ramping the direction sector outward from
# V-bar. Sampling the whole circle from step one drives the policy into
# brute-force thrusting at ~11x optimal -- see ENV_ANGLE_CURRICULUM_* in
# libs/constants.py.

param(
    [string]$RunTag = "random_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 2000000,
    [double]$AngleStart = 0.1,
    [double]$AngleIncrement = 1.0,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.35
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "random | run-tag=$RunTag | steps=$TotalTimesteps"
Write-Host "  sector +-$AngleStart deg, +$AngleIncrement per cleared window | noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

& ./.conda/python.exe -u scripts/train.py `
    --scenario random `
    --total-timesteps $TotalTimesteps `
    --angle-curriculum `
    --angle-curriculum-start $AngleStart `
    --angle-curriculum-increment $AngleIncrement `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --run-tag $RunTag
