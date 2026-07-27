# "random" (all-angles) run, seeded from the V-bar specialist at full distance
# and ramping the direction sector outward from V-bar.
#
# Starts from the specialist rather than from scratch: the "random" dv_ref is
# 3*pi/4 * dv_opt precisely so its actuator scale matches the vbar scenario's
# at 0 deg, letting a vbar model transfer in unchanged (1.15x optimal, 100%
# dock). Sampling the whole circle from step one instead drives the policy into
# brute-force thrusting at ~11x optimal -- see ENV_ANGLE_CURRICULUM_* in
# config.py.
#
# The seed's usable envelope is only ~+-0.12 deg, so the sector starts tiny and
# the opening phase is the policy learning a mid-coast correction burn.
#
# -TotalTimesteps is the ABSOLUTE target including the seed's own steps.

param(
    [string]$RunTag = "random_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$ResumeFrom = "out/vbar_specialist/vbar_td3.zip",
    [int]$TotalTimesteps = 2300000,
    [double]$AngleStart = 0.25,
    [double]$AngleIncrement = 0.25,
    [double]$NoiseStart = 0.3,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.75
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "random | run-tag=$RunTag | steps=$TotalTimesteps | from $ResumeFrom"
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
