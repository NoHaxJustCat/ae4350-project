# Continue an existing "random" run, keeping its angle sector.
#
# Use runs/random.ps1 to START from the V-bar specialist; use this to pick a
# run back up from one of its own checkpoints. The sector is read from the
# checkpoint's .curriculum.json sidecar, so a partly-widened run resumes where
# it left off; pass -AngleStart to deliberately restart the ramp narrower.
#
# -TotalTimesteps is the ABSOLUTE target including the resumed model's steps.

param(
    [Parameter(Mandatory = $true)][string]$ResumeFrom,
    [string]$RunTag = "random_refine_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 4000000,
    [double]$AngleStart = 0.1,
    [double]$AngleIncrement = 1.0,
    [double]$NoiseStart = 0.02,
    [double]$NoiseEnd = 0.005,
    [double]$NoiseDecayFrac = 0.5
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "random refine | run-tag=$RunTag | steps=$TotalTimesteps | from $ResumeFrom"
Write-Host "  sector floor +-$AngleStart deg, +$AngleIncrement per cleared window | noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

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
