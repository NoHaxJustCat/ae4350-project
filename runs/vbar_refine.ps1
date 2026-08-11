# Refine an existing V-bar model at full distance.
#
# Refinement polishes a converged policy rather than searching, so the noise
# defaults are much lower than a fresh run. Note the V-bar dock survives only a
# ~0.006-wide window in the along-track burn: the actor can drift out of it and
# not recover, which is what <run_dir>/best_model.zip is there to protect.
#
# -TotalTimesteps is the ABSOLUTE target including the resumed model's steps.

param(
    [Parameter(Mandatory = $true)][string]$ResumeFrom,
    [string]$RunTag = "vbar_refine_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 800000,
    [double]$NoiseStart = 0.01,
    [double]$NoiseEnd = 0.002,
    [double]$NoiseDecayFrac = 0.5
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "vbar refine | run-tag=$RunTag | steps=$TotalTimesteps | from $ResumeFrom"
Write-Host "  noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"

& ./.conda/python.exe -u scripts/train.py `
    --scenario vbar `
    --total-timesteps $TotalTimesteps `
    --resume-from $ResumeFrom `
    --curriculum-start-distance 1000 `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --run-tag $RunTag
