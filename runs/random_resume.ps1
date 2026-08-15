# Continue a "random" run from its own latest checkpoint.
#
#   .\runs\random_resume.ps1                          # continues -Run below
#   .\runs\random_resume.ps1 -Run random_20260728_095951   # a different run
#   .\runs\random_resume.ps1 -ResumeFrom <path-to.zip> # pick a checkpoint yourself
#
# The checkpoint's .curriculum.json sidecar carries the curriculum distance, so
# the run picks up where it left off -- do NOT pass -CurriculumStartDistance
# unless you deliberately want to move it.
#
# Output goes to a NEW tmp/<run-tag>/ directory. It must not reuse the source
# run's tag: RunPaths wipes the directory it is given, which would delete the
# run being resumed.
#
# Noise picks up exactly where random.ps1 leaves off. That run decays
# 0.10 -> 0.005 over the first 80% of its budget, so it spends its whole tail
# at the 0.005 floor; starting a resume anywhere above that would re-inject
# exploration into a policy that has already settled, and restarting the decay
# from 0.05 (the old default here) meant a 10x noise step UP at the seam. Flat
# 0.005 -> 0.005 keeps the continuation seamless. Drop below the floor only
# deliberately -- runs/random_refine.ps1 is the script for that.
#
# NOTE: the replay buffer is not saved, so gradient updates pause until it
# refills past MIN_BUFFER. -TotalTimesteps is the ABSOLUTE target including the
# resumed model's own steps.

param(
    [string]$Run = "random_20260815_114845",
    [string]$ResumeFrom = "",
    [string]$RunTag = "",
    [int]$TotalTimesteps = 1000000,
    [double]$NoiseStart = 0.005,
    [double]$NoiseEnd = 0.005,
    [double]$NoiseDecayFrac = 0.5,
    [int]$EvalFreq = 10000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

if (-not $ResumeFrom) {
    $dir = "tmp/$Run/model/checkpoints"
    if (-not (Test-Path $dir)) { throw "No checkpoints directory at $dir" }
    $latest = Get-ChildItem "$dir/*_steps.zip" | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $latest) { throw "No checkpoints in $dir" }
    $ResumeFrom = $latest.FullName
}
if (-not (Test-Path $ResumeFrom)) { throw "Checkpoint not found: $ResumeFrom" }

if (-not $RunTag) { $RunTag = "${Run}_cont_$(Get-Date -Format 'HHmmss')" }
if ($RunTag -eq $Run) { throw "RunTag must differ from -Run, or the source run is deleted." }

Write-Host "random resume | run-tag=$RunTag | steps=$TotalTimesteps"
Write-Host "  from: $ResumeFrom"
Write-Host "  noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac | eval every $EvalFreq"

& ./.conda/python.exe -u scripts/train.py `
    --scenario random `
    --total-timesteps $TotalTimesteps `
    --resume-from $ResumeFrom `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --eval-freq $EvalFreq `
    --run-tag $RunTag
