# Continue a "random" run from its own latest checkpoint.
#
#   .\runs\random_resume.ps1                          # continues -Run below
#   .\runs\random_resume.ps1 -Run random_20260728_095951
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
# Noise defaults are low: this continues a policy that already docks, so it
# should refine rather than re-explore. On resume the decay schedule restarts
# over the REMAINING budget.
#
# NOTE: the replay buffer is not saved, so gradient updates pause until it
# refills past MIN_BUFFER. -TotalTimesteps is the ABSOLUTE target including the
# resumed model's own steps.

param(
    [string]$Run = "random_20260728_095951",
    [string]$ResumeFrom = "",
    [string]$RunTag = "",
    [int]$TotalTimesteps = 2000000,
    [double]$NoiseStart = 0.05,
    [double]$NoiseEnd = 0.01,
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
