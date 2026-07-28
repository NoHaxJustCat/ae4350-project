# Refine out/random_specialist at very low noise.
#
# Polishes a converged policy rather than searching, so the noise defaults are
# an order of magnitude below a training run. Use runs/random.ps1 to train from
# scratch, runs/random_resume.ps1 to continue a run in tmp/.
#
# Where the specialist stands (60 deterministic episodes, full circle):
#     dock rate  91.7%
#     dv/opt     1.56x median
#     arrival    3.51 m/s  <-- 1.38x dv_opt, i.e. it is NOT braking
#
# So the thing to watch is reward_term rising and arrival speed falling. It
# currently collects ~10 of a possible 500 stop bonus, and braking is worth
# about +420 net even after the extra fuel, so the gradient does point there.
# dv/opt drifting up toward ~2 while that happens is correct, not a regression.
#
# CAUTION: a V-bar refinement at low noise once collapsed from ~95% dock to 0%
# -- the actor drifted off a narrow basin and the critic inverted. best_model
# protects the artifact, but watch the [eval] line rather than the training log.
#
# -TotalTimesteps is the ABSOLUTE target including the resumed model's steps
# (best_model is at ~960k).

param(
    [string]$ResumeFrom = "out/random_specialist/model/best_model.zip",
    [string]$RunTag = "random_refine_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 1500000,
    [double]$NoiseStart = 0.005,
    [double]$NoiseEnd = 0.001,
    [double]$NoiseDecayFrac = 0.5,
    [int]$EvalFreq = 10000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path $ResumeFrom)) { throw "Checkpoint not found: $ResumeFrom" }

Write-Host "random refine | run-tag=$RunTag | steps=$TotalTimesteps"
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
