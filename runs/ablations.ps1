# Hyperparameter ablation study for the report: one knob changed per run.
#
# EVERY run shares one fixed -Seed and -LearningStarts; apart from the named
# knob they are identical (100k steps, noise 0.10 -> 0.01 over 80%, eval every
# 10k). Seeding is verified bit-for-bit reproducible.
#
# The nominal is tmp/abl_nominal, copied from the winning arm of
# runs/actuator_test.ps1 -- same seed, steps, schedule and warmup, so it is a
# real member of this grid, not an older run pressed into service. It reaches
# 1.08x optimal with 97% dock. Do NOT use tmp/vbar_fresh: at the old
# learning_starts=5000 it sits at 7.20x, and dock rate cannot see the difference.
#
# One seed means n=1 per config, so read a lone large jump as possibly noise
# until a second seed confirms it. The gamma sweep is three points
# (0.95/0.90/0.80) so that knob shows a trend rather than an anecdote.
#
# Resumable: a run whose history.npz already exists is skipped, so an
# interrupted study restarts where it stopped. Pass -Force to redo everything.

param(
    [int]$TotalTimesteps = 100000,
    [int]$Seed = 42,
    # 25000, not the config default 5000. At 5000 the actor starts driving while
    # the critic is still untrained, saturates at the action-box corner, and the
    # run settles for brute-force thrusting at ~7.14x optimal. At 25000 the same
    # config reaches 1.08x -- and finishes FASTER, because the extra warmup runs
    # at ~1200 steps/s against ~80 once gradients start. Measured, three arms,
    # runs/actuator_test.ps1.
    [int]$LearningStarts = 25000,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.8,
    [int]$EvalFreq = 10000,
    [string[]]$Only = @(),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

# name -> extra flags beyond the shared ones. Empty = the nominal.
$Ablations = [ordered]@{
    "nominal"        = @()
    # Three points, not one: gamma is the knob most likely to be confounded with
    # which fuel basin the run lands in, so it needs a trend to be credible.
    "gamma_095"      = @("--gamma", "0.95")
    "gamma_090"      = @("--gamma", "0.90")
    "gamma_080"      = @("--gamma", "0.80")
    "lr_low"         = @("--learning-rate", "1e-5")
    "lr_high"        = @("--learning-rate", "1e-3")
    "net_32"         = @("--net-width", "32")
    "net_64"         = @("--net-width", "64")
    "net_128"        = @("--net-width", "128")
    # Replay capacity down to one batch: each update sees only the newest
    # transitions, which is the "no replay buffer" ablation.
    "no_replay"      = @("--buffer-size", "1280")
    "no_curriculum"  = @("--no-curriculum")
    "no_encoder"     = @("--no-smart-encoder")
    "sac"            = @("--algo", "sac")
    "ddpg"           = @("--algo", "ddpg")
}

$Names = if ($Only.Count) { $Only } else { @($Ablations.Keys) }
foreach ($n in $Names) {
    if (-not $Ablations.Contains($n)) { throw "Unknown ablation '$n'" }
}

Write-Host "ablation study | $($Names.Count) runs x $TotalTimesteps steps | seed $Seed | learning_starts $LearningStarts" -ForegroundColor Cyan

$i = 0
foreach ($name in $Names) {
    $i++
    $tag = "abl_$name"
    $history = "tmp/$tag/model/history.npz"
    if ((Test-Path $history) -and -not $Force) {
        Write-Host "[$i/$($Names.Count)] $name -- already done, skipping" -ForegroundColor DarkGray
        continue
    }

    $extra = $Ablations[$name]
    Write-Host "`n[$i/$($Names.Count)] $name  $($extra -join ' ')" -ForegroundColor Yellow
    $started = Get-Date

    & ./.conda/python.exe -u scripts/train.py `
        --scenario vbar `
        --total-timesteps $TotalTimesteps `
        --noise-std-start $NoiseStart `
        --noise-std-end $NoiseEnd `
        --noise-decay-frac $NoiseDecayFrac `
        --eval-freq $EvalFreq `
        --seed $Seed `
        --learning-starts $LearningStarts `
        --run-tag $tag `
        @extra

    if ($LASTEXITCODE -ne 0) {
        # Don't abort the study: one bad variant should not cost the rest.
        Write-Host "[$i] $name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        continue
    }
    Write-Host ("[$i] $name done in {0:hh\:mm\:ss}" -f ((Get-Date) - $started)) -ForegroundColor Green
}

Write-Host "`nCollecting..." -ForegroundColor Cyan
& ./.conda/python.exe -u scripts/collect_ablations.py
