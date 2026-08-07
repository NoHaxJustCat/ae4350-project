# Hyperparameter ablation study for the report: one knob changed per run.
#
# EVERY run here shares one fixed -Seed, INCLUDING the nominal, which is
# therefore retrained rather than reused. tmp/vbar_fresh is NOT a valid
# baseline: the V-bar outcome is bimodal -- policies land either in the coasting
# basin (~1.1x optimal) or the brute-force-thrust basin (~7.2x) -- and
# vbar_fresh is pinned in the bad one, with 91% of its last 5000 episodes within
# 1% of exactly 7.20x. Dock rate does not reveal this (96% either way); only
# dv_ratio does. Comparing single unseeded runs against it credited basin luck
# to the knob: the first gamma 0.90 run looked like reward 931 vs 515 purely
# because the baseline had fallen into the bad basin.
#
# One seed still means n=1 per config, so a lone large jump should be read as
# possible basin luck until a second seed confirms it. The gamma sweep is three
# points (0.95/0.90/0.80) precisely so that knob shows a trend rather than an
# anecdote.
#
# Apart from the named knob, all runs share the nominal's noise schedule
# (0.10 -> 0.01 over 35%), 150k steps and 20k eval cadence.
#
# The nominal runs FIRST, so if the baseline itself lands in the bad basin the
# study can be stopped early instead of after 7 hours.
#
# Resumable: a run whose history.npz already exists is skipped, so an
# interrupted study restarts where it stopped. Pass -Force to redo everything.

param(
    [int]$TotalTimesteps = 150000,
    [int]$Seed = 42,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.35,
    [int]$EvalFreq = 20000,
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

Write-Host "ablation study | $($Names.Count) runs x $TotalTimesteps steps | seed $Seed" -ForegroundColor Cyan

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
