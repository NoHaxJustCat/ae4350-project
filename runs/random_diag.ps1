# Three 100k "random" runs to find out why the scenario is not learning.
#
# Every arm is the CURRENT configuration -- distance ramp 30 -> 1000 m, full
# circle from step 0 -- with exactly one knob changed:
#
#   base    unchanged. The control.
#   wide    --net-width 512 instead of 256, so both hidden layers AND the
#           encoder features_dim double.
#   warm    --learning-starts 25000 instead of 5000. Five times as much uniform
#           random experience in the replay buffer before the actor takes over
#           and gradient updates begin.
#
# Why "warm" is worth a whole arm: at 5000 the actor starts driving while the
# critic is still untrained, saturates at the action-box corner, and the run
# settles for brute-force thrusting -- measured on V-bar as 7.20x optimal at
# 5000 against 1.08x at 25000, in runs/actuator_test.ps1. "random" has never
# been tested at 25000, and it is the harder scenario: the buffer has to cover
# a whole circle of directions rather than one, so 5000 steps of warmup buys
# proportionally less coverage per direction than it does on V-bar.
#
# It also costs almost nothing in wall-clock: warmup runs at ~1200 steps/s
# against ~70 once gradients start, so the extra 20k steps are ~17 seconds.
#
# All three share one seed, one noise schedule and (except "warm") one warmup,
# so a difference between them is the knob. n=1 per arm: read a small gap as
# noise, not signal.
#
# Runs SEQUENTIALLY on purpose. NUM_ENVS is 6 on a 6-core box, so two at once
# would just halve each other's throughput.
#
# Resumable: an arm whose history.npz already exists is skipped. -Force redoes
# everything.

param(
    [int]$TotalTimesteps = 100000,
    [int]$Seed = 42,
    [int]$LearningStarts = 5000,      # the config default; "warm" overrides it
    [int]$WarmLearningStarts = 25000,
    [int]$NetWidth = 512,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.005,
    [double]$NoiseDecayFrac = 0.8,
    [int]$EvalFreq = 10000,
    [string[]]$Only = @(),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

# name -> extra flags beyond the shared ones. Empty = the unchanged config.
$Arms = [ordered]@{
    "base" = @()
    "wide" = @("--net-width", $NetWidth)
    "warm" = @("--learning-starts", $WarmLearningStarts)
}

$names = if ($Only) { $Only } else { @($Arms.Keys) }
foreach ($name in $names) {
    if (-not $Arms.Contains($name)) { throw "Unknown arm '$name'. Known: $($Arms.Keys -join ', ')" }
}

Write-Host "random diagnostics | $TotalTimesteps steps x $($names.Count) arms | seed $Seed"
Write-Host "  shared: noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac, eval every $EvalFreq"

foreach ($name in $names) {
    $tag = "rdiag_$name"
    if ((Test-Path "tmp/$tag/model/history.npz") -and (-not $Force)) {
        Write-Host "`n=== ${name}: already done, skipping (use -Force to redo) ==="
        continue
    }
    # "warm" carries its own --learning-starts, so do not also pass the shared
    # one: argparse would take the last occurrence and the arm would be a no-op.
    $warmup = if ($name -eq "warm") { @() } else { @("--learning-starts", $LearningStarts) }

    Write-Host "`n=== $name === extra flags: $($Arms[$name] -join ' ')"
    & ./.conda/python.exe -u scripts/train.py `
        --scenario random `
        --total-timesteps $TotalTimesteps `
        --seed $Seed `
        --noise-std-start $NoiseStart `
        --noise-std-end $NoiseEnd `
        --noise-decay-frac $NoiseDecayFrac `
        --eval-freq $EvalFreq `
        @warmup `
        @($Arms[$name]) `
        --run-tag $tag
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

Write-Host "`n=== all arms done ==="
Write-Host "Compare with:"
foreach ($name in $names) {
    Write-Host "  ./.conda/python.exe scripts/eval.py tmp/rdiag_$name/model/best_model.zip --scenario random -n 100"
}
Write-Host "  ./.conda/python.exe scripts/plot.py $(($names | ForEach-Object { "tmp/rdiag_$_" }) -join ' ') --compare"
