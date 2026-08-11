# Why does a fresh V-bar run collapse the moment the policy takes over?
#
# At `learning_starts` two things switch on at once: the actor starts driving,
# and it starts being trained against a critic that has never seen a gradient.
# An untrained critic is ~linear in the action, so "maximize Q" points at a
# corner of the action box. Measured on a fresh run: dv per step pinned at
# 5.0053x optimal, against a predicted corner of max_dv*sqrt(2) = 3.5393*1.41421
# = 5.0053. Exact to five figures -- the actor is commanding +-1 on both thrust
# axes and minimum coast, every step, until it exits the boundary.
#
# Three arms, one variable each:
#   baseline    vbar as-is, max_dv = 3.54*dv_opt
#   dvref15     max_dv = 1.50*dv_opt, matching what commit 17e3bc2 gave "random"
#               (and which config.py's own table says lifts the OPTIMAL action
#               from 82% to 100% dock at 0deg under burn noise)
#   lstart25k   actuator untouched, but 5x the random-action warmup, so the
#               critic sees far more data before the actor engages
#
# Arm 1 doubles as the ablation study's new nominal.

param(
    [int]$TotalTimesteps = 100000,
    [int]$Seed = 42,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.8,
    [int]$EvalFreq = 10000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

$Arms = [ordered]@{
    "act_baseline"  = @()
    "act_dvref15"   = @("--dv-ref-mult", "1.0")
    "act_lstart25k" = @("--learning-starts", "25000")
}

Write-Host "actuator test | $($Arms.Count) arms x $TotalTimesteps steps | seed $Seed" -ForegroundColor Cyan

foreach ($name in $Arms.Keys) {
    if (Test-Path "tmp/$name/model/history.npz") {
        Write-Host "$name -- already done, skipping" -ForegroundColor DarkGray
        continue
    }
    Write-Host "`n== $name  $($Arms[$name] -join ' ')" -ForegroundColor Yellow
    $started = Get-Date

    & ./.conda/python.exe -u scripts/train.py `
        --scenario vbar `
        --total-timesteps $TotalTimesteps `
        --noise-std-start $NoiseStart `
        --noise-std-end $NoiseEnd `
        --noise-decay-frac $NoiseDecayFrac `
        --eval-freq $EvalFreq `
        --seed $Seed `
        --run-tag $name `
        @($Arms[$name])

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        continue
    }
    Write-Host ("$name done in {0:hh\:mm\:ss}" -f ((Get-Date) - $started)) -ForegroundColor Green
}
