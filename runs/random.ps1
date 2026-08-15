# "random" (all-angles) run, trained FROM SCRATCH up the distance curriculum.
#
# Not seeded from the V-bar specialist, and no angle curriculum. Both of those
# start the policy at 1000 m, which measurement says is the worst possible
# place to begin: the docking basin scales as 5/d (the tolerance is a fixed
# 5 m while everything else scales with distance), so it is ~12x wider at 30 m
# than at 1000 m. Dock rate of the OPTIMAL action under 0.02 burn noise:
#
#            30 m   1000 m
#     0 deg   82%       5%
#    15 deg   10%       0%
#    45 deg    3%       0%
#    90 deg   15%       0%
#
# So: learn the angle->(burn, coast) mapping at short range where noise is
# survivable, then let the distance curriculum tighten precision. That mapping
# is itself distance-invariant -- CW is linear, so the optimal normalized burn
# and coast time depend only on direction -- which is what makes the ramp work.
#
# ENV_RANDOM_DV_REF_MULT is also down to 1.0 (max_dv = 1.5*dv_opt), which buys
# back off-axis resolution. See config.py. Both changes are random-only.

param(
    [string]$RunTag = "random_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$TotalTimesteps = 1000000,
    # 25000, not the config default 5000. Measured over three 100k arms
    # (runs/random_diag.ps1): at 5000 the actor starts driving an untrained
    # critic, pins the coast command to its minimum and thrusts its way in at
    # 5-20x optimal; at 25000 the same config coasts, uses 2 burns and holds
    # ~1.6x. The warmup itself is nearly free -- it runs at ~1200 steps/s
    # against ~80 once gradient updates begin.
    [int]$LearningStarts = 25000,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.005,
    [double]$NoiseDecayFrac = 0.6,
    [int]$EvalFreq = 10000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "random | run-tag=$RunTag | steps=$TotalTimesteps | from scratch | seed $Seed"
Write-Host "  distance curriculum 30 -> 1000 m, full angle range | noise $NoiseStart -> $NoiseEnd over $NoiseDecayFrac"
Write-Host "  warmup $LearningStarts steps of uniform random actions before gradients start"

& ./.conda/python.exe -u scripts/train.py `
    --scenario random `
    --total-timesteps $TotalTimesteps `
    --learning-starts $LearningStarts `
    --noise-std-start $NoiseStart `
    --noise-std-end $NoiseEnd `
    --noise-decay-frac $NoiseDecayFrac `
    --eval-freq $EvalFreq `
    --seed $Seed `
    --run-tag $RunTag
