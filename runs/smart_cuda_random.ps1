<#
.SYNOPSIS
    Local training launch for the "random" scenario: arch=smart, device=cuda,
    net-arch 256x256, n-envs=6 (dummy vec-env), gamma=0.99, lr=1e-4.

    "random" spawns the initial displacement at a UNIFORMLY random in-plane
    direction on the circle at the curriculum distance, so one policy must dock
    from every angle. There is no analytic dv_ref here: the fuel bonus grades
    against the numeric per-direction optimum dv_opt (looked up per episode),
    and dv_ref is derived from it (ENV_RANDOM_DV_REF_MULT). See libs/env.py.

    Trains FRESH (no --resume-from): the vbar/rbar models only ever saw one
    fixed geometry, so they are a poor and possibly harmful starting point for
    the all-angles task. The env builds a per-direction optimum table at
    startup (~20 s, one-time) before the first log line appears.

    Does NOT enable --fuel-curriculum (the Δv-budget ratchet): with the terminal
    braking phase + stopping bonus driving Δv efficiency directly, the budget's
    generous 3x floor is non-binding and just adds a moving part.
.USAGE
    .\runs\smart_cuda_random.ps1                        # auto timestamp run-tag
    .\runs\smart_cuda_random.ps1 -RunTag random1
    .\runs\smart_cuda_random.ps1 -TotalTimesteps 3000000
#>

param(
    [string]$RunTag = "smart_cuda_random_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$Scenario = "random",
    [int]$TotalTimesteps = 2000000
)

$ErrorActionPreference = "Stop"

# Always run from the project root regardless of where this script is
# invoked from (it lives one level down, in runs/).
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "Launching training: scenario=$Scenario run-tag=$RunTag total-timesteps=$TotalTimesteps"

& ./.conda/python.exe -u training.py `
    --scenario $Scenario `
    --arch smart `
    --device cuda `
    --n-envs 6 `
    --vec-env dummy `
    --net-arch 64,64,64 `
    --features-dim 64 `
    --n-blocks 2 `
    --activation relu `
    --gamma 0.99 `
    --lr 1e-4 `
    --torch-threads 5 `
    --total-timesteps $TotalTimesteps `
    --checkpoint-freq 10000 `
    --keep-last-checkpoints 3 `
    --resume-from trained/20260723_204210/smart_cuda_random_20260723_204148/checkpoints/random_td3_1499802_steps.zip `
    --run-tag $RunTag
