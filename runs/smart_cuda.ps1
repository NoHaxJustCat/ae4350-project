<#
.SYNOPSIS
    Local training launch: arch=smart, device=cuda, net-arch 256x256,
    n-envs=6 (dummy vec-env), gamma=0.99, lr=1e-4, 10M timesteps.

    Saved from the exact command used to start the "ikitworks" run:
      ./.conda/python.exe -u training.py --scenario vbar --arch smart
        --device cuda --n-envs 6 --vec-env dummy --net-arch 256,256
        --features-dim 256 --n-blocks 2 --activation relu --gamma 0.99
        --lr 1e-4 --torch-threads 5 --total-timesteps 10000000
        --checkpoint-freq 100000 --keep-last-checkpoints 2 --run-tag ikitworks

    Does NOT enable --fuel-curriculum (the Δv-budget ratchet): with the
    terminal braking phase + stopping bonus now driving Δv efficiency
    directly, the budget's generous 3x floor is non-binding and just adds a
    moving part. Re-add --fuel-curriculum below if you want it back.
.USAGE
    .\runs\smart_cuda.ps1                  # auto-generated timestamp run-tag
    .\runs\smart_cuda.ps1 -RunTag mytag2    # named run-tag
    .\runs\smart_cuda.ps1 -Scenario rbar -RunTag rbar1
#>

param(
    [string]$RunTag = "smart_cuda_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$Scenario = "vbar",
    [int]$TotalTimesteps = 300000
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
    --net-arch 256,256 `
    --features-dim 256 `
    --n-blocks 2 `
    --activation relu `
    --gamma 0.99 `
    --lr 1e-4 `
    --torch-threads 5 `
    --total-timesteps $TotalTimesteps `
    --checkpoint-freq 200000 `
    --keep-last-checkpoints 2 `
    --run-tag $RunTag
