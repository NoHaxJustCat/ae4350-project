<#
.SYNOPSIS
    Local training launch for the "random" scenario: arch=smart, device=cuda,
    net-arch 256x256, n-envs=6 (dummy vec-env), gamma=0.99, lr=1e-4.

    "random" spawns the initial displacement at a random in-plane direction at
    the curriculum distance, so one policy must dock from every angle. There is
    no analytic dv_ref here: the fuel bonus grades against the numeric
    per-direction optimum dv_opt (looked up per episode), and dv_ref is derived
    from it (ENV_RANDOM_DV_REF_MULT). See libs/env.py.

    This launcher takes the ANGLE-CURRICULUM approach, after a per-angle sweep
    of the previous uniform-sampling runs showed why they plateaued at ~11x the
    achievable optimum while the V-bar specialist reaches 1.15x:

      * The uniform-trained policies never learn to COAST — coast_units sits
        pinned at its minimum (brute-force thrusting) or bangs between the two
        rails. Coasting is the entire source of Δv efficiency in a CW transfer.
      * dv_opt varies 23.6x with direction and V-bar is a CUSP in it (1.00x at
        0 deg, 3.54x at 5 deg, 5.78x at 10 deg), yet only 3.9% of uniform
        angles land within 2x of the V-bar optimum.
      * The optimal coasting trajectory peaks at 1.02x the initial distance at
        V-bar but 1.57-1.64x at 30-60 deg, against an excursion_limit of 2x —
        so there is ~96% margin for noisy exploration of the efficient solution
        on-axis but only ~20% off it. Uniform sampling terminates most
        exploratory coasts out-of-bounds, and the ~600 of terminal bonus lost
        each time points the policy straight back at brute force.

    So: start from out/best_1 (the 1.15x V-bar specialist), at FULL 1000 m
    distance, sampling only +-2 deg around the V-bar axis, and let
    AngleCurriculumCallback widen the sector as the dock rate holds. At +-90 deg
    the sector IS the uniform draw, so a graduated run has trained the original
    task with a working coasting solution in the replay buffer the whole way.

    EXPECT A LOW DOCK RATE FOR THE FIRST STRETCH. best_1 reproduces its
    1.15x-optimal dock exactly at 0 deg in this scenario (that is what the
    3*pi/4 ENV_RANDOM_DV_REF_MULT buys), but it misses by 0.5 deg and goes
    out-of-bounds past 1.5 deg: it fires one burn and never corrects, so it has
    essentially no robustness to carry off-axis. The opening phase is the
    policy learning a mid-coast correction burn. The sector floors at
    -AngleStart and only ever widens on a >=50% dock-rate window, so this
    costs time, not progress — watch angle_half_width_deg in status.json.

    net-arch is back to 256,256 / features-dim 256 — the 64,64,64 / dim-64
    variant tried previously is both deeper and 11x smaller (17.5k vs 201k actor
    params) and took the dock rate from 100% to 61% despite 5x more steps, which
    matches the known --arch smart depth limit (see --n-blocks help text).

    --curriculum-start-distance 1000 skips the distance ramp entirely (best_1
    already docks at 1000 m) AND floors the curriculum there, so a dock-rate dip
    while the sector widens cannot regress the distance out from under it.

    Does NOT enable --fuel-curriculum (the Δv-budget ratchet): with the terminal
    braking phase + stopping bonus driving Δv efficiency directly, the budget's
    generous 3x floor is non-binding and just adds a moving part.

    NOTE: --total-timesteps is the ABSOLUTE target including the resumed model's
    own steps (best_1 is at 300k), so the default below buys 2M new ones.
.USAGE
    .\runs\smart_cuda_random.ps1                        # auto timestamp run-tag
    .\runs\smart_cuda_random.ps1 -RunTag random1
    .\runs\smart_cuda_random.ps1 -TotalTimesteps 5300000
    .\runs\smart_cuda_random.ps1 -ResumeFrom trained/.../checkpoints/random_td3_900000_steps.zip
#>

param(
    [string]$RunTag = "smart_cuda_random_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$Scenario = "random",
    [int]$TotalTimesteps = 2300000,
    [string]$ResumeFrom = "out/best_1/vbar_td3.zip",
    [double]$AngleStart = 2.0
)

$ErrorActionPreference = "Stop"

# Always run from the project root regardless of where this script is
# invoked from (it lives one level down, in runs/).
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

Write-Host "Launching training: scenario=$Scenario run-tag=$RunTag total-timesteps=$TotalTimesteps"
Write-Host "  resume-from: $ResumeFrom   angle sector start: +-$AngleStart deg"

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
    --checkpoint-freq 10000 `
    --keep-last-checkpoints 3 `
    --curriculum-start-distance 1000 `
    --angle-curriculum `
    --angle-curriculum-start $AngleStart `
    --angle-curriculum-increment 5 `
    --resume-from $ResumeFrom `
    --run-tag $RunTag
