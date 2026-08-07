# Hyperparameter ablation study for the report: one knob changed per run.
#
# The NOMINAL run is not launched here -- tmp/vbar_fresh already is it. That run
# used the same observation/action spaces, gamma 0.99, lr 1e-4, buffer 300k,
# batch 1280, net [256,256] and 150k steps; every config.py change since is
# random-scenario-only. scripts/collect_ablations.py writes its params.json.
#
# All runs share the nominal's noise schedule (0.10 -> 0.01 over 35%) and its
# 20k eval cadence, so the ONLY difference from the baseline is the named knob.
#
# Seeds are left unset, matching the nominal, which was also unseeded. Read
# small differences with that in mind.
#
# Resumable: a run whose history.npz already exists is skipped, so an
# interrupted study restarts where it stopped. Pass -Force to redo everything.

param(
    [int]$TotalTimesteps = 150000,
    [double]$NoiseStart = 0.10,
    [double]$NoiseEnd = 0.01,
    [double]$NoiseDecayFrac = 0.35,
    [int]$EvalFreq = 20000,
    [string[]]$Only = @(),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

# name -> extra flags. Empty = the nominal, which is reused, not rerun.
$Ablations = [ordered]@{
    "gamma_low"      = @("--gamma", "0.90")
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

Write-Host "ablation study | $($Names.Count) runs x $TotalTimesteps steps" -ForegroundColor Cyan

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
