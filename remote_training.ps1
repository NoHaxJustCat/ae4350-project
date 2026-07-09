<#
.SYNOPSIS
    Syncs the project to aurora-server, creates/uses a conda env,
    runs training (blocking, with live logs), then pulls back out/ and trained/.
    While training runs, periodically pulls back the latest trajectory PNG.
.USAGE
    .\deploy_train.ps1
#>

$ErrorActionPreference = "Stop"

#############################################
# CONFIG
#############################################
$RemoteHost      = "nicolobasso@aurora-server"
$RemoteSubfolder = "hdp_training"
$CondaEnvName    = "hdp-cw"
$PythonVersion   = "3.11"
$TrainCommand    = "python -u training.py --scenario vbar --n-envs 32"
$ExcludeNames    = @(".git", ".conda", ".vscode", "out", "trained", "results", "tmp",
                     "__pycache__", ".venv", "venv")
$PollIntervalSeconds = 20
#############################################

# ── 1. Sync ───────────────────────────────────────────────────────────────────
Write-Host "`n==> [1/4] Syncing to ${RemoteHost}:~/${RemoteSubfolder}/" -ForegroundColor Cyan
ssh $RemoteHost "mkdir -p ~/${RemoteSubfolder}/out ~/${RemoteSubfolder}/trained"

$StagingDir = Join-Path $env:TEMP "hdp_stage"
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

$roboArgs = @(".", $StagingDir, "/E", "/XO", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
foreach ($ex in $ExcludeNames) { $roboArgs += @("/XD", $ex) }
robocopy @roboArgs | Out-Null

scp -r "$StagingDir\*" "${RemoteHost}:~/${RemoteSubfolder}/"
Remove-Item -Recurse -Force $StagingDir
Write-Host "    Sync complete."

# ── 2. Remote: setup + train ──────────────────────────────────────────────────
Write-Host "`n==> [2/4] Remote setup + training (live logs below)" -ForegroundColor Cyan
Write-Host "    SSH session open — do not close this window.`n" -ForegroundColor Yellow

$remoteSh = Join-Path $env:TEMP "hdp_remote_$(Get-Random).sh"

$bashScript = @'
#!/usr/bin/env bash
set -euo pipefail
trap 'echo "REMOTE ERROR at line $LINENO (exit $?)" >&2' ERR

REMOTE_DIR=~/__SUBFOLDER__
CONDA_ENV=__CONDAENV__
PYTHON_VER=__PYVER__

echo "[setup] Remote dir : $REMOTE_DIR"
echo "[setup] Conda env  : $CONDA_ENV"
echo "[setup] Python ver : $PYTHON_VER"
echo ""

# locate conda
CONDA_SH=""
for c in /opt/miniconda3/etc/profile.d/conda.sh \
          ~/miniconda3/etc/profile.d/conda.sh \
          ~/anaconda3/etc/profile.d/conda.sh \
          /opt/conda/etc/profile.d/conda.sh \
          /usr/local/conda/etc/profile.d/conda.sh; do
    if [ -f "$c" ]; then CONDA_SH="$c"; break; fi
done
if [ -z "$CONDA_SH" ]; then
    echo "ERROR: conda not found. SSH in and run: conda info | grep base" >&2
    exit 1
fi
echo "[setup] conda init : $CONDA_SH"
source "$CONDA_SH"

# create env if needed
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "[setup] Creating conda env '${CONDA_ENV}' (python ${PYTHON_VER})..."
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
else
    echo "[setup] Conda env '${CONDA_ENV}' already exists."
fi
conda activate "$CONDA_ENV"

# install deps
echo "[setup] Installing Python packages..."
pip install --upgrade pip
pip install stable_baselines3 torch numpy matplotlib "gymnasium[classic-control]" scipy

for req in "$REMOTE_DIR/requirements.txt" "$REMOTE_DIR/libs/requirements.txt"; do
    if [ -f "$req" ]; then
        echo "[setup] Installing from $req"
        pip install -r "$req"
    fi
done

# train
echo ""
echo "========================================="
echo " Training started  -- $(date)"
echo "========================================="
cd "$REMOTE_DIR"
rm -rf tmp
mkdir -p out trained tmp
__TRAINCMD__
echo ""
echo "========================================="
echo " Training finished -- $(date)"
echo "========================================="
'@

$bashScript = $bashScript.Replace("__SUBFOLDER__", $RemoteSubfolder)
$bashScript = $bashScript.Replace("__CONDAENV__",  $CondaEnvName)
$bashScript = $bashScript.Replace("__PYVER__",     $PythonVersion)
$bashScript = $bashScript.Replace("__TRAINCMD__",  $TrainCommand)

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$unixScript = $bashScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($remoteSh, $unixScript, $utf8NoBom)

scp "$remoteSh" "${RemoteHost}:~/${RemoteSubfolder}/_run.sh"
Remove-Item -Force $remoteSh

New-Item -ItemType Directory -Force -Path ".\tmp" | Out-Null
$LocalTmpDir = (Resolve-Path ".\tmp").Path
New-Item -ItemType Directory -Force -Path ".\trained\checkpoints" | Out-Null
$LocalTrainedDir = (Resolve-Path ".\trained").Path

# ── 3. Background poller ──────────────────────────────────────────────────────
# Also periodically pulls trained/checkpoints/ back (training.py now saves a
# checkpoint every --checkpoint-freq timesteps). If the SSH session below
# drops, the final "retrieve results" step never runs — this poller is what
# actually saves you from losing all progress, since it already pulled the
# most recent checkpoint within the last $PollIntervalSeconds.
#
# Uses `scp -r ".../dir/*" "local\dir\"` rather than rsync — rsync isn't
# installed on this machine (only OpenSSH's ssh/scp are). Unlike
# `rsync -az src/ dst/`, plain `scp -r src dst` does NOT merge into an
# existing dst — a second run would nest as dst\src\src\. Globbing the
# source (`src/*`) copies the *contents* into dst instead, which is safe to
# repeat every poll. 2>$null because early polls hit an empty/missing
# checkpoints dir before the first checkpoint is written — expected, not an
# error.
$pollJob = Start-Job -ScriptBlock {
    param($RemoteHost, $RemoteSubfolder, $LocalTmpDir, $LocalTrainedDir, $Interval)
    while ($true) {
        scp "${RemoteHost}:~/${RemoteSubfolder}/tmp/latest_trajectory.png" `
            (Join-Path $LocalTmpDir "latest_trajectory.png") 2>$null
        scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/checkpoints/*" `
            "$LocalTrainedDir/checkpoints/" 2>$null
        Start-Sleep -Seconds $Interval
    }
} -ArgumentList $RemoteHost, $RemoteSubfolder, $LocalTmpDir, $LocalTrainedDir, $PollIntervalSeconds

Write-Host "    Live trajectory + checkpoint polling started -> .\tmp\latest_trajectory.png, .\trained\checkpoints\ (every ${PollIntervalSeconds}s)`n" -ForegroundColor DarkGray

try {
    ssh $RemoteHost "bash ~/${RemoteSubfolder}/_run.sh"
    $trainExitCode = $LASTEXITCODE
}
finally {
    Stop-Job $pollJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $pollJob -ErrorAction SilentlyContinue | Out-Null
}

if ($trainExitCode -ne 0) {
    Write-Host "`nERROR: remote script exited with code $trainExitCode" -ForegroundColor Red
    exit $trainExitCode
}

# ── 4. Retrieve results ───────────────────────────────────────────────────────
# scp, not rsync — see note above the background poller.
Write-Host "`n==> [4/4] Retrieving out/, trained/, tmp/ ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ".\out"             | Out-Null
New-Item -ItemType Directory -Force -Path ".\trained\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path ".\tmp"             | Out-Null

scp -r "${RemoteHost}:~/${RemoteSubfolder}/out/*"               ".\out\"
scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/*.zip"       ".\trained\" 2>$null
scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/checkpoints/*" ".\trained\checkpoints\" 2>$null
scp -r "${RemoteHost}:~/${RemoteSubfolder}/tmp/*"                ".\tmp\"

Write-Host "`n==> All done." -ForegroundColor Green
Write-Host "`n--- out\ ---"
Get-ChildItem ".\out"     -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- trained\ ---"
Get-ChildItem ".\trained" -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- tmp\ ---"
Get-ChildItem ".\tmp"     -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime