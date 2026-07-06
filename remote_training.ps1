<#
.SYNOPSIS
    Copies the whole project to aurora-server, creates/uses a conda env,
    runs training (blocking, with live logs), then pulls back out/ and trained/.
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
$TrainCommand    = "python training.py"
$ExcludeNames    = @(".git", ".conda", ".vscode", "out", "trained", "results", "tmp",
                     "__pycache__", ".venv", "venv")
#############################################

# ── 1. Stage ─────────────────────────────────────────────────────────────────
Write-Host "`n==> [1/4] Staging local files..." -ForegroundColor Cyan
$StagingDir = Join-Path $env:TEMP "hdp_stage_$(Get-Random)"
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null
$roboArgs = @(".", $StagingDir, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
foreach ($ex in $ExcludeNames) { $roboArgs += @("/XD", $ex) }
robocopy @roboArgs | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

# ── 2. Upload ─────────────────────────────────────────────────────────────────
Write-Host "`n==> [2/4] Uploading to ${RemoteHost}:~/${RemoteSubfolder}/" -ForegroundColor Cyan
ssh $RemoteHost "mkdir -p ~/${RemoteSubfolder}/out ~/${RemoteSubfolder}/trained"
scp -r "$StagingDir\*" "${RemoteHost}:~/${RemoteSubfolder}/"
Remove-Item -Recurse -Force $StagingDir
Write-Host "    Upload complete."

# ── 3. Remote: setup + train (ONE ssh session, live output) ──────────────────
Write-Host "`n==> [3/4] Remote setup + training (live logs below)" -ForegroundColor Cyan
Write-Host "    SSH session open — do not close this window.`n" -ForegroundColor Yellow

# Write the remote bash script to a temp file so we avoid all heredoc
# escaping issues between PowerShell and bash
$remoteSh = Join-Path $env:TEMP "hdp_remote_$(Get-Random).sh"

# Use single-quoted here-string (@' '@) so PowerShell does NOT expand any $
# Then inject PS variables explicitly via string replacement afterwards
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
pip install torch numpy matplotlib "gymnasium[classic-control]" scipy

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

# Now inject PowerShell variable values into the placeholders
$bashScript = $bashScript.Replace("__SUBFOLDER__", $RemoteSubfolder)
$bashScript = $bashScript.Replace("__CONDAENV__",  $CondaEnvName)
$bashScript = $bashScript.Replace("__PYVER__",     $PythonVersion)
$bashScript = $bashScript.Replace("__TRAINCMD__",  $TrainCommand)

# Write as UTF-8 no BOM, Unix line endings
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$unixScript = $bashScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($remoteSh, $unixScript, $utf8NoBom)

scp "$remoteSh" "${RemoteHost}:~/${RemoteSubfolder}/_run.sh"
Remove-Item -Force $remoteSh
ssh $RemoteHost "bash ~/${RemoteSubfolder}/_run.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: remote script exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ── 4. Retrieve results ───────────────────────────────────────────────────────
Write-Host "`n==> [4/4] Retrieving out/ and trained/ ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ".\out"     | Out-Null
New-Item -ItemType Directory -Force -Path ".\trained" | Out-Null
New-Item -ItemType Directory -Force -Path ".\tmp" | Out-Null

scp -r "${RemoteHost}:~/${RemoteSubfolder}/out/*"     ".\out\"     2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "  WARN: out/ was empty." -ForegroundColor Yellow }

scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/*" ".\trained\" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "  WARN: trained/ was empty." -ForegroundColor Yellow }


scp -r "${RemoteHost}:~/${RemoteSubfolder}/tmp/*" ".\tmp\" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "  WARN: tmp/ was empty." -ForegroundColor Yellow }

Write-Host "`n==> All done." -ForegroundColor Green
Write-Host "`n--- out\ ---"
Get-ChildItem ".\out"     -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- trained\ ---"
Get-ChildItem ".\trained" -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- tmo\ ---"
Get-ChildItem ".\tmp" -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime