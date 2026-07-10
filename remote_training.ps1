<#
.SYNOPSIS
    Syncs the project to aurora-server, creates/uses a conda env, launches
    training DETACHED in a remote tmux session (survives SSH disconnects —
    e.g. the local machine sleeping — instead of dying with the connection),
    live-tails its log with automatic reconnect, then pulls back out/ and
    trained/. While training runs, periodically pulls back the latest
    trajectory PNG and checkpoints.
.USAGE
    .\remote_training.ps1
#>

$ErrorActionPreference = "Stop"

#############################################
# CONFIG
#############################################
$RemoteHost      = "nicolobasso@aurora-server"
$RemoteSubfolder = "hdp_training"
$CondaEnvName    = "hdp-cw"
$PythonVersion   = "3.11"
$TmuxSession     = "hdp_training"
# Perf flags for the remote EPYC 7532 (32c/64t) Linux/RHEL box. History of
# what was tried, kept here so nobody re-derives this the hard way:
#   --net-arch 64,64  : right-sized net, ~1.9x locally, confirmed platform-
#                        agnostic win, keep always.
#   --vec-env dummy    : CONFIRMED on this machine (not just assumed from the
#                        local result) — SubprocVecEnv's real bottleneck isn't
#                        worker compute, it's step_async/step_wait in SB3:
#                        `for remote in self.remotes: remote.send(...)` then
#                        `[remote.recv() for remote in self.remotes]` — a
#                        SERIAL Python loop of n_envs pipe round-trips, every
#                        vec-step, in the main process. That's syscall/pickle
#                        overhead, not compute, so it (a) doesn't care how
#                        many cores exist and (b) gets WORSE as n_envs grows
#                        (n_envs=32->64 doubled the round-trips). This is why
#                        going 32->64 envs + more torch-threads didn't move
#                        steps/s and CPU stayed at ~8-13%. Switching to
#                        DummyVecEnv (single process, no IPC at all) measured
#                        ~335 steps/s steady-state vs ~290 for subproc, AND
#                        reached 100% dock rate + 2 curriculum advances
#                        (10m->15m->20m) within 14k timesteps in the same
#                        test where subproc was still stuck near 15m past
#                        100k. Don't switch back to subproc without a real
#                        head-to-head showing it actually wins here.
#   --n-envs 64        : fine to keep — DummyVecEnv still benefits from more
#                        envs per vec-step (more transitions collected =
#                        more gradient steps at UTD=1.0), and without
#                        per-worker IPC cost there's no penalty for a high
#                        count the way there was with subproc. Worth a sweep
#                        (32/64/128) later, not urgent.
#   --torch-threads    : left at default (4) here — no longer competing with
#                        32-64 separate worker processes for cores now that
#                        there's only one process total, so the earlier
#                        oversubscription logic doesn't apply. Still true
#                        that net_arch=[64,64] batch-256 matmuls are small
#                        enough that more threads may not help much; if you
#                        want to chase the remaining ~8-13% CPU number,
#                        sweep --torch-threads 2/4/8/16 and compare
#                        steps/s — but note CPU% staying low is not
#                        inherently a problem now, it may just reflect that
#                        this problem genuinely doesn't need 64 threads of
#                        compute. Watch dock-rate-vs-timesteps, not CPU%.
#   --vec-env batched  : (2026-07, REMOVED) BatchedCWVecEnv (libs/batched_env.py)
#                        stepped all 64 envs as ONE (64,4)@(4,4) numpy matmul
#                        per vec-step instead of DummyVecEnv's Python loop of
#                        64 .step() calls. Verified equivalent to the per-env
#                        stack to 1e-9 over 2570 episodes, and genuinely fast
#                        at env-stepping (17.1k -> 149.8k env-steps/s, 8.7x).
#                        But steady-state TRAINING throughput was measured at
#                        ~344 (batched) vs ~348 (dummy) steps/s — parity,
#                        because steady state is ~98% gradient updates (64
#                        sequential batch-256 updates per vec-step at UTD
#                        1.0), so env backend can't move it regardless. Kept
#                        initially as future-proofing ("useful if n_envs/UTD
#                        ever change"), but it duplicates every reward/physics
#                        formula in libs/env.py by hand — and that duplication
#                        caused a real bug (a reward-function edit silently
#                        not applying on remote runs, which use the batched
#                        path). Zero measured benefit for the actual training
#                        regime + a real maintenance/correctness cost that
#                        already bit once = removed. If a future config
#                        genuinely becomes env-step-bound, re-benchmark before
#                        reaching for this again rather than assuming it'll
#                        still be the fix.
#   "Only one core busy" during steady-state training is the CORRECT resting
#   state for this workload, not a bug: the gradient loop is inherently
#   serial and too small (batch-256, net_arch=[64,64]) for more threads to
#   help — DummyVecEnv's make_single_env() thunks run in the MAIN process and
#   accidentally pin torch to 1 thread there too, which measured faster than
#   more threads anyway (344 vs 312 steps/s at 4 threads).
#   --vec-env-start-method : irrelevant for dummy, no multiprocessing at all.
#                        Left unset.
#   --train-freq / --gradient-steps : deliberately NOT overridden — left at
#                        training.py's defaults (train_freq=1, gradient_steps
#                        =-1), a true 1 learning-update per 1 collected step.
#                        A sparser ratio (--train-freq 2 --gradient-steps 1)
#                        gave huge steps/s but ~117 real updates/sec and
#                        confirmed-fast-but-not-learning behavior on a real
#                        run. Re-derive gradient_steps/(train_freq*n_envs)
#                        before ever touching these again.
# Not enabled: --compile (torch.compile). More mature on Linux than Windows
# in general, but unverified here — needs a working C/C++ toolchain in the
# remote conda env and hasn't been benchmarked. Try it manually first:
#   python -u training.py --scenario vbar --n-envs 64 --vec-env dummy --compile --total-timesteps 20000
# To resume a checkpoint that got interrupted (SSH dropped before tmux
# detachment existed, machine rebooted, tmux session killed, etc.), add
# --resume-from trained/checkpoints/<name>_<steps>_steps.zip here — see
# training.py --help for details (curriculum_distance resumes automatically
# from that checkpoint's sidecar .curriculum.json if present).
$TrainCommand    = "python -u training.py --scenario vbar --n-envs 64 " +
                    "--net-arch 128,128 --vec-env dummy"
# $TrainCommand    = "python -u training.py --scenario vbar --n-envs 64 " +
#                     "--net-arch 64,64 --vec-env dummy"
$ExcludeNames    = @(".git", ".conda", ".vscode", "out", "trained", "results", "tmp",
                     "__pycache__", ".venv", "venv")
$PollIntervalSeconds     = 20
$ReconnectWaitSeconds    = 15
#############################################

# ── 1. Sync ───────────────────────────────────────────────────────────────────
Write-Host "`n==> [1/5] Syncing to ${RemoteHost}:~/${RemoteSubfolder}/" -ForegroundColor Cyan
ssh $RemoteHost "mkdir -p ~/${RemoteSubfolder}/out ~/${RemoteSubfolder}/trained"

$StagingDir = Join-Path $env:TEMP "hdp_stage"
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

$roboArgs = @(".", $StagingDir, "/E", "/XO", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
foreach ($ex in $ExcludeNames) { $roboArgs += @("/XD", $ex) }
robocopy @roboArgs | Out-Null

scp -r "$StagingDir\*" "${RemoteHost}:~/${RemoteSubfolder}/"
Remove-Item -Recurse -Force $StagingDir
Write-Host "    Sync complete."

# ── 2. Remote: setup + launch training (detached) ─────────────────────────────
# Split into two remote scripts on purpose:
#   _run.sh         : conda setup (blocking, foreground, usually fast) then
#                      launches _train_inner.sh INSIDE a detached tmux
#                      session and returns immediately.
#   _train_inner.sh : the actual conda-activate + $TrainCommand + writes
#                      train.done with the exit code when finished.
# Previously $TrainCommand ran directly in _run.sh's foreground, which ran
# directly in this script's ssh session — so the instant that SSH connection
# dropped (e.g. the local machine sleeping), the remote process got SIGHUP'd
# and died. That's exactly what happened to a run 76.6% of the way through
# 3M timesteps. tmux detaches the actual training process from this SSH
# session entirely: the connection can drop and reconnect (or drop for
# good) without touching it.
Write-Host "`n==> [2/5] Remote setup + launching training in detached tmux session '$TmuxSession'" -ForegroundColor Cyan

$remoteInnerSh = Join-Path $env:TEMP "hdp_remote_inner_$(Get-Random).sh"
$innerScript = @'
#!/usr/bin/env bash
REMOTE_DIR=~/__SUBFOLDER__
CONDA_ENV=__CONDAENV__

CONDA_SH=""
for c in /opt/miniconda3/etc/profile.d/conda.sh \
          ~/miniconda3/etc/profile.d/conda.sh \
          ~/anaconda3/etc/profile.d/conda.sh \
          /opt/conda/etc/profile.d/conda.sh \
          /usr/local/conda/etc/profile.d/conda.sh; do
    if [ -f "$c" ]; then CONDA_SH="$c"; break; fi
done
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REMOTE_DIR"

echo "========================================="
echo " Training started  -- $(date)"
echo "========================================="
__TRAINCMD__
echo $? > "$REMOTE_DIR/train.done"
echo "========================================="
echo " Training finished -- $(date)"
echo "========================================="
'@
$innerScript = $innerScript.Replace("__SUBFOLDER__", $RemoteSubfolder)
$innerScript = $innerScript.Replace("__CONDAENV__",  $CondaEnvName)
$innerScript = $innerScript.Replace("__TRAINCMD__",  $TrainCommand)

$remoteSh = Join-Path $env:TEMP "hdp_remote_$(Get-Random).sh"
$bashScript = @'
#!/usr/bin/env bash
set -euo pipefail
trap 'echo "REMOTE ERROR at line $LINENO (exit $?)" >&2' ERR

REMOTE_DIR=~/__SUBFOLDER__
CONDA_ENV=__CONDAENV__
PYTHON_VER=__PYVER__
SESSION=__TMUXSESSION__

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

cd "$REMOTE_DIR"
rm -rf tmp
mkdir -p out trained tmp
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -f train.log train.done
chmod +x "$REMOTE_DIR/_train_inner.sh"
tmux new-session -d -s "$SESSION" \
    "bash '$REMOTE_DIR/_train_inner.sh' > '$REMOTE_DIR/train.log' 2>&1"
echo "[setup] Training launched in detached tmux session '$SESSION' -- safe to disconnect now."
'@

$bashScript = $bashScript.Replace("__SUBFOLDER__", $RemoteSubfolder)
$bashScript = $bashScript.Replace("__CONDAENV__",  $CondaEnvName)
$bashScript = $bashScript.Replace("__PYVER__",     $PythonVersion)
$bashScript = $bashScript.Replace("__TMUXSESSION__", $TmuxSession)

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($remoteInnerSh, ($innerScript -replace "`r`n", "`n"), $utf8NoBom)
[System.IO.File]::WriteAllText($remoteSh,      ($bashScript  -replace "`r`n", "`n"), $utf8NoBom)

scp "$remoteInnerSh" "${RemoteHost}:~/${RemoteSubfolder}/_train_inner.sh"
scp "$remoteSh"      "${RemoteHost}:~/${RemoteSubfolder}/_run.sh"
Remove-Item -Force $remoteInnerSh, $remoteSh

ssh $RemoteHost "bash ~/${RemoteSubfolder}/_run.sh"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: remote setup script exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path ".\tmp" | Out-Null
$LocalTmpDir = (Resolve-Path ".\tmp").Path
New-Item -ItemType Directory -Force -Path ".\trained\checkpoints" | Out-Null
$LocalTrainedDir = (Resolve-Path ".\trained").Path

# ── 3. Background poller ──────────────────────────────────────────────────────
# Also periodically pulls trained/checkpoints/ back (training.py now saves a
# checkpoint every --checkpoint-freq timesteps, plus a curriculum sidecar
# .json needed to --resume-from it correctly). This is the actual progress
# safety net now — training itself survives disconnects via tmux, but this
# is still what gets checkpoints onto your local disk periodically.
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

# ── 4. Live-tail training.log with automatic reconnect ────────────────────────
# Training itself is safe (detached in tmux) regardless of what this loop
# does — this is purely for live visibility. `ssh ... tail -f` blocks until
# either the connection drops or the remote shell exits; either way, when it
# returns we check train.done on the remote to tell "training actually
# finished" apart from "just got disconnected," and reconnect in the latter
# case instead of giving up. First attempt tails from the start of the file
# (matches old behavior); reconnects only show recent context, not a full
# replay of a potentially huge log.
Write-Host "==> [3/5] Watching training.log (Ctrl+C stops watching only — training keeps running remotely)`n" -ForegroundColor Cyan

$trainingDone = $false
$firstAttempt = $true
try {
    while (-not $trainingDone) {
        if ($firstAttempt) {
            ssh $RemoteHost "tail -n +1 -f ~/${RemoteSubfolder}/train.log"
            $firstAttempt = $false
        } else {
            ssh $RemoteHost "tail -n 40 -f ~/${RemoteSubfolder}/train.log"
        }
        # tail -f only returns here if the SSH connection dropped or the
        # remote shell exited — either way, check whether training actually
        # finished (train.done written) before deciding to reconnect.
        $doneCheck = ssh -o ConnectTimeout=10 $RemoteHost `
            "test -f ~/${RemoteSubfolder}/train.done && echo DONE" 2>$null
        if ($doneCheck -match "DONE") {
            $trainingDone = $true
        } else {
            Write-Host "`n[reconnect] Connection to $RemoteHost lost -- training continues in the remote tmux session '$TmuxSession' regardless. Reconnecting in ${ReconnectWaitSeconds}s...`n" -ForegroundColor Yellow
            Start-Sleep -Seconds $ReconnectWaitSeconds
        }
    }
}
finally {
    Stop-Job $pollJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $pollJob -ErrorAction SilentlyContinue | Out-Null
}

$trainExitCodeRaw = ssh $RemoteHost "cat ~/${RemoteSubfolder}/train.done" 2>$null
$trainExitCode = 1
[int]::TryParse($trainExitCodeRaw, [ref]$trainExitCode) | Out-Null

if ($trainExitCode -ne 0) {
    Write-Host "`nERROR: remote training exited with code $trainExitCode (see train.log)" -ForegroundColor Red
    exit $trainExitCode
}

# ── 5. Retrieve results ───────────────────────────────────────────────────────
# scp, not rsync — see note above the background poller.
Write-Host "`n==> [5/5] Retrieving out/, trained/, tmp/ ..." -ForegroundColor Cyan
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
