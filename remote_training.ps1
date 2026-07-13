<#
.SYNOPSIS
    Syncs the project to aurora-server, creates/uses a conda env, launches
    training DETACHED in a remote tmux session (survives SSH disconnects —
    e.g. the local machine sleeping — instead of dying with the connection),
    shows a live status dashboard (one row per run, refreshed periodically)
    instead of raw log tailing, then pulls back out/ and trained/. While
    training runs, periodically pulls back the latest trajectory PNGs,
    checkpoints, diagnostics plots, and status.json files.

    To actually stop a detached remote run (Ctrl+C here only stops
    watching — see the dashboard message), use .\stop_remote_training.ps1.
.USAGE
    .\remote_training.ps1
#>

$ErrorActionPreference = "Stop"

# Always operate from the script's own directory. Without this the sync in
# step 1 (robocopy of ".") depends on the caller's CWD — if the script is
# invoked from anywhere else (e.g. a background launcher that doesn't inherit
# the shell's cd), robocopy copies an empty/wrong tree, staging comes up
# empty, and the run silently trains STALE remote code. Pin CWD here so "."
# is always the project root.
Set-Location -LiteralPath $PSScriptRoot

#############################################
# CONFIG
#############################################
$RemoteHost      = "nicolobasso@aurora-server"
$RemoteSubfolder = "hdp_training"
$CondaEnvName    = "hdp-cw"
$PythonVersion   = "3.11"
$TmuxSession     = "hdp_training"
# Shared across every run in $TrainCommand below (via --session-id) so a
# sweep's N runs land together under trained/<SessionId>/<run-tag>/ instead
# of each picking its own timestamp. Also what the dashboard (step 3) reads
# to find their status.json files.
$SessionId       = Get-Date -Format "yyyyMMdd_HHmmss"
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
#   Async actor/learner split (Ape-X-style): (2026-07, INVESTIGATED, NOT
#                        BUILT) proposed as N worker processes stepping
#                        envs into a shared replay buffer while 1 learner
#                        process trains continuously off it (most cores
#                        simulate, one trains). Rejected without building
#                        it: env-stepping is only ~2% of steady-state
#                        wall-clock time here — the BatchedCWVecEnv result
#                        above is the direct experiment for "what if
#                        simulation were free" (8.7x faster stepping moved
#                        overall steps/s by 0%), so even a PERFECT overlap
#                        of simulation with gradient compute caps out
#                        around a ~2% win. The actual bottleneck (64
#                        sequential single-core TD3 gradient steps per
#                        vec-step) is a dependent chain — each step needs
#                        the previous step's updated weights — not
#                        independent parallel work, so it can't be split
#                        across cores the way env-stepping can. Splitting
#                        a single gradient step's own compute across
#                        threads already lost to fewer threads (see 1
#                        torch thread beating 4 above). A real multi-
#                        writer buffer would also need a from-scratch
#                        rewrite onto shared memory (SB3's ReplayBuffer is
#                        plain numpy + a non-atomic pos counter, single-
#                        process only) for that ~2% ceiling — not worth
#                        it. Don't re-propose this unless the env itself
#                        becomes genuinely expensive to step (a much
#                        heavier physics model), which would change the
#                        98/2 split this conclusion rests on.
# Not enabled: --compile (torch.compile). More mature on Linux than Windows
# in general, but unverified here — needs a working C/C++ toolchain in the
# remote conda env and hasn't been benchmarked. Try it manually first:
#   python -u training.py --scenario vbar --n-envs 64 --vec-env dummy --compile --total-timesteps 20000
# To resume a checkpoint that got interrupted (SSH dropped before tmux
# detachment existed, machine rebooted, tmux session killed, etc.), add
# --resume-from trained/checkpoints/<name>_<steps>_steps.zip here — see
# training.py --help for details (curriculum_distance resumes automatically
# from that checkpoint's sidecar .curriculum.json if present).
# $TrainCommand    = "python -u training.py --scenario vbar --n-envs 64 " +
#                     "--net-arch 128,128,64 --vec-env dummy --torch-threads 16"
# $TrainCommand    = "python -u training.py --scenario vbar --n-envs 64 " +
#                     "--net-arch 64,64 --vec-env dummy"
#
# UTD-ratio x seed sweep (2026-07, TRIED, REVERTED): launched 24 independent
# single-threaded processes (3 UTD levels x 8 seeds) concurrently to search
# hyperparameters/seeds in parallel — genuine multi-core use (unlike
# multithreading one run, already disproven above), but it produces 24
# separate candidate models, and each one is somewhat SLOWER than a solo
# run because all 24 share the same ~32-64 cores. Reverted because the
# actual near-term need is one model trained as fast as a single run can
# go, not 24 explored-in-parallel-but-individually-slower ones. Kept here
# for when parallel hyperparameter search is actually wanted again:
#   $TrainCommand = @'
#   SEEDS=(0 1 2 3 4 5 6 7)
#   UTD_LEVELS=(1 2 4)
#   PIDS=()
#   for UTD in "${UTD_LEVELS[@]}"; do
#     GSTEPS=$((UTD * 64))
#     for SEED in "${SEEDS[@]}"; do
#       TAG="utd${UTD}_seed${SEED}"
#       python -u training.py --scenario vbar --n-envs 64 \
#         --net-arch 128,128,64 --vec-env dummy --torch-threads 1 \
#         --train-freq 1 --gradient-steps "$GSTEPS" --seed "$SEED" \
#         --session-id "$SESSION_ID" --run-tag "$TAG" \
#         > "train_${TAG}.log" 2>&1 &
#       PIDS+=($!)
#     done
#   done
#   echo "Launched ${#PIDS[@]} parallel runs (UTD x seed), session_id=$SESSION_ID: ${PIDS[*]}"
#   for PID in "${PIDS[@]}"; do
#     wait "$PID"
#   done
#   '@
#
# Net-architecture sweep (2026-07, CURRENT): after retuning the reward for
# fuel efficiency (gamma 0.9995->0.9999 + a steeper fuel bonus
# 25*(dv_used+0.01)^-1 — see libs/constants.py & libs/env.py), sweep 5
# plain-MLP capacities head-to-head UNDER THE SAME new reward to see which
# depth/width best exploits it toward the optimal (~0.0115 m/s) V-bar
# transfer instead of the fuel-wasteful ~1.53 m/s brute-force dock.
#   64,64 / 128,128 / 64,64,64 / 128,128,128 / 256,256
# All CPU, launched in parallel inside the one detached tmux session, each
# in its own trained/<session>/arch_<a>_<b>/ (via --run-tag) so the live
# dashboard shows one row per arch. torch-threads 1 + vec-env dummy per run
# (same rationale as the reverted UTD x seed sweep above): single-process,
# no IPC, and the EPYC box has plenty of cores for 5 concurrent 1-thread
# gradient loops. Everything except net_arch is held identical so the
# comparison is clean. (For the LayerNorm arch, add --arch smart --n-blocks 2
# to a run — see libs/policies.py — but this sweep is the plain-MLP baseline.)
$TrainCommand = @'
ARCHS=("64,64" "128,128" "64,64,64" "128,128,128" "256,256")
PIDS=()
for ARCH in "${ARCHS[@]}"; do
  TAG="arch_${ARCH//,/_}"
  python -u training.py --scenario vbar --n-envs 32 \
    --net-arch "$ARCH" --vec-env dummy --torch-threads 1 \
    --session-id "$SESSION_ID" --run-tag "$TAG" \
    --arch smart --gamma 0.99 --lr 1e-4 --total-timesteps 20000000 --checkpoint-freq 100000 --keep-last-checkpoints 2 \
    > "train_${TAG}.log" 2>&1 &
  PIDS+=($!)
done
echo "Launched ${#PIDS[@]} net-arch runs, session_id=$SESSION_ID: ${PIDS[*]}"
for PID in "${PIDS[@]}"; do wait "$PID"; done
'@
$ExcludeNames    = @(".git", ".conda", ".vscode", "out", "trained", "results", "tmp",
                     "__pycache__", ".venv", "venv")
$PollIntervalSeconds     = 20
#############################################

# ── 1. Sync ───────────────────────────────────────────────────────────────────
Write-Host "`n==> [1/5] Syncing to ${RemoteHost}:~/${RemoteSubfolder}/" -ForegroundColor Cyan
ssh $RemoteHost "mkdir -p ~/${RemoteSubfolder}/out ~/${RemoteSubfolder}/trained"

$StagingDir = Join-Path $env:TEMP "hdp_stage"
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

$roboArgs = @(".", $StagingDir, "/E", "/XO", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
foreach ($ex in $ExcludeNames) { $roboArgs += @("/XD", $ex) }
robocopy @roboArgs | Out-Null
# robocopy exit codes 0-7 are success (1 = files copied); >=8 is a real
# failure. Anything that leaves staging empty means nothing will transfer —
# fail loudly here instead of scp'ing nothing and training stale remote code.
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE); nothing staged." }
$stagedCount = (Get-ChildItem -Recurse -File $StagingDir | Measure-Object).Count
if ($stagedCount -eq 0) {
    $cwd = Get-Location
    throw "Staging dir is empty after robocopy (CWD=$cwd). Refusing to sync an empty tree; the remote would keep running stale code."
}
Write-Host "    Staged $stagedCount files."

# Transfer: scp the staging dir as ONE directory. Windows OpenSSH scp does
# NOT expand a local "dir\*" glob — it stat()s a literal "*" and fails 255
# (this is why the old `scp -r "$StagingDir\*"` silently transferred nothing
# and the remote kept training stale code). Land it in a FRESH remote temp
# dir so scp -r never nests into an existing remote subdir (libs -> libs/libs),
# then merge the contents into place with cp on the remote — overwrites
# changed files, adds new ones, no nesting.
$remoteStage = "~/${RemoteSubfolder}_stage"
ssh $RemoteHost "rm -rf $remoteStage && mkdir -p $remoteStage"
if ($LASTEXITCODE -ne 0) { throw "failed to prepare remote staging dir (exit $LASTEXITCODE)." }
scp -r "$StagingDir" "${RemoteHost}:$remoteStage/src"
if ($LASTEXITCODE -ne 0) { throw "scp of staged tree failed (exit $LASTEXITCODE)." }
ssh $RemoteHost "cp -rf $remoteStage/src/. ~/${RemoteSubfolder}/ && rm -rf $remoteStage"
if ($LASTEXITCODE -ne 0) { throw "remote merge of staged tree failed (exit $LASTEXITCODE)." }
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
rm -f train.log train.done train_*.log
chmod +x "$REMOTE_DIR/_train_inner.sh"
tmux new-session -d -s "$SESSION" \
    "SESSION_ID='__SESSIONID__' bash '$REMOTE_DIR/_train_inner.sh' > '$REMOTE_DIR/train.log' 2>&1"
echo "[setup] Training launched in detached tmux session '$SESSION' -- safe to disconnect now."
'@

$bashScript = $bashScript.Replace("__SUBFOLDER__", $RemoteSubfolder)
$bashScript = $bashScript.Replace("__CONDAENV__",  $CondaEnvName)
$bashScript = $bashScript.Replace("__PYVER__",     $PythonVersion)
$bashScript = $bashScript.Replace("__TMUXSESSION__", $TmuxSession)
$bashScript = $bashScript.Replace("__SESSIONID__", $SessionId)

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
New-Item -ItemType Directory -Force -Path ".\trained" | Out-Null
$LocalTrainedDir = (Resolve-Path ".\trained").Path

# ── 3. Background poller + live dashboard ──────────────────────────────────────
# Pulls the WHOLE trained/$SessionId/ tree back periodically (not just
# checkpoints/) — training.py now writes checkpoints, the final model,
# diagnostics.png, history.npz, AND a small status.json all under
# trained/<session-id>/<run-tag>/ (see training.py), and the dashboard
# below needs those status.json files synced locally to render live
# progress. Also still pulls tmp/ (per-run latest_trajectory.png).
#
# Uses `scp -r ".../dir/*" "local\dir\"` rather than rsync — rsync isn't
# installed on this machine (only OpenSSH's ssh/scp are). Unlike
# `rsync -az src/ dst/`, plain `scp -r src dst` does NOT merge into an
# existing dst — a second run would nest as dst\src\src\. Globbing the
# source (`src/*`) copies the *contents* into dst instead, which is safe to
# repeat every poll. 2>$null because early polls hit an empty/missing
# session dir before the first checkpoint is written — expected, not an
# error.
$pollJob = Start-Job -ScriptBlock {
    param($RemoteHost, $RemoteSubfolder, $LocalTmpDir, $LocalTrainedDir, $Interval)
    while ($true) {
        scp -r "${RemoteHost}:~/${RemoteSubfolder}/tmp/*" `
            "$LocalTmpDir/" 2>$null
        scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/*" `
            "$LocalTrainedDir/" 2>$null
        Start-Sleep -Seconds $Interval
    }
} -ArgumentList $RemoteHost, $RemoteSubfolder, $LocalTmpDir, $LocalTrainedDir, $PollIntervalSeconds

Write-Host "    Live trajectory + trained/ polling started -> .\tmp\, .\trained\ (every ${PollIntervalSeconds}s)`n" -ForegroundColor DarkGray

# ── 4. Live status dashboard (replaces raw log tailing) ────────────────────────
# Used to `ssh ... tail -f train.log` here — with a 24-run sweep, that's 24
# processes' per-episode print lines interleaved in one stream, which is
# unreadable as an "overall progress" view (each run still writes its own
# train_<tag>.log on the remote if you want that per-run detail — see step
# 5). Instead, render one summary row per run from the status.json files
# the background poller (step 3) already syncs locally, refreshed on the
# same cadence. Doesn't need its own SSH reconnect handling for the
# display itself (it's reading already-local files); only the periodic
# train.done check touches the network, tolerantly (failures just mean
# "check again next cycle," same as before).
Write-Host "==> [3/5] Live dashboard for session $SessionId (Ctrl+C stops watching only -- training keeps running remotely; use .\stop_remote_training.ps1 to actually stop it)`n" -ForegroundColor Cyan

function Show-Dashboard {
    param($SessionDir, $SessionId, $ConnectionWarning)
    Clear-Host
    Write-Host "Session: $SessionId  |  $(Get-Date -Format 'HH:mm:ss')  |  Ctrl+C stops watching only; .\stop_remote_training.ps1 actually stops remote training`n" -ForegroundColor Cyan
    if ($ConnectionWarning) {
        Write-Host "[reconnect] Lost contact with $RemoteHost on the last check -- training continues in the remote tmux session '$TmuxSession' regardless; retrying...`n" -ForegroundColor Yellow
    }

    $statusFiles = Get-ChildItem -Path $SessionDir -Filter "status.json" -Recurse -ErrorAction SilentlyContinue
    if (-not $statusFiles) {
        Write-Host "(no status.json synced yet -- waiting on first diagnostics update from the remote run(s))`n" -ForegroundColor DarkGray
        return
    }

    $rows = foreach ($f in $statusFiles) {
        try {
            $s = Get-Content $f.FullName -Raw | ConvertFrom-Json
            [PSCustomObject]@{
                RunTag      = if ($s.run_tag) { $s.run_tag } else { "(single run)" }
                Progress    = "{0:N0}/{1:N0}" -f $s.num_timesteps, $s.total_timesteps
                Episodes    = $s.episode_count
                DockRate    = if ($null -ne $s.recent_dock_rate) { "{0:P0}" -f $s.recent_dock_rate } else { "n/a" }
                AvgReward   = if ($null -ne $s.recent_avg_reward) { "{0:N2}" -f $s.recent_avg_reward } else { "n/a" }
                CurDist     = if ($null -ne $s.curriculum_distance) { "{0:N1}" -f $s.curriculum_distance } else { "n/a" }
                StepsPerSec = if ($null -ne $s.steps_per_sec) { "{0:N1}" -f $s.steps_per_sec } else { "n/a" }
                Updated     = $s.updated_at
            }
        } catch { }
    }
    $rows | Sort-Object RunTag | Format-Table -AutoSize
}

$trainingDone = $false
$connectionWarning = $false
$SessionDir = Join-Path $LocalTrainedDir $SessionId
try {
    while (-not $trainingDone) {
        Show-Dashboard -SessionDir $SessionDir -SessionId $SessionId -ConnectionWarning $connectionWarning
        $doneCheck = ssh -o ConnectTimeout=10 $RemoteHost `
            "test -f ~/${RemoteSubfolder}/train.done && echo DONE" 2>$null
        # ssh itself exits 255 specifically for a connection-level failure
        # (can't resolve/connect/auth) vs. passing through the remote
        # command's own exit status otherwise — `test -f ... && echo DONE`
        # legitimately exits non-zero (1) every cycle before training.done
        # exists, which is the normal/expected case, not a dropped
        # connection, so only 255 should count as "lost contact."
        $connectionWarning = ($LASTEXITCODE -eq 255)
        if ($doneCheck -match "DONE") {
            $trainingDone = $true
        } else {
            Start-Sleep -Seconds $PollIntervalSeconds
        }
    }
    Show-Dashboard -SessionDir $SessionDir -SessionId $SessionId
    Write-Host "Training finished remotely.`n" -ForegroundColor Green
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
Write-Host "`n==> [5/5] Retrieving out/, trained/, tmp/, per-run logs ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ".\out"                          | Out-Null
New-Item -ItemType Directory -Force -Path ".\trained"                     | Out-Null
New-Item -ItemType Directory -Force -Path ".\tmp"                         | Out-Null
New-Item -ItemType Directory -Force -Path ".\trained\$SessionId\logs"     | Out-Null

scp -r "${RemoteHost}:~/${RemoteSubfolder}/out/*"       ".\out\"     2>$null
scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/*"   ".\trained\" 2>$null
scp -r "${RemoteHost}:~/${RemoteSubfolder}/tmp/*"       ".\tmp\"     2>$null
# Per-run detailed logs (train_<tag>.log for a sweep, just the coordinator
# train.log for a plain single run) — not synced by the periodic poller,
# archived here alongside that session's checkpoints/diagnostics/history.
scp "${RemoteHost}:~/${RemoteSubfolder}/train_*.log" ".\trained\$SessionId\logs\" 2>$null
scp "${RemoteHost}:~/${RemoteSubfolder}/train.log"    ".\trained\$SessionId\logs\" 2>$null

Write-Host "`n==> All done." -ForegroundColor Green
Write-Host "`n--- out\ ---"
Get-ChildItem ".\out"     -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- trained\ ---"
Get-ChildItem ".\trained" -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
Write-Host "--- tmp\ ---"
Get-ChildItem ".\tmp"     -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
