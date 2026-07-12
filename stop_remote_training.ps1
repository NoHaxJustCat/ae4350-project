<#
.SYNOPSIS
    Actually stops a training run launched by remote_training.ps1 — Ctrl+C
    on that script only stops watching (by design, so a dropped/closed SSH
    session doesn't kill training); this kills the remote tmux session AND
    any training.py processes still running, in case some escaped the tmux
    process group (e.g. background jobs launched with `&` inside a sweep).
.USAGE
    .\stop_remote_training.ps1          # asks for confirmation first
    .\stop_remote_training.ps1 -Force   # skips the confirmation prompt
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Must match remote_training.ps1's CONFIG section.
$RemoteHost      = "nicolobasso@aurora-server"
$RemoteSubfolder = "hdp_training"
$TmuxSession     = "hdp_training"

if (-not $Force) {
    $confirm = Read-Host "Stop remote training on ${RemoteHost} (tmux session '$TmuxSession')? [y/N]"
    if ($confirm -notmatch "^[yY]") {
        Write-Host "Aborted -- remote training left running." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "`n==> Stopping tmux session '$TmuxSession' and any training.py processes on ${RemoteHost}..." -ForegroundColor Cyan

# tmux kill-session sends SIGHUP to the pane; that alone isn't guaranteed to
# take down background (`&`-launched) child processes a sweep's
# $TrainCommand starts (non-interactive bash doesn't forward SIGHUP to
# background jobs on exit unless `huponexit` is set) — so pkill explicitly
# too rather than assuming the tmux kill was enough.
$remoteCmd = "tmux kill-session -t '$TmuxSession' 2>/dev/null; " +
             "pkill -f 'python -u training.py' 2>/dev/null; " +
             "echo done"
ssh $RemoteHost $remoteCmd | Out-Null

# Verify: neither the tmux session nor any training.py process should
# still be alive.
$tmuxCheck = ssh $RemoteHost "tmux has-session -t '$TmuxSession' 2>/dev/null && echo ALIVE" 2>$null
$procCheck = ssh $RemoteHost "pgrep -f 'python -u training.py' 2>/dev/null" 2>$null

if ($tmuxCheck -match "ALIVE" -or $procCheck) {
    Write-Host "`nWARNING: something may still be running -- tmux alive: $([bool]($tmuxCheck -match 'ALIVE')), leftover training.py PIDs: $procCheck" -ForegroundColor Red
    Write-Host "SSH in manually to double check: ssh $RemoteHost" -ForegroundColor Red
} else {
    Write-Host "`nConfirmed stopped: no '$TmuxSession' tmux session, no training.py processes remaining on ${RemoteHost}." -ForegroundColor Green
}
