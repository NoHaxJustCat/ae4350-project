<#
.SYNOPSIS
    Read-only reattachment to whatever is already running on aurora-server.
    Unlike remote_training.ps1, this NEVER syncs code, launches training, or
    kills a tmux session — it only scp's trained/ and tmp/ back periodically
    and renders the same live dashboard. Safe to run/Ctrl+C repeatedly (e.g.
    after a local reboot or closed terminal) without touching remote state.
.USAGE
    .\watch_remote_training.ps1
#>

$ErrorActionPreference = "Stop"

$RemoteHost      = "nicolobasso@aurora-server"
$RemoteSubfolder = "hdp_training"
$PollIntervalSeconds = 20

New-Item -ItemType Directory -Force -Path ".\tmp"     | Out-Null
New-Item -ItemType Directory -Force -Path ".\trained" | Out-Null
$LocalTmpDir     = (Resolve-Path ".\tmp").Path
$LocalTrainedDir = (Resolve-Path ".\trained").Path

function Show-Dashboard {
    param($ConnectionWarning)
    Clear-Host
    Write-Host "Watching ${RemoteHost}:~/${RemoteSubfolder}  |  $(Get-Date -Format 'HH:mm:ss')  |  Ctrl+C stops watching only -- remote is untouched`n" -ForegroundColor Cyan
    if ($ConnectionWarning) {
        Write-Host "[reconnect] Lost contact with $RemoteHost on the last check -- retrying...`n" -ForegroundColor Yellow
    }

    $statusFiles = Get-ChildItem -Path $LocalTrainedDir -Filter "status.json" -Recurse -ErrorAction SilentlyContinue
    if (-not $statusFiles) {
        Write-Host "(no status.json synced yet)`n" -ForegroundColor DarkGray
        return
    }

    $rows = foreach ($f in $statusFiles) {
        try {
            $s = Get-Content $f.FullName -Raw | ConvertFrom-Json
            $sessionId = Split-Path (Split-Path $f.FullName -Parent) -Leaf
            if ($s.run_tag) { $sessionId = "$sessionId/$($s.run_tag)" }
            [PSCustomObject]@{
                Session     = $sessionId
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
    $rows | Sort-Object Session | Format-Table -AutoSize
}

Write-Host "Read-only remote watcher started. This never touches tmux/processes on the remote.`n" -ForegroundColor DarkGray

$connectionWarning = $false
try {
    while ($true) {
        scp -r "${RemoteHost}:~/${RemoteSubfolder}/tmp/*"     "$LocalTmpDir/"     2>$null
        scp -r "${RemoteHost}:~/${RemoteSubfolder}/trained/*" "$LocalTrainedDir/" 2>$null
        $connectionWarning = ($LASTEXITCODE -ne 0)
        Show-Dashboard -ConnectionWarning $connectionWarning
        Start-Sleep -Seconds $PollIntervalSeconds
    }
} finally {
    Write-Host "`nStopped watching (remote training, if any, keeps running untouched)." -ForegroundColor Green
}
