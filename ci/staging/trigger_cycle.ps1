# On-time trigger for the staging cycle (docs/staging.md). GitHub's cron starts the cycles
# hours late, so Windows Task Scheduler runs this every 10 min (install_trigger.ps1). It
# dispatches the Staging workflow once per t0, as soon as t0 + WaitMinutes has passed; if the
# laptop was asleep, it catches up on wake. The workflow skips a t0 that is already recorded,
# so GitHub's own cron stays as the backup.
# It also dispatches the Benchlab monitor (monitor.yml, mode=record) once per t0, at
# t0 + MonitorMinutes, after Benchlab's run; GitHub's cron for it often drops passes.
param(
    [string]$Gh = "gh",
    [string]$Repo = "chingozhatinashe/benchlab-competition",
    [string]$StateDir = (Join-Path $env:LOCALAPPDATA "benchlab-staging"),
    [int]$WaitMinutes = 5,  # Benchlab starts "a few minutes" after t0
    [int]$MonitorMinutes = 30,  # Benchlab's run is done by ~t0 + 7 min
    [switch]$DryRun
)

New-Item -ItemType Directory -Force $StateDir | Out-Null
$log = Join-Path $StateDir "trigger.log"
function Write-Log([string]$msg) {
    Add-Content -Path $log -Encoding utf8 -Value ("{0:yyyy-MM-ddTHH:mm:ssZ} {1}" -f [DateTime]::UtcNow, $msg)
}

$now = [DateTime]::UtcNow
$t0Time = $now.Date.AddHours([Math]::Floor($now.Hour / 6) * 6)
$t0 = $t0Time.ToString("yyyyMMdd'T'HH'0000Z'")

# Dispatches a workflow once per t0, once t0 + $Minutes has passed; returns $false on failure.
function Send-Once([string]$Name, [string]$StateFile, [int]$Minutes, [string[]]$Arguments) {
    if (($now - $t0Time).TotalMinutes -lt $Minutes) { return $true }
    $state = Join-Path $StateDir $StateFile
    $last = ""
    if (Test-Path $state) { $last = ([string](Get-Content $state -TotalCount 1)).Trim() }
    if ($last -eq $t0) { return $true }  # already dispatched
    if ($DryRun) { Write-Host "would dispatch $Name $t0 (last dispatched: '$last')"; return $true }
    $ErrorActionPreference = "Continue"  # gh writes progress to stderr
    $out = & $Gh workflow run @Arguments --repo $Repo --ref main 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        Set-Content -Path $state -Value $t0 -Encoding ascii
        Write-Log "dispatched $Name $t0"
        return $true
    }
    Write-Log ("dispatch of $Name $t0 failed (exit $LASTEXITCODE): " + ($out -replace "\s+", " ").Trim())
    return $false
}

$ok = Send-Once "cycle" "last_t0.txt" $WaitMinutes @("staging.yml", "-f", "job=cycle", "-f", "t0=$t0")
$ok = (Send-Once "monitor" "last_monitor_t0.txt" $MonitorMinutes @("monitor.yml", "-f", "mode=record")) -and $ok
if (-not $ok) { exit 1 }
