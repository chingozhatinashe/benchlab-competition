# On-time trigger for the staging cycle (docs/staging.md). GitHub's cron starts the cycles
# hours late, so Windows Task Scheduler runs this every 10 min (install_trigger.ps1). It
# dispatches the Staging workflow once per t0, as soon as t0 + WaitMinutes has passed; if the
# laptop was asleep, it catches up on wake. The workflow skips a t0 that is already recorded,
# so GitHub's own cron stays as the backup.
param(
    [string]$Gh = "gh",
    [string]$Repo = "chingozhatinashe/benchlab-competition",
    [string]$StateDir = (Join-Path $env:LOCALAPPDATA "benchlab-staging"),
    [int]$WaitMinutes = 5,  # Benchlab starts "a few minutes" after t0
    [switch]$DryRun
)

New-Item -ItemType Directory -Force $StateDir | Out-Null
$state = Join-Path $StateDir "last_t0.txt"
$log = Join-Path $StateDir "trigger.log"
function Write-Log([string]$msg) {
    Add-Content -Path $log -Encoding utf8 -Value ("{0:yyyy-MM-ddTHH:mm:ssZ} {1}" -f [DateTime]::UtcNow, $msg)
}

$now = [DateTime]::UtcNow
$t0Time = $now.Date.AddHours([Math]::Floor($now.Hour / 6) * 6)
$t0 = $t0Time.ToString("yyyyMMdd'T'HH'0000Z'")
if (($now - $t0Time).TotalMinutes -lt $WaitMinutes) { exit 0 }

$last = ""
if (Test-Path $state) { $last = ([string](Get-Content $state -TotalCount 1)).Trim() }
if ($last -eq $t0) { exit 0 }  # already dispatched

if ($DryRun) { "would dispatch cycle $t0 (last dispatched: '$last')"; exit 0 }

$ErrorActionPreference = "Continue"  # gh writes progress to stderr
$out = & $Gh workflow run staging.yml --repo $Repo --ref main -f job=cycle -f "t0=$t0" 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Set-Content -Path $state -Value $t0 -Encoding ascii
    Write-Log "dispatched cycle $t0"
} else {
    Write-Log ("dispatch of cycle $t0 failed (exit $LASTEXITCODE): " + ($out -replace "\s+", " ").Trim())
    exit 1
}
