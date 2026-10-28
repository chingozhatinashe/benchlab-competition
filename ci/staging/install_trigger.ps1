# Installs (or with -Uninstall removes) the Windows scheduled task that runs
# trigger_cycle.ps1 every 10 min, so staging cycles start right after each t0
# (docs/staging.md). The script is copied to %LOCALAPPDATA%\benchlab-staging, so the
# task doesn't depend on which branch this checkout is on. It uses this user's `gh`
# login (needs the `workflow` scope) and runs only while the user is logged on.
#
#   powershell -ExecutionPolicy Bypass -File ci\staging\install_trigger.ps1
#   powershell -ExecutionPolicy Bypass -File ci\staging\install_trigger.ps1 -Uninstall
param(
    [switch]$Uninstall,
    [string]$TaskName = "Benchlab staging trigger"
)
$ErrorActionPreference = "Stop"
$dir = Join-Path $env:LOCALAPPDATA "benchlab-staging"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    "Removed the task '$TaskName' (state and log remain in $dir)"
    return
}

$gh = (Get-Command gh).Source
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item (Join-Path $PSScriptRoot "trigger_cycle.ps1") $dir -Force
$script = Join-Path $dir "trigger_cycle.ps1"

# conhost --headless: no console window flashing up every 10 min
$action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument (
    "--headless powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass " +
    "-File `"$script`" -Gh `"$gh`"")
# fires at :05, :15, ... so a cycle starts at t0 + 5 min (local midnight + 5 min is :05 UTC
# for a whole-hour UTC offset)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(5) `
    -RepetitionInterval (New-TimeSpan -Minutes 10)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force `
    -Description "Dispatches the benchlab-competition Staging cycle right after each 6-hourly t0" | Out-Null
"Installed '$TaskName': every 10 min, $script (log: $dir\trigger.log)"
