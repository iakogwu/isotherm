# Registers an hourly Windows scheduled task that runs the Isotherm logger.
# Runs as the current user while logged on; no admin rights needed.
# Remove with: scripts\uninstall_task.ps1
$ErrorActionPreference = "Stop"

$TaskName = "Isotherm Logger"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
$PythonW = Join-Path (Split-Path $Python) "pythonw.exe"   # no console window flashing
if (-not (Test-Path $PythonW)) { $PythonW = $Python }

$action = New-ScheduledTaskAction -Execute $PythonW -Argument "-m isotherm run" -WorkingDirectory $ProjectDir
# Start at :07 past the hour so freshly published model data and NBM text are available.
$start = (Get-Date).Date.AddHours((Get-Date).Hour + 1).AddMinutes(7)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 50) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Isotherm Phase 1: read-only logging of Kalshi temperature markets and forecasts" -Force | Out-Null

Write-Host "Registered '$TaskName' (hourly from $start) -> $PythonW -m isotherm run in $ProjectDir"
