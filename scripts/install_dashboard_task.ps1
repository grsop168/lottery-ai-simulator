[CmdletBinding()]
param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$name = 'Lottery Dashboard'
$root = 'C:\Users\wtjps\lottery-ai-simulator'
if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Stop-ScheduledTask -TaskName $name
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
    Write-Host "Task: $name (removed)"
    return
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\scripts\auto_dashboard.ps1`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity.Name
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999 -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
# Interactive token supports the requested login lifecycle without passwords or elevation.
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $name
$env:PYTHONPATH = Join-Path $root 'src'
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    & "$root\.venv\Scripts\python.exe" -m lottery_sim.dashboard_service --check
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
$task = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "Task: $($task.TaskName)"
Write-Host "State: $($task.State)"
Write-Host "Last Result: $($info.LastTaskResult)"
Write-Host "Trigger: AtLogOn ($($identity.Name))"
Write-Host "Retry: $($task.Settings.RestartInterval), count=$($task.Settings.RestartCount)"
Write-Host "MultipleInstances: $($task.Settings.MultipleInstances)"
Write-Host "ExecutionTimeLimit: $($task.Settings.ExecutionTimeLimit)"
if (-not $ready) { throw 'Dashboard health check failed; inspect reports\logs\dashboard-*.log' }
$response = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/' -UseBasicParsing -TimeoutSec 10
Write-Host "HTTP: $($response.StatusCode) http://127.0.0.1:8765/"
