[CmdletBinding()]
param(
    [switch]$Remove,
    [ValidateSet('PL5', 'Dashboard', 'All')][string]$Task = 'PL5'
)
$ErrorActionPreference = 'Stop'
if ($Task -eq 'Dashboard') {
    & (Join-Path $PSScriptRoot 'install_dashboard_task.ps1') -Remove:$Remove
    return
}
if ($Task -eq 'All') {
    & (Join-Path $PSScriptRoot 'install_pl5_task.ps1') -Task Dashboard -Remove:$Remove
}
$name = 'Lottery PL5 Daily'
$root = 'C:\Users\wtjps\lottery-ai-simulator'
if ($Remove) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
    Write-Host "Task: $name (removed)"
    return
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\auto_pl5.ps1`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At '21:30'
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -RestartInterval (New-TimeSpan -Minutes 15) -RestartCount 2 -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$level = if ($isAdmin) { 'Highest' } else { 'Limited' }
# S4U runs without an interactive login and without storing the user's password.
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType S4U -RunLevel $level
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
$scheduledTask = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "Task: $name"
Write-Host "Next Run: $($info.NextRunTime)"
Write-Host "State: $($scheduledTask.State)"
Write-Host "WakeToRun: $($scheduledTask.Settings.WakeToRun)"
Write-Host "StartWhenAvailable: $($scheduledTask.Settings.StartWhenAvailable)"
Write-Host "Retry: $($scheduledTask.Settings.RestartInterval), count=$($scheduledTask.Settings.RestartCount)"
Write-Host "MultipleInstances: $($scheduledTask.Settings.MultipleInstances)"
Write-Host "RunLevel: $($scheduledTask.Principal.RunLevel)"
