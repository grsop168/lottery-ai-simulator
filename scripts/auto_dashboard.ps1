[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\wtjps\lottery-ai-simulator'
$code = 1
try {
    Set-Location -LiteralPath $root
    $logs = Join-Path $root 'reports\logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $log = Join-Path $logs ('dashboard-{0}-{1}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), $PID)
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $env:PYTHONPATH = Join-Path $root 'src'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:Path = "$root\.venv\Scripts;" + $env:Path
    "Start: $((Get-Date).ToString('o')) Python: $python Working directory: $root" | Set-Content $log -Encoding UTF8
    if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv python unavailable' }
    $ErrorActionPreference = 'Continue'
    # Synchronous invocation: never detach the server from this wrapper.
    & $python -u "$root\src\lottery_sim\dashboard_service.py" 2>&1 | ForEach-Object { "$_" | Add-Content $log -Encoding UTF8 }
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
}
catch {
    $detail = ($_ | Format-List * -Force | Out-String) + $_.ScriptStackTrace
    if ($log) { $detail | Add-Content $log -Encoding UTF8 }
    Write-Error $detail -ErrorAction Continue
    $code = 1
}
finally {
    if ($log) { "Finish: $((Get-Date).ToString('o')) Exit: $code" | Add-Content $log -Encoding UTF8 }
}
exit $code
