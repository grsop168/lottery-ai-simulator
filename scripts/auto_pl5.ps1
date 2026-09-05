[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$RepoRoot = 'C:\Users\wtjps\lottery-ai-simulator'
$started = Get-Date
$exitCode = 1
$transcribing = $false
try {
    Set-Location -LiteralPath $RepoRoot
    $logDir = Join-Path $RepoRoot 'reports\logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir ('pl5-daily-{0}-{1}.log' -f $started.ToString('yyyyMMdd-HHmmss-fff'), $PID)
    Start-Transcript -Path $log | Out-Null
    $transcribing = $true
    Write-Host "Start: $($started.ToString('o'))"
    $env:Path = "$RepoRoot\.venv\Scripts;" + $env:Path
    $env:PYTHONPATH = Join-Path $RepoRoot 'src'
    $env:PYTHONIOENCODING = 'utf-8'
    Remove-Item Env:LOTTERY_CLI_EXE -ErrorAction SilentlyContinue
    Remove-Item Env:LOTTERY_PL5_WORKER -ErrorAction SilentlyContinue
    $python = (Get-Command python -CommandType Application | Select-Object -First 1).Source
    Write-Host "Python: $python"
    Write-Host "Working directory: $((Get-Location).Path)"
    if ($python -ne "$RepoRoot\.venv\Scripts\python.exe") { throw 'Project .venv python unavailable' }
    $ErrorActionPreference = 'Continue'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$RepoRoot\scripts\daily.ps1" -Game pl5 -FastGenerate -ReportDir "reports/users/admin/latest" -RecommendationDir "data/users/admin/recommendations" -ModelDir "data/users/admin/models" 2>&1 | ForEach-Object { Write-Host $_ }
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "daily.ps1 failed: $LASTEXITCODE" }
    $exitCode = 0
    Write-Host 'SUCCESS'
}
catch {
    Write-Host 'FAILURE'
    Write-Host ($_ | Format-List * -Force | Out-String)
    Write-Host $_.ScriptStackTrace
}
finally {
    Write-Host "Finish: $((Get-Date).ToString('o'))"
    Write-Host "Elapsed seconds: $(((Get-Date) - $started).TotalSeconds)"
    if ($transcribing) { Stop-Transcript | Out-Null }
}
exit $exitCode
