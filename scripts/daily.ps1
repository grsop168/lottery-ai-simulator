[CmdletBinding()]
param(
    [string]$DataDir = "data/normalized",
    [string]$ReportDir = "reports/latest",
    [string]$RecommendationDir = "data/recommendations",
    [string]$ModelDir = "data/models",
    [string]$Game = "",
    [switch]$All,
    [int]$Seed = 20260505,
    [string]$Seeds = "1,2,3,4,5",
    [int]$Window = 30,
    [int]$Count = 10,
    [int]$MlMinHistory = 30,
    [int]$MlMinTrain = 200,
    [int]$MlTrainEpochs = 8,
    [int]$MlBacktestEpochs = 3,
    [int]$MlBacktestLimit = 30,
    [int]$MlRetrainEvery = 10,
    [switch]$SkipNormalize,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Utf8 = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = $Utf8
$OutputEncoding = $Utf8
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$env:PYTHONPATH = "src"

$Games = @(
    [pscustomobject]@{ Code = "3d"; Csv = "$DataDir/fucai3d.csv"; Update = "update-3d"; Backtest = "backtest-3d"; Compare = "compare-3d"; Stability = "stability-3d"; Recommend = "recommend-3d"; Record = "record-recommend-3d"; Verify = "verify-recommend-3d"; Extra = @() },
    [pscustomobject]@{ Code = "pl3"; Csv = "$DataDir/pl3.csv"; Update = "update-pl3"; Backtest = "backtest-pl3"; Compare = "compare-pl3"; Stability = "stability-pl3"; Recommend = "recommend-pl3"; Record = "record-recommend-pl3"; Verify = "verify-recommend-pl3"; Extra = @() },
    [pscustomobject]@{ Code = "pl5"; Csv = "$DataDir/pl5.csv"; Update = "update-pl5"; Backtest = "backtest-pl5"; Compare = "compare-pl5"; Stability = "stability-pl5"; Recommend = "recommend-pl5"; Record = "record-recommend-pl5"; Verify = "verify-recommend-pl5"; Extra = @() },
    [pscustomobject]@{ Code = "qxc"; Csv = "$DataDir/qxc.csv"; Update = "update-qxc"; Backtest = "backtest-qxc"; Compare = "compare-qxc"; Stability = "stability-qxc"; Recommend = "recommend-qxc"; Record = "record-recommend-qxc"; Verify = "verify-recommend-qxc"; Extra = @() },
    [pscustomobject]@{ Code = "qlc"; Csv = "$DataDir/qlc.csv"; Update = "update-qlc"; Backtest = "backtest-qlc"; Compare = "compare-qlc"; Stability = "stability-qlc"; Recommend = "recommend-qlc"; Record = "record-recommend-qlc"; Verify = "verify-recommend-qlc"; Extra = @() },
    [pscustomobject]@{ Code = "kl8"; Csv = "$DataDir/kl8.csv"; Update = "update-kl8"; Backtest = "backtest-kl8"; Compare = "compare-kl8"; Stability = "stability-kl8"; Recommend = "recommend-kl8"; Record = "record-recommend-kl8"; Verify = "verify-recommend-kl8"; Extra = @("--pick-size", "10") },
    [pscustomobject]@{ Code = "ssq"; Csv = "$DataDir/ssq.csv"; Update = "update-ssq"; Backtest = "backtest-ssq"; Compare = "compare-ssq"; Stability = "stability-ssq"; Recommend = "recommend-ssq"; Record = "record-recommend-ssq"; Verify = "verify-recommend-ssq"; Extra = @() },
    [pscustomobject]@{ Code = "dlt"; Csv = "$DataDir/dlt.csv"; Update = "update-dlt"; Backtest = "backtest-dlt"; Compare = "compare-dlt"; Stability = "stability-dlt"; Recommend = "recommend-dlt"; Record = "record-recommend-dlt"; Verify = "verify-recommend-dlt"; Extra = @() }
)

if ($All -and $Game) {
    throw "Use either -Game <code> or -All, not both."
}
if (-not $All -and -not $Game) {
    throw "Please specify -Game <code> or use -All explicitly."
}

$AllowedGames = @($Games | ForEach-Object { $_.Code })
if ($Game -and $AllowedGames -notcontains $Game) {
    throw "Unsupported -Game '$Game'. Allowed: $($AllowedGames -join ', ')"
}

$SelectedGames = @()
if ($All) {
    $SelectedGames = @($Games)
}
else {
    $SelectedGames = @($Games | Where-Object { $_.Code -eq $Game })
}

function Ensure-Directory {
    param([string]$Path)

    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Assert-DataFile {
    param([string]$Path)

    if (-not $DryRun -and -not (Test-Path -LiteralPath $Path)) {
        throw "Data file not found: $Path. Run without -SkipNormalize first."
    }
}

function Get-LotteryCommandText {
    param([string[]]$Arguments)

    if ($env:LOTTERY_CLI_EXE) {
        return "`"$env:LOTTERY_CLI_EXE`" --cli " + ($Arguments -join " ")
    }
    return "python -m lottery_sim.cli " + ($Arguments -join " ")
}

function Invoke-LotteryCli {
    param([string[]]$Arguments)

    if ($env:LOTTERY_CLI_EXE) {
        & $env:LOTTERY_CLI_EXE --cli @Arguments
        return
    }
    & python -m lottery_sim.cli @Arguments
}

function Invoke-LotteryCommand {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$OutputFile = ""
    )

    $commandText = Get-LotteryCommandText -Arguments $Arguments
    if ($OutputFile) {
        $commandText = "$commandText > $OutputFile"
    }
    Write-Host "[$Name] $commandText"

    if ($DryRun) {
        return
    }

    if ($OutputFile) {
        $output = Invoke-LotteryCli -Arguments $Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $output | Set-Content -Path $OutputFile -Encoding UTF8
        Write-Host "  -> $OutputFile"
    }
    else {
        Invoke-LotteryCli -Arguments $Arguments
        $exitCode = $LASTEXITCODE
    }

    if ($exitCode -ne 0) {
        if ($OutputFile -and $output) {
            $output | Write-Host
        }
        throw "Command failed with exit code $($exitCode): $Name"
    }
}

Ensure-Directory -Path $DataDir
Ensure-Directory -Path $ReportDir
Ensure-Directory -Path $RecommendationDir
Ensure-Directory -Path $ModelDir

foreach ($gameConfig in $SelectedGames) {
    if (-not $SkipNormalize) {
        Invoke-LotteryCommand -Name "update-$($gameConfig.Code)" -Arguments @($gameConfig.Update, "--csv", $gameConfig.Csv)
    }
    else {
        Assert-DataFile -Path $gameConfig.Csv
    }

    Invoke-LotteryCommand -Name "backtest-$($gameConfig.Code)" -Arguments (@($gameConfig.Backtest, "--csv", $gameConfig.Csv) + $gameConfig.Extra + @("--seed", "$Seed")) -OutputFile "$ReportDir/backtest-$($gameConfig.Code).txt"
    Invoke-LotteryCommand -Name "compare-$($gameConfig.Code)" -Arguments (@($gameConfig.Compare, "--csv", $gameConfig.Csv) + $gameConfig.Extra + @("--seed", "$Seed", "--window", "$Window")) -OutputFile "$ReportDir/compare-$($gameConfig.Code).txt"
    Invoke-LotteryCommand -Name "stability-$($gameConfig.Code)" -Arguments (@($gameConfig.Stability, "--csv", $gameConfig.Csv) + $gameConfig.Extra + @("--seeds", $Seeds, "--window", "$Window")) -OutputFile "$ReportDir/stability-$($gameConfig.Code).txt"
    Invoke-LotteryCommand -Name "recommend-$($gameConfig.Code)" -Arguments (@($gameConfig.Recommend, "--csv", $gameConfig.Csv) + $gameConfig.Extra + @("--seed", "$Seed", "--window", "$Window", "--count", "$Count")) -OutputFile "$ReportDir/recommend-$($gameConfig.Code).txt"

    $mlModel = Join-Path $ModelDir "$($gameConfig.Code)-ml.json"
    Invoke-LotteryCommand -Name "train-ml-$($gameConfig.Code)" -Arguments (@("train-ml-$($gameConfig.Code)", "--csv", $gameConfig.Csv, "--model", $mlModel, "--min-history", "$MlMinHistory", "--epochs", "$MlTrainEpochs") + $gameConfig.Extra)
    Invoke-LotteryCommand -Name "backtest-ml-$($gameConfig.Code)" -Arguments (@("backtest-ml-$($gameConfig.Code)", "--csv", $gameConfig.Csv, "--min-train", "$MlMinTrain", "--min-history", "$MlMinHistory", "--limit", "$MlBacktestLimit", "--retrain-every", "$MlRetrainEvery", "--epochs", "$MlBacktestEpochs") + $gameConfig.Extra) -OutputFile "$ReportDir/backtest-ml-$($gameConfig.Code).txt"
    if ($gameConfig.Code -eq "pl5") {
        Invoke-LotteryCommand -Name "compare-models-pl5" -Arguments @(
            "compare-models-pl5", "--csv", $gameConfig.Csv,
            "--models", "random,logistic,markov,lightgbm,house",
            "--min-train-draws", "$MlMinTrain", "--backtest-draws", "$MlBacktestLimit",
            "--retrain-every", "$MlRetrainEvery", "--candidate-count", "100",
            "--min-history", "$MlMinHistory", "--logistic-epochs", "$MlBacktestEpochs",
            "--history-db", (Join-Path (Split-Path $ModelDir -Parent) "history.sqlite3"),
            "--output-json", "$ReportDir/model-comparison-pl5.json",
            "--output-text", "$ReportDir/model-comparison-pl5.txt"
        )
        Invoke-LotteryCommand -Name "analyze-house-pl5" -Arguments @(
            "analyze-house-pl5", "--csv", $gameConfig.Csv,
            "--bettor-model", "behavioral", "--house-temperature", "1000000",
            "--simulated-ticket-count", "1000000",
            "--output-json", "$ReportDir/house-analysis-pl5.json",
            "--output-text", "$ReportDir/house-analysis-pl5.txt"
        )
    }
    Invoke-LotteryCommand -Name "recommend-ml-$($gameConfig.Code)" -Arguments (@("recommend-ml-$($gameConfig.Code)", "--csv", $gameConfig.Csv, "--model", $mlModel, "--count", "$Count", "--min-history", "$MlMinHistory", "--epochs", "$MlTrainEpochs") + $gameConfig.Extra) -OutputFile "$ReportDir/recommend-ml-$($gameConfig.Code).txt"
    Invoke-LotteryCommand -Name "record-ml-$($gameConfig.Code)" -Arguments (@(
        "record-recommend-ml-$($gameConfig.Code)",
        "--csv",
        $gameConfig.Csv,
        "--model",
        $mlModel,
        "--store",
        $RecommendationDir,
        "--count",
        "$Count",
        "--min-history",
        "$MlMinHistory",
        "--epochs",
        "$MlTrainEpochs"
    ) + $gameConfig.Extra)

    $gameDir = Join-Path $RecommendationDir $gameConfig.Code
    $recordFiles = @()
    if (Test-Path -LiteralPath $gameDir) {
        $recordFiles = @(Get-ChildItem -LiteralPath $gameDir -Filter "*.csv" | Sort-Object FullName)
    }

    if ($DryRun -and $recordFiles.Count -eq 0) {
        $sampleRecord = Join-Path $gameDir "<issue>.csv"
        Invoke-LotteryCommand -Name "verify-$($gameConfig.Code)" -Arguments (@(
            $gameConfig.Verify,
            "--csv",
            $gameConfig.Csv,
            "--records",
            $sampleRecord,
            "--output",
            $sampleRecord
        ) + $gameConfig.Extra)
    }
    else {
        foreach ($recordFile in $recordFiles) {
            Invoke-LotteryCommand -Name "verify-$($gameConfig.Code)-$($recordFile.BaseName)" -Arguments (@(
                $gameConfig.Verify,
                "--csv",
                $gameConfig.Csv,
                "--records",
                $recordFile.FullName,
                "--output",
                $recordFile.FullName
            ) + $gameConfig.Extra)
        }
    }

    Invoke-LotteryCommand -Name "record-$($gameConfig.Code)" -Arguments (@(
        $gameConfig.Record,
        "--csv",
        $gameConfig.Csv,
        "--store",
        $RecommendationDir,
        "--seed",
        "$Seed",
        "--window",
        "$Window",
        "--count",
        "$Count"
    ) + $gameConfig.Extra)
}

$summaryPath = Join-Path $ReportDir "recommendation-summary.txt"
if ($DryRun) {
    Invoke-LotteryCommand -Name "summarize-recommendations" -Arguments @(
        "summarize-recommendations",
        "--records",
        "$RecommendationDir/*/*.csv",
        "--output",
        $summaryPath
    )
}
else {
    $recordPaths = @(Get-ChildItem -Path $RecommendationDir -Recurse -Filter "*.csv" | Sort-Object FullName)
    if ($recordPaths.Count -gt 0) {
        $summaryArgs = @("summarize-recommendations", "--records")
        foreach ($recordPath in $recordPaths) {
            $summaryArgs += $recordPath.FullName
        }
        $summaryArgs += @("--output", $summaryPath)
        Invoke-LotteryCommand -Name "summarize-recommendations" -Arguments $summaryArgs
    }
    else {
        Write-Host "[summarize-recommendations] no recommendation records found"
    }
}

Write-Host "Daily workflow finished for: $($SelectedGames.Code -join ', ')"
