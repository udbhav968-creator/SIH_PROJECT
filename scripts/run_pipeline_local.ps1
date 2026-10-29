# ROAD-SHIELD Local PowerShell Execution Helper
# Usage: .\scripts\run_pipeline_local.ps1
#        .\scripts\run_pipeline_local.ps1 -Fast

param(
    [switch]$Fast,
    [switch]$AllArchs,
    [int]$EpochsCNN = 40,
    [int]$EpochsUNet = 30,
    [int]$EpochsDAN = 65,
    [int]$BatchSize = 16
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD Local Deep Training Pipeline (PowerShell)" -ForegroundColor Cyan
Write-Host "  Location: $RepoRoot" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$CmdArgs = @("scripts/colab_master_deep_training.py")
if ($Fast) {
    $CmdArgs += "--fast"
} else {
    $CmdArgs += @("--epochs-cnn", $EpochsCNN, "--epochs-unet", $EpochsUNet, "--epochs-dan", $EpochsDAN, "--batch-size", $BatchSize)
    if ($AllArchs) {
        $CmdArgs += "--all-archs"
    }
}

python @CmdArgs
