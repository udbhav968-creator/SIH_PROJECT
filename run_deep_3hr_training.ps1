# ==============================================================================
# ROAD-SHIELD: 3-Hour Deep Multi-Modal Training Launcher (PowerShell)
# SIH 2026 Problem Statement SIH26124 | Bharat Electronics Limited (BEL)
# ==============================================================================
[CmdletBinding()]
param (
    [int]$DanEpochs = 80,
    [int]$SegImages = 2000,
    [int]$TargetRare = 80,
    [string]$Backbone = "mobilenetv2",
    [switch]$OpenBrowserOnComplete
)

$ErrorActionPreference = "Stop"

# 1. Environment & Concurrency Safeguards
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OPENBLAS_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location -Path $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

# 2. Locate Python Runtime
$PythonExe = "C:\Python314\python.exe"
if (-not (Test-Path $PythonExe)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $PythonExe = $cmd.Source
    } else {
        Write-Error "Python runtime not found at C:\Python314\python.exe or in PATH."
        exit 1
    }
}

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD: 3-HOUR DEEP MULTI-MODAL TRAINING PIPELINE (POWERSHELL)" -ForegroundColor Cyan
Write-Host "  SIH26124 / BEL | Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "  Using Python Runtime: $PythonExe" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Planned Pipeline Stages:" -ForegroundColor Yellow
Write-Host "  1. Rare-Class Dataset Expansion via 64-bit DCT pHash  (~10 min)"
Write-Host "  2. Real-Image Civil Models (ASTM PCI, HDM-4, Depth)   (~15 min)"
Write-Host "  3. 100 Hz IMU Tri-Axial Chassis Shock Classifier      (~5 min)"
Write-Host "  4. 11-Feature Pixel Defect Segmenter on 2,000 photos  (~35 min)"
Write-Host "  5. Deep Mirror-Averaged CNN Embedder Space            (~45 min)"
Write-Host "  6. Flagship DAN-DAG Deep PyTorch Training (80 Epochs) (~40 min)"
Write-Host "  7. Regression Test Suite & Verification               (~10 min)"
Write-Host "-------------------------------------------------------------------------" -ForegroundColor Gray
Write-Host "Estimated Total Time: 2.5 - 3.0 Hours on CPU.`n" -ForegroundColor Yellow

$t0 = Get-Date

& $PythonExe -u -m scripts.train_deep_3hr_suite `
    --dan-epochs $DanEpochs `
    --seg-images $SegImages `
    --target-rare $TargetRare `
    --backbone $Backbone

if ($LASTEXITCODE -eq 0) {
    $totalMinutes = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host "`n=========================================================================" -ForegroundColor Green
    Write-Host "  TRAINING FINISHED SUCCESSFULLY in $totalMinutes minutes!" -ForegroundColor Green
    Write-Host "  All production checkpoints have been updated in checkpoints/." -ForegroundColor Green
    Write-Host "=========================================================================" -ForegroundColor Green

    if ($OpenBrowserOnComplete) {
        Write-Host "Starting server and launching web inspection dashboard..." -ForegroundColor Cyan
        powershell -ExecutionPolicy Bypass -File .\start_server.ps1 -OpenBrowser
    } else {
        Write-Host "`nTo view your newly trained models on the live dashboard, run:" -ForegroundColor Cyan
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\start_server.ps1 -OpenBrowser`n" -ForegroundColor White
    }
} else {
    Write-Host "`nTraining encountered an error. Check logs above." -ForegroundColor Red
    exit $LASTEXITCODE
}
