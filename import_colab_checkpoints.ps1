# ==============================================================================
# ROAD-SHIELD: Import Colab GPU Checkpoints, Verify CI Gate, Push & Deploy
# ==============================================================================
[CmdletBinding()]
param (
    [string]$ZipPath = "C:\Users\Dell\Downloads\road_shield_trained_checkpoints.zip"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ProjectRoot = "C:\Users\Dell\Downloads\SIH_PROJECT"
Set-Location -Path $ProjectRoot

if (-not (Test-Path $ZipPath)) {
    Write-Host "ERROR: $ZipPath not found!" -ForegroundColor Red
    Write-Host "Please download 'road_shield_trained_checkpoints.zip' from your Google Drive:" -ForegroundColor Yellow
    Write-Host "  MyDrive -> ROAD_SHIELD_Training -> road_shield_trained_checkpoints.zip" -ForegroundColor Yellow
    Write-Host "into C:\Users\Dell\Downloads\ and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Extracting Colab GPU checkpoints from $ZipPath ..." -ForegroundColor Cyan
$TempExtract = Join-Path $env:TEMP "road_shield_colab_extract"
if (Test-Path $TempExtract) { Remove-Item $TempExtract -Recurse -Force }
Expand-Archive -Path $ZipPath -DestinationPath $TempExtract -Force

# Copy GPU-trained artifacts (fine-tuned CNNs, U-Net, 1-D IMU CNN) while preserving
# any local report if the Colab run used an older commit before d753631
$GpuPatterns = @(
    "deep_vision_*.onnx",
    "deep_vision_*.json",
    "finetune_*.json",
    "vision_model_selection.json",
    "defect_segmenter_unet.onnx",
    "defect_segmenter_unet.json",
    "segmenter_selection.json",
    "imu_shock_cnn.onnx",
    "imu_shock_cnn.json",
    "imu_deep_report.json",
    "imu_model_selection.json"
)

$SrcCkpt = Join-Path $TempExtract "checkpoints"
if (Test-Path $SrcCkpt) {
    foreach ($pat in $GpuPatterns) {
        Get-ChildItem -Path $SrcCkpt -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination (Join-Path $ProjectRoot "checkpoints\$($_.Name)") -Force
            Write-Host "  + Imported GPU artifact: $($_.Name)" -ForegroundColor Green
        }
    }
}
Remove-Item $TempExtract -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`n[2/4] Refreshing claims.json, README.md, and running MLOps CI gate ..." -ForegroundColor Cyan
C:\Python314\python.exe -m scripts.build_claims
C:\Python314\python.exe -m scripts.build_readme
C:\Python314\python.exe -m mlops.ci_check

Write-Host "`n[3/4] Running unit test verification suite ..." -ForegroundColor Cyan
C:\Python314\python.exe -m unittest tests.test_road_shield

Write-Host "`n[4/4] Pushing to GitHub & Force-Deploying to Vercel ..." -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File .\push_all_changes.ps1 -CommitMessage "feat(gpu): import Colab T4 GPU fine-tuned CNN, U-Net, and 1D-CNN checkpoints"
