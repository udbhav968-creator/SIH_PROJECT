# ==============================================================================
# ROAD-SHIELD: Apply & Audit Colab Checkpoints Helper
# ==============================================================================
# Usage:
#   .\scripts\apply_colab_checkpoints.ps1
#   .\scripts\apply_colab_checkpoints.ps1 -ZipPath "$HOME\Downloads\road_shield_trained_checkpoints.zip"
# ==============================================================================

param(
    [string]$ZipPath = ""
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD: APPLYING TRAINED CHECKPOINTS" -ForegroundColor Cyan
Write-Host "  Repository: $RepoRoot" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Locate Zip Archive
if (-not $ZipPath) {
    $Candidates = @(
        "$HOME\Downloads\road_shield_trained_checkpoints.zip",
        "$RepoRoot\road_shield_trained_checkpoints.zip",
        "$HOME\Google Drive\ROAD_SHIELD_Checkpoints\road_shield_trained_checkpoints.zip"
    )
    foreach ($cand in $Candidates) {
        if (Test-Path $cand) {
            $ZipPath = $cand
            break
        }
    }
}

if (-not $ZipPath -or -not (Test-Path $ZipPath)) {
    Write-Host "[!] Could not automatically locate 'road_shield_trained_checkpoints.zip'." -ForegroundColor Yellow
    Write-Host "    Please download the zip file from your Google Drive (ROAD_SHIELD_Checkpoints) to your Downloads folder," -ForegroundColor Yellow
    Write-Host "    or specify the path with: .\scripts\apply_colab_checkpoints.ps1 -ZipPath 'C:\path\to\file.zip'" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Found Checkpoint Archive: $ZipPath" -ForegroundColor Green
Write-Host "      Archive Size: $((Get-Item $ZipPath).Length / 1MB | ForEach-Object { '{0:N2} MB' -f $_ })" -ForegroundColor Gray

Write-Host "`n[2/4] Extracting trained weights into checkpoints/..." -ForegroundColor Cyan
Expand-Archive -LiteralPath $ZipPath -DestinationPath $RepoRoot -Force
Write-Host "      Extraction complete." -ForegroundColor Green

Write-Host "`n[3/4] Running MLOps Quality Gate Audit on Newly Extracted Weights..." -ForegroundColor Cyan
python -m mlops.ci_check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] MLOps quality gates failed. Check output above." -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] Running Multi-Defect (100 Holes) & 3 Zebras Stress Test..." -ForegroundColor Cyan
python -m unittest tests.test_multi_defect_and_multi_zebra_verification
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Stress test failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All fresh Colab weights verified and active!" -ForegroundColor Green
Write-Host "  Ready to commit and deploy." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
