# Copies the Colab training outputs into this repository and pushes them.
#
#   1. In Colab, the last notebook cell downloads road_shield_colab_outputs.zip
#   2. Run from PowerShell:
#        powershell -ExecutionPolicy Bypass -File C:\Users\Dell\SIH_PROJECT\scripts\apply_colab_outputs.ps1
#
# It unpacks the zip over the repo (checkpoints, logs, the rebuilt report),
# shows what changed, commits on the audit branch and pushes. Nothing goes to master.
param([string]$Zip = "$env:USERPROFILE\Downloads\road_shield_colab_outputs.zip",
      [string]$ExtraZip = "$env:USERPROFILE\Downloads\road_shield_extra_outputs.zip")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$haveMain = Test-Path -LiteralPath $Zip
$haveExtra = Test-Path -LiteralPath $ExtraZip
if (-not $haveMain -and -not $haveExtra) { Write-Host "Not found: $Zip (or $ExtraZip)" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
if ($haveMain) { Write-Host "Unpacking $Zip into $repo" -ForegroundColor Cyan; Expand-Archive -LiteralPath $Zip -DestinationPath $repo -Force }
# The extra models (U-Net segmenter, RDD2022 detector) are applied after the main run, so their
# claims and selection files win.
if ($haveExtra) { Write-Host "Unpacking $ExtraZip into $repo" -ForegroundColor Cyan; Expand-Archive -LiteralPath $ExtraZip -DestinationPath $repo -Force }
$ErrorActionPreference = "Continue"
git checkout audit-2026-10-03
git status --short checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
if (Test-Path "checkpoints\finetune_summary.json") {
    Write-Host "`nServed image classifier:" -ForegroundColor Cyan
    Get-Content "checkpoints\vision_model_selection.json" | Select-String '"served"|"evidence"'
}
if (Test-Path "checkpoints\segmenter_selection.json") {
    Write-Host "`nServed segmenter:" -ForegroundColor Cyan
    Get-Content "checkpoints\segmenter_selection.json" | Select-String '"served"|"why"' | Select-Object -First 2
}
if (Test-Path "checkpoints\road_damage_detector_report.json") {
    Write-Host "`nRoad-damage detector (test):" -ForegroundColor Cyan
    Get-Content "checkpoints\road_damage_detector_report.json" | Select-String '"map50"' | Select-Object -Last 1
}
if (Test-Path "logs\tests_after.txt") { Write-Host "`nTests after training:"; Get-Content "logs\tests_after.txt" -Tail 3 }
# datasets/ and runs/ stay out of git (see .gitignore); only results are committed
git add -- checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
git commit -m "Colab GPU training results: fine-tuned CNNs, IMU 1-D CNN, retrained head, U-Net segmenter, RDD2022 detector, claims" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin audit-2026-10-03
Read-Host "Press Enter to close"
