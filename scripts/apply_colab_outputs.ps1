# Copies the Colab training outputs into this repository and pushes them.
#
#   1. In Colab, the last notebook cell downloads road_shield_colab_outputs.zip
#   2. Run from PowerShell:
#        powershell -ExecutionPolicy Bypass -File C:\Users\Dell\SIH_PROJECT\scripts\apply_colab_outputs.ps1
#
# It unpacks the zip over the repo (checkpoints, logs, the rebuilt report),
# shows what changed, commits on the audit branch and pushes. Nothing goes to master.
param([string]$Zip = "$env:USERPROFILE\Downloads\road_shield_colab_outputs.zip")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
if (-not (Test-Path -LiteralPath $Zip)) { Write-Host "Not found: $Zip" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
Write-Host "Unpacking $Zip into $repo" -ForegroundColor Cyan
Expand-Archive -LiteralPath $Zip -DestinationPath $repo -Force
$ErrorActionPreference = "Continue"
git checkout audit-2026-10-03
git status --short checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
if (Test-Path "checkpoints\finetune_summary.json") {
    Write-Host "`nServed image classifier:" -ForegroundColor Cyan
    Get-Content "checkpoints\vision_model_selection.json" | Select-String '"served"|"evidence"'
}
if (Test-Path "logs\tests_after.txt") { Write-Host "`nTests after training:"; Get-Content "logs\tests_after.txt" -Tail 3 }
# datasets/ and runs/ stay out of git (see .gitignore); only results are committed
git add -- checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
git commit -m "Colab GPU training results: fine-tuned CNNs, IMU 1-D CNN, retrained head, rebuilt report and claims" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin audit-2026-10-03
Read-Host "Press Enter to close"
