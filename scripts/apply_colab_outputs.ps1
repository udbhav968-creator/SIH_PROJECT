# Takes the Colab training outputs into this repository through the model gates, then pushes them.
#
#   1. In Colab, notebooks/train_everything.ipynb ends by downloading road_shield_all_outputs.zip
#      (older runs: road_shield_colab_outputs.zip, road_shield_extra_outputs.zip, road_shield_deep_outputs.zip)
#   2. Run from PowerShell:
#        powershell -ExecutionPolicy Bypass -File C:\Users\Dell\SIH_PROJECT\scripts\apply_colab_outputs.ps1
#
# Each zip goes through `python -m mlops intake`: every model in it is registered, gated against the one in
# production and promoted only if it passes (a failed one is listed with its reason; production stays). Logs,
# measurement reports and the rebuilt report are copied in. Then the claims are rebuilt from what is served,
# the tests run, and the result is committed on the audit branch and pushed. Nothing goes to master.
param([string]$Downloads = "$env:USERPROFILE\Downloads")
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
git checkout audit-2026-10-03
git pull --ff-only origin audit-2026-10-03

# The one-notebook zip if there is one; otherwise the older per-script zips, oldest first. Each zip is moved to
# Downloads\road_shield_applied after it has been taken in, so running this again never lays an old run over a
# newer model.
$all = Join-Path $Downloads "road_shield_all_outputs.zip"
if (Test-Path -LiteralPath $all) { $zips = @($all) } else {
    $zips = @("road_shield_colab_outputs.zip", "road_shield_extra_outputs.zip", "road_shield_imu_fix.zip",
              "road_shield_deep_outputs.zip") |
            ForEach-Object { Join-Path $Downloads $_ } | Where-Object { Test-Path -LiteralPath $_ }
}
if (-not $zips) { Write-Host "No Colab outputs found in $Downloads" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
$applied = Join-Path $Downloads "road_shield_applied"
New-Item -ItemType Directory -Force -Path $applied | Out-Null

foreach ($z in $zips) {
    Write-Host "`n=== $z ===" -ForegroundColor Cyan
    python -m mlops intake "$z" --note "Colab: $(Split-Path -Leaf $z)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "intake reported an error for $z (see above); it was left in Downloads" -ForegroundColor Red
        continue
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Move-Item -LiteralPath $z -Destination (Join-Path $applied "$stamp-$(Split-Path -Leaf $z)")
}

Write-Host "`nClaims rebuilt from the models now served:" -ForegroundColor Cyan
python -m scripts.build_claims
Write-Host "`nModel registry:" -ForegroundColor Cyan
python -m mlops status
Write-Host "`nTests:" -ForegroundColor Cyan
python -m unittest discover -s tests -t . 2>&1 | Select-Object -Last 3

git status --short checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
$answer = Read-Host "`nCommit and push these results? (y/n)"
if ($answer -eq "y") {
    # datasets/, runs/ and mlops_store/ stay out of git (see .gitignore); only results are committed
    git add -- checkpoints logs CSET485_ROAD_SHIELD_Milestone2_Report_Audited_TrackedChanges.docx
    git commit -m "Colab training results, taken in through the model gates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    git push origin audit-2026-10-03
}
Read-Host "Press Enter to close"
