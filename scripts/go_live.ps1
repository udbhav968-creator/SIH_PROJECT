# ROAD-SHIELD: put the live engine on a Hugging Face Space (Docker Spaces need a PRO subscription) and connect the site.
# Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File .\scripts\go_live.ps1
# Safe to run again: every step skips what is already done.

$ErrorActionPreference = "Continue"
$repo   = "udbhav968-creator/SIH_PROJECT"
$branch = "audit-2026-10-03"
# works from scripts\ in the project, or from Downloads (then the project is assumed at %USERPROFILE%\SIH_PROJECT)
$proj = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { $proj = Join-Path $env:USERPROFILE "SIH_PROJECT" }
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { Write-Host "Cannot find the project folder ($proj)." -ForegroundColor Red; exit 1 }
Set-Location $proj

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

function Merge-ToMaster($title) {
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        gh pr create --repo $repo --base master --head $branch --title $title --body $title 2>$null | Out-Null
        Write-Host "If gh asks whether to DELETE the branch, answer NO (training uses it)." -ForegroundColor Yellow
        gh pr merge $branch --repo $repo --merge
    } else {
        Write-Host "Open https://github.com/$repo/compare/master...$branch" -ForegroundColor Yellow
        Write-Host "Click 'Create pull request' (or 'View pull request'), then 'Merge pull request' -> 'Confirm merge'." -ForegroundColor Yellow
        Read-Host "Press Enter here after merging"
    }
}

# ---------------------------------------------------------------- 1
Step "1/6  huggingface_hub 1.x (keeps transformers / datasets working)"
pip install -q "huggingface_hub>=1.5,<2.0"

# ---------------------------------------------------------------- 2
Step "2/6  Hugging Face login"
$who = python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $who) {
    Write-Host "Choose 'Log in with your browser' (or paste a NEW Write token). Never paste a token in chat." -ForegroundColor Yellow
    python -c "from huggingface_hub import login; login()"
    $who = python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>$null
}
if (-not $who) { Write-Host "Login failed - run this script again." -ForegroundColor Red; exit 1 }
Write-Host "Logged in as $who" -ForegroundColor Green

# ---------------------------------------------------------------- 3
Step "3/6  merge the latest code into master (the Space builds from master)"
git pull -q origin $branch
Merge-ToMaster "Live engine + site redesign"

# ---------------------------------------------------------------- 4
Step "4/6  create the Space and upload its two files"
$py = @'
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()["name"]
rid = f"{user}/road-shield-engine"
api.create_repo(rid, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path="deploy/huggingface", repo_id=rid, repo_type="space",
                  commit_message="ROAD-SHIELD engine, built from GitHub")
print("https://" + rid.replace("/", "-").replace("_", "-").replace(".", "-").lower() + ".hf.space")
'@
Set-Content "$env:TEMP\make_space.py" $py -Encoding ascii
$engine = (python "$env:TEMP\make_space.py" | Select-Object -Last 1)
if (-not $engine -or -not $engine.StartsWith("https://")) { Write-Host "Creating the Space failed (see above)." -ForegroundColor Red; exit 1 }
$engine = $engine.Trim()
Write-Host "Space page : https://huggingface.co/spaces/$who/road-shield-engine" -ForegroundColor Green
Write-Host "Engine     : $engine" -ForegroundColor Green

# ---------------------------------------------------------------- 5
Step "5/6  wait for the build (first time 10-15 min; the Space page's Logs tab shows progress)"
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        $r = Invoke-RestMethod "$engine/api/v1/ready" -TimeoutSec 20
        Write-Host ("{0}  ready={1}  missing={2}" -f (Get-Date -Format HH:mm), $r.ready, ($r.missing -join ","))
        if ($r.ready -or $r.models.vision_classifier) { $ready = $true; break }
    } catch { Write-Host ("{0}  still building / starting..." -f (Get-Date -Format HH:mm)) }
    Start-Sleep 30
}
if (-not $ready) {
    Write-Host "Not up after 30 minutes. Open the Space page -> Logs, copy the last 30 lines and send them." -ForegroundColor Red
    exit 1
}
Write-Host "Engine is up." -ForegroundColor Green

# ---------------------------------------------------------------- 6
Step "6/6  point the website at the engine and publish"
$cfg = Get-Content web/config.js -Raw
$cfg = $cfg -replace '(?m)^window\.ROAD_SHIELD_ENGINE_URL = ".*";', ('window.ROAD_SHIELD_ENGINE_URL = "' + $engine + '";')
Set-Content web/config.js $cfg -NoNewline -Encoding ascii
Select-String -Path web/config.js -Pattern '^window'
python scripts/build_readme.py
git add web/config.js README.md
git commit -q -m "Connect the website to the live engine ($engine)"
git push -q origin $branch
Merge-ToMaster "Connect live engine"

Write-Host "`nDONE." -ForegroundColor Green
Write-Host "In about 2 minutes open https://road-shield-ai-engine.vercel.app/inspect"
Write-Host "The pill should say 'live engine - online'. Upload a road photo and press Analyse."
Write-Host "The engine on its own (full site, live): $engine"
