# ==============================================================================
# ROAD-SHIELD: Full Git Stage, Commit, Merge & Push Script (PowerShell)
# Repository: https://github.com/udbhav968-creator/SIH_PROJECT.git
# ==============================================================================
[CmdletBinding()]
param (
    [string]$CommitMessage = "feat: sync latest trained checkpoints, datasets, and training scripts"
)

$ErrorActionPreference = "Stop"

# 1. Navigate to Project Root
$ProjectRoot = "C:\Users\Dell\Downloads\SIH_PROJECT"
Set-Location -Path $ProjectRoot

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD: FULL GIT SYNC & PUSH TO GITHUB" -ForegroundColor Cyan
Write-Host "  Workspace: $ProjectRoot" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Cyan

# 2. Stage all new, modified, and deleted files (respecting .gitignore)
Write-Host "`n[1/4] Staging all changes..." -ForegroundColor Yellow
git add -A
git status --short

# 3. Commit if there are staged changes
$staged = git diff --cached --name-only
if ($staged) {
    Write-Host "`n[2/4] Committing staged files..." -ForegroundColor Yellow
    git commit -m $CommitMessage
} else {
    Write-Host "`n[2/4] Working tree is already clean (no uncommitted local changes)." -ForegroundColor Green
}

# 4. Ensure master and add-dan-dag are synced and push both to origin
Write-Host "`n[3/4] Pushing to GitHub (origin/master and origin/add-dan-dag)..." -ForegroundColor Yellow
$currentBranch = (git branch --show-current).Trim()

if ($currentBranch -ne "master") {
    git checkout master
    git merge $currentBranch --no-edit
}

git branch -f add-dan-dag master
git push origin master
git push origin master:add-dan-dag

# 5. Final Verification
Write-Host "`n[4/4] Verifying remote sync status..." -ForegroundColor Yellow
git status -sb
git log --oneline --decorate -n 3

Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All changes are committed and pushed to GitHub!" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
