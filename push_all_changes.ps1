# ==============================================================================
# ROAD-SHIELD: Full Cache Purge, Git Sync/Push & Vercel Force Deployment Script
# Repository: https://github.com/udbhav968-creator/SIH_PROJECT.git
# ==============================================================================
[CmdletBinding()]
param (
    [string]$CommitMessage = "feat(vercel): purge old caches, enforce no-store headers, and sync all trained reports"
)

$ErrorActionPreference = "Stop"

# 1. Navigate to Project Root
$ProjectRoot = "C:\Users\Dell\Downloads\SIH_PROJECT"
Set-Location -Path $ProjectRoot

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD: CACHE PURGE + GITHUB PUSH + VERCEL FORCE DEPLOY" -ForegroundColor Cyan
Write-Host "  Workspace: $ProjectRoot" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Cyan

# 2. Purge old local & Vercel caches
Write-Host "`n[1/5] Purging old local & Vercel build caches..." -ForegroundColor Yellow
if (Test-Path "scratch") {
    Remove-Item -Path "scratch\*" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "scratch" -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path ".vercel\.cache") {
    Remove-Item -Path ".vercel\.cache" -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "  Local scratch/, __pycache__/, and .vercel/.cache purged." -ForegroundColor Green

# 3. Stage all new, modified, and deleted files (respecting .gitignore)
Write-Host "`n[2/5] Staging all changes..." -ForegroundColor Yellow
git add -A
git status --short

# 4. Commit if there are staged changes
$staged = git diff --cached --name-only
if ($staged) {
    Write-Host "`n[3/5] Committing staged files..." -ForegroundColor Yellow
    git commit -m $CommitMessage
} else {
    Write-Host "`n[3/5] Working tree is already clean (no uncommitted local changes)." -ForegroundColor Green
}

# 5. Ensure master and add-dan-dag are synced and push both to origin
Write-Host "`n[4/5] Pushing to GitHub (origin/master and origin/add-dan-dag)..." -ForegroundColor Yellow
$currentBranch = (git branch --show-current).Trim()

if ($currentBranch -ne "master") {
    git checkout master
    git merge $currentBranch --no-edit
}

git branch -f add-dan-dag master
git push origin master
git push origin master:add-dan-dag

# 6. Force-deploy to Vercel Production bypassing all build/CDN caches
Write-Host "`n[5/5] Deploying to Vercel Production with --force (cache bypassed)..." -ForegroundColor Yellow
vercel.ps1 --prod --force --yes

Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "  SUCCESS: Caches deleted, GitHub synced, and Vercel updated!" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
