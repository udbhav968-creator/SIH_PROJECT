# ROAD-SHIELD: make the live engine on THIS laptop reachable from the internet, free, no account.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\go_live_tunnel.ps1
#
# 1. starts the engine in public-demo mode (uploads analysed and not stored, write endpoints locked,
#    no access to files outside datasets\, 20 analyses per minute per visitor)
# 2. opens a Cloudflare quick tunnel (cloudflared, downloaded once from Cloudflare's GitHub releases)
# 3. prints two public links and copies the website one to the clipboard
# The links work while this window stays open. Ctrl+C stops everything. Each run gives a new link.

$ErrorActionPreference = "Continue"
$proj = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { $proj = Join-Path $env:USERPROFILE "SIH_PROJECT" }
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { Write-Host "Cannot find the project folder ($proj)." -ForegroundColor Red; exit 1 }
Set-Location $proj
New-Item -ItemType Directory -Force logs | Out-Null
function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. engine
Step "1/3  start the engine in public-demo mode"
$port = 8001
while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $port++ }
$env:ROAD_SHIELD_PUBLIC = "1"
$env:ROAD_SHIELD_RATE_LIMIT = "20"
$env:ROAD_SHIELD_MAX_BODY_MB = "12"
$env:ROAD_SHIELD_WRITABLE_DIR = Join-Path $env:TEMP "road_shield_public"
$engine = Start-Process python -ArgumentList "-m", "api.server", "$port" -WorkingDirectory $proj -PassThru `
          -WindowStyle Hidden -RedirectStandardOutput "logs\engine_public.log" -RedirectStandardError "logs\engine_public.err"
foreach ($v in "ROAD_SHIELD_PUBLIC", "ROAD_SHIELD_RATE_LIMIT", "ROAD_SHIELD_MAX_BODY_MB", "ROAD_SHIELD_WRITABLE_DIR") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 3
    if ($engine.HasExited) { break }
    try { $h = Invoke-RestMethod "http://127.0.0.1:$port/api/v1/health" -TimeoutSec 5; if ($h.status -eq "ONLINE") { $up = $true; break } } catch {}
}
if (-not $up) {
    Write-Host "The engine did not start. Last lines of its log:" -ForegroundColor Red
    Get-Content "logs\engine_public.err", "logs\engine_public.log" -Tail 15 -ErrorAction SilentlyContinue
    if (-not $engine.HasExited) { Stop-Process -Id $engine.Id -Force }
    exit 1
}
Write-Host "Engine running on http://127.0.0.1:$port (public-demo mode: $($h.public_demo))" -ForegroundColor Green

# ---------------------------------------------------------------- 2. tunnel
Step "2/3  open a Cloudflare tunnel"
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { $cf = Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe" }
if (-not (Test-Path $cf)) {
    Write-Host "Downloading cloudflared (about 60 MB) from Cloudflare's GitHub releases..."
    New-Item -ItemType Directory -Force (Split-Path $cf) | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $cf -UseBasicParsing
}
Remove-Item "logs\tunnel.log" -ErrorAction SilentlyContinue
$tunnel = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port" -PassThru `
          -WindowStyle Hidden -RedirectStandardError "logs\tunnel.log" -RedirectStandardOutput "logs\tunnel.out"
$url = $null
for ($i = 0; $i -lt 40 -and -not $url; $i++) {
    Start-Sleep 2
    $m = Select-String -Path "logs\tunnel.log" -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $url = $m.Matches[0].Value }
}
if (-not $url) {
    Write-Host "The tunnel did not open. Last lines of its log:" -ForegroundColor Red
    Get-Content "logs\tunnel.log" -Tail 15 -ErrorAction SilentlyContinue
    Stop-Process -Id $engine.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "Tunnel: $url  (checking it answers...)"
for ($i = 0; $i -lt 30; $i++) {
    try { $p = Invoke-RestMethod "$url/api/v1/health" -TimeoutSec 10; if ($p.status -eq "ONLINE") { break } } catch {}
    Start-Sleep 3
}

# ---------------------------------------------------------------- 3. links
Step "3/3  your public links"
$site = "https://road-shield-ai-engine.vercel.app/inspect?engine=$url"
Write-Host ""
Write-Host "  Full live site, running on this laptop:" -ForegroundColor Green
Write-Host "      $url"
Write-Host "  Your Vercel site using this laptop as its engine (copied to the clipboard):" -ForegroundColor Green
Write-Host "      $site"
Write-Host ""
Write-Host "  Open either link on any phone or computer, upload a road photo, press Analyse."
Write-Host "  Keep this window open. Press Ctrl+C to stop; the links stop working when you do."
try { Set-Clipboard $site } catch {}

try {
    while (-not $engine.HasExited -and -not $tunnel.HasExited) { Start-Sleep 5 }
    if ($engine.HasExited) { Write-Host "The engine stopped. Log:" -ForegroundColor Red; Get-Content "logs\engine_public.err" -Tail 15 }
    if ($tunnel.HasExited) { Write-Host "The tunnel stopped. Log:" -ForegroundColor Red; Get-Content "logs\tunnel.log" -Tail 15 }
} finally {
    Stop-Process -Id $engine.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped the engine and the tunnel."
}
