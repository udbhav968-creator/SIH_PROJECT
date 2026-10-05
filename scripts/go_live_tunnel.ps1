# ROAD-SHIELD: make the live engine on THIS laptop reachable from the internet, free, no account.
#
#     powershell -ExecutionPolicy Bypass -File .\scripts\go_live_tunnel.ps1
#
# What it does
#   1. starts the engine in public-demo mode (uploads analysed and not stored, write endpoints locked,
#      no access to files outside datasets\, 20 analyses per minute per visitor)
#   2. opens a Cloudflare quick tunnel (cloudflared is downloaded once from Cloudflare's GitHub releases),
#      checks the link really answers from the internet, and tries again if it does not
#      (falls back to localhost.run over ssh if Cloudflare is unreachable from this network)
#   3. opens your website on the right link in the browser, and copies that link to the clipboard
#   4. keeps the laptop awake, restarts the engine if it stops, and opens a new tunnel if the link dies
#      (a new tunnel means a new link: it is printed, copied and opened again)
# Keep this window open. Ctrl+C stops everything. Links go in the BROWSER, never in PowerShell.

$ErrorActionPreference = "Continue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$proj = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { $proj = Join-Path $env:USERPROFILE "SIH_PROJECT" }
if (-not (Test-Path (Join-Path $proj "web\config.js"))) { Write-Host "Cannot find the project folder ($proj)." -ForegroundColor Red; exit 1 }
Set-Location $proj
New-Item -ItemType Directory -Force logs | Out-Null
$SITE = "https://road-shield-ai-engine.vercel.app"

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Say($t, $c = "Gray") { Write-Host ("{0}  {1}" -f (Get-Date -Format HH:mm:ss), $t) -ForegroundColor $c }

# Processes this script starts are recorded as "PID STARTTICKS" (the start time guards against a reused PID).
# The first line names the PowerShell running the script, so a second copy can tell one is already live.
$PIDS = "logs\go_live_pids.txt"
function Track($p) { if ($p) { try { Add-Content $PIDS "$($p.Id) $($p.StartTime.Ticks)" } catch { } } }
function Stop-Tracked {
    Get-Content $PIDS -ErrorAction SilentlyContinue | Select-Object -Skip 1 | ForEach-Object {
        $id, $ticks = $_ -split " "
        $q = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($q -and $q.Name -in "python", "cloudflared", "ssh" -and "$($q.StartTime.Ticks)" -eq $ticks) {
            Stop-Process -Id $q.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item $PIDS -ErrorAction SilentlyContinue
}
$owner = (Get-Content $PIDS -TotalCount 1 -ErrorAction SilentlyContinue)
if ($owner) {
    $oid, $oticks = $owner -split " "
    $op = Get-Process -Id $oid -ErrorAction SilentlyContinue
    if ($op -and $op.Id -ne $PID -and $op.Name -match "^(powershell|pwsh)" -and "$($op.StartTime.Ticks)" -eq $oticks) {
        Say "go_live_tunnel is already running in another window - use the link shown there." Yellow
        Say "To restart it, press Ctrl+C in that window first." Yellow
        exit 0
    }
}
Stop-Tracked     # leftovers of an earlier run that was closed without Ctrl+C
Set-Content $PIDS "$PID $((Get-Process -Id $PID).StartTime.Ticks)"

# ------------------------------------------------------------ keep awake
# Asks Windows not to sleep while this script runs (undone when it stops). Closing the lid may still sleep,
# depending on the lid setting - keep the lid open during a demo.
$awake = $false
try {
    Add-Type -Namespace RoadShield -Name Power -ErrorAction Stop -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
    [void][RoadShield.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    $awake = $true
} catch { }

# ------------------------------------------------------------ engine
$port = 8001
while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $port++ }
$script:engineRuns = 0

function Start-Engine {
    $script:engineRuns++
    $n = $script:engineRuns
    $env:ROAD_SHIELD_PUBLIC = "1"
    $env:ROAD_SHIELD_RATE_LIMIT = "20"
    $env:ROAD_SHIELD_MAX_BODY_MB = "12"
    $env:ROAD_SHIELD_WRITABLE_DIR = Join-Path $env:TEMP "road_shield_public"
    $p = Start-Process python -ArgumentList "-m", "api.server", "$port" -WorkingDirectory $proj -PassThru `
         -WindowStyle Hidden -RedirectStandardOutput "logs\engine_public_$n.log" -RedirectStandardError "logs\engine_public_$n.err"
    Track $p
    foreach ($v in "ROAD_SHIELD_PUBLIC", "ROAD_SHIELD_RATE_LIMIT", "ROAD_SHIELD_MAX_BODY_MB", "ROAD_SHIELD_WRITABLE_DIR") {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep 3
        if ($p.HasExited) { break }
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$port/api/v1/health" -TimeoutSec 5
            if ($h.status -eq "ONLINE") { return $p }
        } catch { }
    }
    Say "The engine did not start. Last lines of its log:" Red
    Get-Content "logs\engine_public_$n.err", "logs\engine_public_$n.log" -Tail 15 -ErrorAction SilentlyContinue | Out-Host
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    return $null
}

# ------------------------------------------------------------ tunnels
$script:tunnelRuns = 0

function Get-Cloudflared {
    $cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
    if ($cf) { return $cf }
    $cf = Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe"
    if (-not (Test-Path $cf)) {
        Say "Downloading cloudflared (about 60 MB) from Cloudflare's GitHub releases..."
        New-Item -ItemType Directory -Force (Split-Path $cf) | Out-Null
        $ProgressPreference = "SilentlyContinue"
        try {
            Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
                -OutFile "$cf.part" -UseBasicParsing -ErrorAction Stop
            Move-Item "$cf.part" $cf -Force
        } catch { Say "Download failed: $($_.Exception.Message)" Red; Remove-Item "$cf.part" -ErrorAction SilentlyContinue; return $null }
    }
    return $cf
}

function Start-CfTunnel($cf) {
    $script:tunnelRuns++
    $n = $script:tunnelRuns
    $log = "logs\tunnel_$n.log"
    $p = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port" -PassThru `
         -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "logs\tunnel_$n.out"
    Track $p
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep 2
        if ($p.HasExited) { break }
        $m = Select-String -Path $log -Pattern "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { return @{ Proc = $p; Url = $m.Matches[0].Value; Kind = "Cloudflare" } }
    }
    Say "cloudflared gave no link. Last lines of its log:" Yellow
    Get-Content $log -Tail 8 -ErrorAction SilentlyContinue | Out-Host
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    return $null
}

function Start-LhrTunnel {
    if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) { Say "ssh is not installed, so localhost.run cannot be used." Yellow; return $null }
    $script:tunnelRuns++
    $n = $script:tunnelRuns
    $log = "logs\tunnel_$n.log"
    $p = Start-Process ssh -ArgumentList "-T", "-n", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", `
         "-R", "80:127.0.0.1:$port", "nokey@localhost.run" -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput $log -RedirectStandardError "logs\tunnel_$n.err"
    Track $p
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep 2
        if ($p.HasExited) { break }
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.lhr\.life" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { return @{ Proc = $p; Url = $m.Matches[0].Value; Kind = "localhost.run" } }
    }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    return $null
}

function Test-Public($url, $tries) {
    for ($i = 0; $i -lt $tries; $i++) {
        try {
            $r = Invoke-RestMethod "$url/api/v1/health" -TimeoutSec 10
            if ($r.status -eq "ONLINE") { return $true }
        } catch { }
        Start-Sleep 3
    }
    return $false
}

function Open-Tunnel($cf) {
    if ($cf) {
        for ($a = 1; $a -le 3; $a++) {
            $t = Start-CfTunnel $cf
            if ($t) {
                Say "Tunnel $($t.Url) opened - checking it answers from the internet..."
                if (Test-Public $t.Url 30) { return $t }
                Say "That link does not answer (attempt $a of 3). Opening a new tunnel." Yellow
                Stop-Process -Id $t.Proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
        Say "Cloudflare links are not reachable from this network. Trying localhost.run instead..." Yellow
    }
    $t = Start-LhrTunnel
    if ($t) {
        Say "Tunnel $($t.Url) opened - checking it answers..."
        if (Test-Public $t.Url 20) { return $t }
        Stop-Process -Id $t.Proc.Id -Force -ErrorAction SilentlyContinue
    }
    return $null
}

function Show-Links($t) {
    $link = "$SITE/inspect?engine=$($t.Url)"
    Write-Host ""
    Write-Host "  YOUR LIVE LINKS (via $($t.Kind)) - open them in a browser, not in PowerShell:" -ForegroundColor Green
    Write-Host "    Website with the live engine (opened for you, and copied to the clipboard):" -ForegroundColor Green
    Write-Host "      $link"
    Write-Host "    The engine's own full site:" -ForegroundColor Green
    Write-Host "      $($t.Url)"
    Write-Host ""
    try { Set-Clipboard $link } catch { }
    try { Start-Process $link } catch { }
}

# ------------------------------------------------------------ run
$engine = $null; $t = $null
try {
    Step "1/3  start the engine in public-demo mode"
    $engine = Start-Engine
    if (-not $engine) { exit 1 }
    Say "Engine running on http://127.0.0.1:$port" Green
    if ($awake) { Say "This laptop will stay awake while this window is open (keep the lid open)." }

    Step "2/3  open a public tunnel"
    $cf = Get-Cloudflared
    $t = Open-Tunnel $cf
    if (-not $t) {
        Say "No public link could be opened from this network (Cloudflare and localhost.run both failed)." Red
        Say "Try another connection (e.g. a phone hotspot) and run this script again." Red
        Say "The engine still works on this laptop: http://127.0.0.1:$port" Yellow
        exit 1
    }

    Step "3/3  live"
    Show-Links $t
    Say "Keep this window open. Ctrl+C stops everything. Checking the link every 2 minutes."

    $tick = 0; $bad = 0
    while ($true) {
        Start-Sleep 15
        $tick++
        if (-not $engine -or $engine.HasExited) {
            Say "The engine is not running - starting it (the link stays the same)." Yellow
            $engine = Start-Engine
            if ($engine) { Say "Engine back." Green } else { Say "Engine did not start; trying again in 15 s." Red }
            continue
        }
        if (-not $t -or $t.Proc.HasExited) {
            Say "The tunnel is closed - opening a new one. The OLD link no longer works." Yellow
            $t = Open-Tunnel $cf
            if ($t) { Show-Links $t; $bad = 0 } else { Say "No tunnel yet (network?); trying again in 15 s." Red }
            continue
        }
        if ($tick % 8 -eq 0) {
            if (Test-Public $t.Url 2) { $bad = 0; continue }
            if (-not (Test-Public "http://127.0.0.1:$port" 2)) {
                # the engine itself is frozen: restart it and keep the tunnel (and the link)
                Say "The engine stopped answering - restarting it (the link stays the same)." Yellow
                Stop-Process -Id $engine.Id -Force -ErrorAction SilentlyContinue
                $engine = $null
                continue
            }
            $bad++
            Say "The public link did not answer ($bad)." Yellow
            if ($bad -ge 2) {
                Say "Opening a new tunnel. The OLD link no longer works." Yellow
                Stop-Process -Id $t.Proc.Id -Force -ErrorAction SilentlyContinue
                $t = Open-Tunnel $cf
                if ($t) { Show-Links $t; $bad = 0 } else { Say "No tunnel yet (network?); trying again in 15 s." Red }
            }
        }
    }
} finally {
    if ($engine) { Stop-Process -Id $engine.Id -Force -ErrorAction SilentlyContinue }
    if ($t) { Stop-Process -Id $t.Proc.Id -Force -ErrorAction SilentlyContinue }
    Stop-Tracked
    if ($awake) { [void][RoadShield.Power]::SetThreadExecutionState([uint32]2147483648) }   # ES_CONTINUOUS: sleep allowed again
    Say "Stopped the engine and the tunnel."
}
