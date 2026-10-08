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
# $PSScriptRoot is empty when the script is pasted instead of run as a file: then use the current folder
$proj = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
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
# logs of earlier runs are not needed any more (and only the last few tunnel logs are kept while running)
Remove-Item "logs\tunnel_*", "logs\engine_public_*" -Force -ErrorAction SilentlyContinue
function Drop-OldLogs($n) {
    if ($n -gt 4) { Remove-Item "logs\tunnel_$($n - 4).*" -Force -ErrorAction SilentlyContinue }
}
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
# Operator key for the write endpoints (adding a defect to the map, issuing a work order). Visitors can analyse
# photographs without it; only the person running this window, who sees the key below, can change the ledger.
$OPERATOR_KEY = if ($env:ROAD_SHIELD_API_KEY) { $env:ROAD_SHIELD_API_KEY } else { -join ((48..57) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ }) }

function Start-Engine {
    $script:engineRuns++
    $n = $script:engineRuns
    $env:ROAD_SHIELD_PUBLIC = "1"
    $env:ROAD_SHIELD_API_KEY = $OPERATOR_KEY
    $env:ROAD_SHIELD_RATE_LIMIT = "20"
    $env:ROAD_SHIELD_MAX_BODY_MB = "12"
    $env:ROAD_SHIELD_WRITABLE_DIR = Join-Path $env:TEMP "road_shield_public"
    $p = Start-Process python -ArgumentList "-m", "api.server", "$port" -WorkingDirectory $proj -PassThru `
         -WindowStyle Hidden -RedirectStandardOutput "logs\engine_public_$n.log" -RedirectStandardError "logs\engine_public_$n.err"
    Track $p
    foreach ($v in "ROAD_SHIELD_PUBLIC", "ROAD_SHIELD_API_KEY", "ROAD_SHIELD_RATE_LIMIT", "ROAD_SHIELD_MAX_BODY_MB", "ROAD_SHIELD_WRITABLE_DIR") {
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
    Drop-OldLogs $n
    $log = "logs\tunnel_$n.log"
    $p = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port" -PassThru `
         -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "logs\tunnel_$n.out"
    Track $p
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep 2
        if ($p.HasExited) { break }
        $m = Select-String -Path $log -Pattern "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) {
            # the name goes live once cloudflared reports a registered connection; give DNS a few more seconds
            $registered = $false
            for ($j = 0; $j -lt 20; $j++) {
                if (Select-String -Path $log -Pattern "Registered tunnel connection" -CaseSensitive -Quiet -ErrorAction SilentlyContinue) {
                    $registered = $true; break
                }
                Start-Sleep 2
            }
            if (-not $registered) {
                # cloudflared reaches Cloudflare on port 7844; many campus and office networks block it
                Say "Cloudflare could not connect from this network (its port 7844 is probably blocked). Last log lines:" Yellow
                Get-Content $log -Tail 4 -ErrorAction SilentlyContinue | Out-Host
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                return @{ Proc = $null; Url = $null; Kind = "Cloudflare"; Blocked = $true }
            }
            Start-Sleep 8
            return @{ Proc = $p; Url = $m.Matches[0].Value; Kind = "Cloudflare" }
        }
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
    Drop-OldLogs $n
    $log = "logs\tunnel_$n.log"
    $p = Start-Process ssh -ArgumentList "-T", "-n", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", `
         "-R", "80:127.0.0.1:$port", "nokey@localhost.run" -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput $log -RedirectStandardError "logs\tunnel_$n.err"
    Track $p
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep 2
        if ($p.HasExited) { break }
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.lhr\.life" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { Start-Sleep 8; return @{ Proc = $p; Url = $m.Matches[0].Value; Kind = "localhost.run"; Log = $log } }
    }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    return $null
}

# Why the public check never uses this laptop's own DNS:
# a new tunnel name does not exist for a few seconds after cloudflared prints it. If Windows asks for it in
# that moment, it remembers "no such name" for up to 15 minutes, and then this laptop (script AND browser)
# sees a dead link that works for everyone else. So the check resolves the name through Cloudflare's
# DNS-over-HTTPS with curl.exe (built into Windows 10/11), which leaves Windows' DNS cache untouched.
$CURL = Join-Path $env:SystemRoot "System32\curl.exe"
# Which DNS-over-HTTPS service works from this network (Cloudflare's or Google's, by IP address), if any.
$script:DOH = $null
$script:REVOKE = ""          # "--ssl-revoke-best-effort" if this curl knows it (campus networks often block revocation checks)
if (Test-Path $CURL) {
    foreach ($flag in "--ssl-revoke-best-effort", "") {
        foreach ($d in "https://1.1.1.1/dns-query", "https://8.8.8.8/dns-query") {
            $cargs = @("-s", "-m", "10", "-o", "NUL")
            if ($flag) { $cargs += $flag }
            $cargs += @("--doh-url", $d, "https://www.cloudflare.com/cdn-cgi/trace")
            $null = & $CURL $cargs 2>$null
            if ($LASTEXITCODE -eq 0) { $script:DOH = $d; $script:REVOKE = $flag; break }
        }
        if ($script:DOH) { break }
    }
}

function Get-PublicHealth($url) {
    if ($script:DOH) {
        $cargs = @("-s", "-m", "12")
        if ($script:REVOKE) { $cargs += $script:REVOKE }
        $cargs += @("--doh-url", $script:DOH, "$url/api/v1/health")
        $out = & $CURL $cargs 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { try { return (($out | Out-String) | ConvertFrom-Json) } catch { } }
        return $null
    }
    # no DNS-over-HTTPS here (old curl, or blocked): this laptop's DNS. Asked only after the tunnel has
    # registered, so the name already exists the first time Windows looks it up.
    try { return Invoke-RestMethod "$url/api/v1/health" -TimeoutSec 10 } catch { return $null }
}

function Test-Public($url, $tries) {
    for ($i = 0; $i -lt $tries; $i++) {
        $r = Get-PublicHealth $url
        if ($r -and $r.status -eq "ONLINE") { return $true }
        Start-Sleep 3
    }
    return $false
}

function Test-Local($tries) {
    for ($i = 0; $i -lt $tries; $i++) {
        try {
            $r = Invoke-RestMethod "http://127.0.0.1:$port/api/v1/health" -TimeoutSec 10
            if ($r.status -eq "ONLINE") { return $true }
        } catch { }
        Start-Sleep 3
    }
    return $false
}

# After the link works: can THIS laptop's own DNS see it? (Others can either way.)
function Test-LocalDns($url) {
    $hostName = ([Uri]$url).Host
    for ($i = 0; $i -lt 3; $i++) {
        try { $null = Resolve-DnsName $hostName -Type A -ErrorAction Stop; return $true } catch { }
        ipconfig /flushdns | Out-Null       # forget a "no such name" this laptop may have cached
        Start-Sleep 5
    }
    return $false
}

$script:cfBroken = $false      # set once Cloudflare has failed on this network; later reconnects skip it
function Open-Tunnel($cf) {
    if ($cf -and -not $script:cfBroken) {
        for ($a = 1; $a -le 2; $a++) {
            $t = Start-CfTunnel $cf
            if ($t -and $t.Blocked) { break }
            if ($t) {
                Say "Tunnel $($t.Url) opened - checking it answers from the internet..."
                if (Test-Public $t.Url 20) { return $t }
                Say "That link does not answer (attempt $a of 2)." Yellow
                Stop-Process -Id $t.Proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
        $script:cfBroken = $true
        Say "Cloudflare tunnels do not work on this network; using localhost.run from now on." Yellow
        Say "(For a demo, a phone hotspot usually allows Cloudflare: its links last for hours.)" Yellow
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
    if (-not (Test-LocalDns $t.Url)) {
        Say "The link works on the internet, but THIS laptop's DNS cannot see it yet (your network's DNS)." Yellow
        Say "Phones and other computers can open it. To open it on this laptop too, in Edge go to" Yellow
        Say "  Settings > Privacy, search, and services > Security > Use secure DNS > Choose a provider: Cloudflare (1.1.1.1)" Yellow
    }
    Write-Host ""
    Write-Host "  YOUR LIVE LINKS (via $($t.Kind)) - open them in a browser, not in PowerShell:" -ForegroundColor Green
    Write-Host "    Website with the live engine (opened for you, and copied to the clipboard):" -ForegroundColor Green
    Write-Host "      $link"
    Write-Host "    The engine's own full site:" -ForegroundColor Green
    Write-Host "      $($t.Url)"
    Write-Host "    Operator key (only for 'Add to map' and work orders; do not share it):" -ForegroundColor Green
    Write-Host "      $OPERATOR_KEY"
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
    if ($script:DOH) { Say "Links are checked through $script:DOH (DNS over HTTPS)." }
    else { Say "DNS over HTTPS is not available on this network; links are checked through this laptop's DNS." Yellow }
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
        if ($t.Log) {
            # free localhost.run addresses change every so often; the same connection prints the new one
            $last = Select-String -Path $t.Log -Pattern "https://[a-z0-9-]+\.lhr\.life" -ErrorAction SilentlyContinue | Select-Object -Last 1
            if ($last -and $last.Matches[0].Value -ne $t.Url) {
                $t.Url = $last.Matches[0].Value
                Say "localhost.run changed the address. New link below; the OLD link no longer works." Yellow
                Show-Links $t; $bad = 0; continue
            }
        }
        if ($tick % 8 -eq 0) {
            if (Test-Public $t.Url 2) { $bad = 0; continue }
            if (-not (Test-Local 2)) {
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
