# ==============================================================================
# ROAD-SHIELD AI Intelligence Gateway - PowerShell Server Launcher
# ==============================================================================

[CmdletBinding()]
param (
    [int]$Port = 8000,
    [switch]$OpenBrowser
)

# 1. Set Project Root Directory
$ProjectRoot = "C:\Users\Dell\Desktop\SIH_PROJECT"
if (Test-Path $ProjectRoot) {
    Set-Location -Path $ProjectRoot
} else {
    $ProjectRoot = $PSScriptRoot
    if ($ProjectRoot) { Set-Location -Path $ProjectRoot }
}

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  ROAD-SHIELD AI Intelligence Gateway - MoRTH / NHAI SIH2026" -ForegroundColor Cyan
Write-Host "  Working Directory: $(Get-Location)" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Cyan

# 2. Environment Variables & Concurrency Isolation
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

# 3. Resolve Python Runtime
$PythonExe = "C:\Python314\python.exe"
if (-not (Test-Path $PythonExe)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $PythonExe = $cmd.Source
    } else {
        Write-Error "Python executable not found at C:\Python314\python.exe or in system PATH."
        exit 1
    }
}
Write-Host "[Runtime] Using Python: $PythonExe" -ForegroundColor Green

# 4. Check for existing process on the target port
$ActiveConnection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ActiveConnection) {
    $OccupyingPID = $ActiveConnection.OwningProcess
    Write-Warning "Port $Port is currently occupied by PID $OccupyingPID."
    Write-Host "Stopping existing process on port $Port..." -ForegroundColor Yellow
    Stop-Process -Id $OccupyingPID -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 5. Open Browser if requested
if ($OpenBrowser) {
    Start-Process "http://localhost:$Port"
}

# 6. Start Server Process
Write-Host "`nStarting server on http://127.0.0.1:$Port (Press CTRL+C to terminate)...`n" -ForegroundColor Green
& $PythonExe -u -m api.server $Port