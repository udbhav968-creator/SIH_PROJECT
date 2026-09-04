@echo off
title ROAD-SHIELD AI Intelligence Gateway (Port 8000)
color 0b
echo =========================================================================
echo   ROAD-SHIELD AI Intelligence Gateway - MoRTH / NHAI SIH2026
echo   Starting high-throughput Python API Server on http://127.0.0.1:8000
echo =========================================================================
cd /d "%~dp0"
python api\server.py 8000
pause
