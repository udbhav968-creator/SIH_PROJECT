@echo off
title ROAD-SHIELD Master Training Pipeline
color 0a
echo =========================================================================
echo   ROAD-SHIELD Master Multi-Benchmark Training Pipeline
echo   Loading 7 Datasets (73,000 samples) and Training Model M1 / M4 / M_PCI
echo =========================================================================
cd /d "%~dp0"
python datasets\master_pipeline.py --train
pause
