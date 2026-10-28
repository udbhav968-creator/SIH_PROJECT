Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ROAD-SHIELD: RUNNING COMPLETE TEST SUITE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Running 37 Core Unit Tests (ASTM D6433, U-Net, Calibration, IMU)..." -ForegroundColor Yellow
python -m unittest tests.test_road_shield
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Core unit tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/3] Checking 8 MLOps Quality Gates..." -ForegroundColor Yellow
python -m mlops.ci_check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] MLOps quality gates failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[3/3] Testing Marking & Zebra Crossing Specificity..." -ForegroundColor Yellow
python -m unittest tests.test_geometry_and_storage.MarkingDetection
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Marking detection tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " ALL TESTS PASSED SUCCESSFULLY! (100% GREEN)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
