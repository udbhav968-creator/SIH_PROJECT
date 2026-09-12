<#
    upgrade_road_shield.ps1

    One command, start to finish, for demo day:

      1. applies ROAD-SHIELD_rewrite.zip to the repo and pushes it
      2. installs onnxruntime if it is missing
      3. ingests the Kaggle road datasets, if a token is present, and retrains
      4. downloads the ImageNet ResNet-50 backbone (98 MB, once)
      4. trains the classifier head on its embeddings and prints the measured score
      5. runs the regression tests
      6. starts the server and opens the dashboard

    Run from PowerShell:
      powershell -ExecutionPolicy Bypass -File .\upgrade_road_shield.ps1

    Options:
      -SkipPush          repo is already up to date, just do the CNN steps
      -SkipKaggle        don't touch Kaggle (no token, or no time for the download)
      -Backbone mobilenetv2   14 MB instead of 98 MB; 83.5% instead of 88.5%
      -NoServer          stop after training, don't start the server
#>

param(
    [string]$RepoDir  = (Join-Path $env:USERPROFILE "Desktop\SIH_PROJECT"),
    [string]$ZipPath  = (Join-Path $env:USERPROFILE "Downloads\ROAD-SHIELD_rewrite.zip"),
    [string]$PushScript = (Join-Path $env:USERPROFILE "Downloads\push_road_shield.ps1"),
    [ValidateSet("resnet50", "mobilenetv2")]
    [string]$Backbone = "resnet50",
    [string]$CommitMessage = "Deep CNN embeddings replace hand-crafted features: 88.5% held-out accuracy",
    [switch]$SkipPush,
    [switch]$SkipKaggle,
    [switch]$NoServer
)

$ErrorActionPreference = "Stop"
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# 1. Apply the new code and push it
# ---------------------------------------------------------------------------
if (-not $SkipPush) {
    if (-not (Test-Path $PushScript)) { Fail "push_road_shield.ps1 not found at '$PushScript'." }
    if (-not (Test-Path $ZipPath))    { Fail "ROAD-SHIELD_rewrite.zip not found at '$ZipPath'." }
    Step "Applying the new code and pushing"
    & powershell -ExecutionPolicy Bypass -File $PushScript -ZipPath $ZipPath -RepoDir $RepoDir -CommitMessage $CommitMessage
    if ($LASTEXITCODE -ne 0) { Fail "push_road_shield.ps1 failed. Fix that first, then rerun with -SkipPush." }
}

if (-not (Test-Path $RepoDir)) { Fail "Repo not found at '$RepoDir'." }
Set-Location $RepoDir

$py = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found, using the system python." -ForegroundColor Yellow
    $py = "python"
}

# ---------------------------------------------------------------------------
# 2. ONNX Runtime - this is what runs the CNN. No PyTorch anywhere.
# ---------------------------------------------------------------------------
Step "Checking ONNX Runtime"
& $py -c "import onnxruntime, sys; print('onnxruntime', onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing onnxruntime ..." -ForegroundColor Yellow
    & $py -m pip install --quiet "onnxruntime>=1.16.0"
    if ($LASTEXITCODE -ne 0) { Fail "Could not install onnxruntime. Without it the server falls back to the 87.7% path, which still works." }
}

# ---------------------------------------------------------------------------
# 3. Backbone weights (ONNX Model Zoo, downloaded once)
# ---------------------------------------------------------------------------
Step "Fetching the $Backbone backbone"
& $py -m scripts.fetch_cnn_backbone --model $Backbone
if ($LASTEXITCODE -ne 0) {
    Write-Host "Backbone download failed. The server will still run on the hand-crafted-feature path (87.7%)." -ForegroundColor Yellow
    Write-Host "Retry later with:  .\.venv\Scripts\python.exe -m scripts.fetch_cnn_backbone --model mobilenetv2" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 3b. Kaggle datasets (optional - needs a token in ~/.kaggle/kaggle.json)
# ---------------------------------------------------------------------------
if (-not $SkipKaggle) {
    Step "Kaggle datasets"
    & $py -c "import kaggle" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing the kaggle package ..." -ForegroundColor Yellow
        & $py -m pip install --quiet kaggle
    }
    $tok = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    if (-not (Test-Path $tok)) {
        Write-Host "No $tok - skipping Kaggle." -ForegroundColor Yellow
        Write-Host "  kaggle.com -> avatar -> Settings -> API -> Create New Token, save it there, rerun." -ForegroundColor Yellow
    } else {
        Write-Host "Checking which catalogue datasets are reachable (downloads nothing) ..."
        & $py -m scripts.fetch_kaggle_datasets --verify
        Write-Host "Downloading and ingesting. Duplicates against the existing data are dropped." -ForegroundColor Yellow
        & $py -m scripts.fetch_kaggle_datasets
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Kaggle ingest did not complete. Training continues on the data already on disk." -ForegroundColor Yellow
        } else {
            Step "Retraining the baseline models on the enlarged dataset"
            & $py -m training.train_mega_suite
        }
    }
}

# ---------------------------------------------------------------------------
# 4. Train the head on the embeddings, and score the old features on the same split
# ---------------------------------------------------------------------------
Step "Training the classifier head (about 2 minutes)"
& $py -m training.train_cnn_head --backbone $Backbone --compare
if ($LASTEXITCODE -ne 0) { Fail "Head training failed - read the error above." }

# ---------------------------------------------------------------------------
# 5. Tests
# ---------------------------------------------------------------------------
Step "Running the regression tests"
& $py -m unittest tests.test_road_shield
if ($LASTEXITCODE -ne 0) { Write-Host "Some tests failed - read the output above before demoing." -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# 6. Serve
# ---------------------------------------------------------------------------
if ($NoServer) {
    Write-Host "`nDone. Start the server with:  .\.venv\Scripts\python.exe -m api.server" -ForegroundColor Green
    exit 0
}

Step "Starting the server"
Write-Host "Watch the startup line. You want:" -ForegroundColor Yellow
Write-Host "    Vision classifier: deep CNN embeddings (cnn:$Backbone+logistic)" -ForegroundColor Yellow
Write-Host "If it says 'HOG/LBP + SVM baseline' instead, the backbone is missing." -ForegroundColor Yellow
Start-Process "http://127.0.0.1:8000/dashboard"
& $py -m api.server
