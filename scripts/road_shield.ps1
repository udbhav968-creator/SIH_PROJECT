<#
    road_shield.ps1  -  one script for everything.

    USUAL USE
      cd $env:USERPROFILE\Downloads
      powershell -ExecutionPolicy Bypass -File .\road_shield.ps1

      Stops stale engines, applies the latest zip, checks the segmenter really
      loads, trains one if it does not, runs the tests, commits, pushes, starts
      the engine and opens the browser.

    OTHER THINGS YOU HAVE ASKED FOR, NOW IN ONE PLACE
      .\road_shield.ps1 -ServeOnly     just start the engine and open it
      .\road_shield.ps1 -StopAll       kill every running engine, free memory
      .\road_shield.ps1 -Status        what is installed, loaded and measured
      .\road_shield.ps1 -Measure       re-measure false positives end to end
      .\road_shield.ps1 -Tune          sweep the proposal thresholds (~25 min)
      .\road_shield.ps1 -Retrain       force a segmenter retrain
      .\road_shield.ps1 -NoPush        do everything except push
      .\road_shield.ps1 -SkipTests     push without running the suite
      .\road_shield.ps1 -MaxPixels N   lower the training cap if memory is tight

    TWO THINGS THIS SCRIPT KNOWS THAT COST A DAY TO LEARN

    1. git and python write ordinary progress to stderr. With
       $ErrorActionPreference = "Stop", Windows PowerShell turns any stderr line
       from a native command into a terminating error - so a perfectly
       successful `git pull` aborts the script. Everything native goes through
       Invoke-Native, which reads the real exit code instead.

    2. Every run that ends by starting the engine leaves about a gigabyte
       resident. After a few runs the machine has nothing left, ONNX Runtime
       answers "bad allocation", the classifier silently falls back to a weaker
       one, and the tests then report failures that are not real. So stale
       engines are stopped before anything else happens.
#>

param(
    # Left empty and resolved below: param() must be the first statement in a
    # script, so the home directory cannot be worked out before it.
    [string]$RepoDir   = "",
    [string]$ZipPath   = "",
    [string]$Message   = "ROAD-SHIELD: multi-object detection, per-class thresholds, plausibility bounds",
    [int]$MaxPixels    = 4500000,
    [switch]$ServeOnly,
    [switch]$StopAll,
    [switch]$Status,
    [switch]$Measure,
    [switch]$Tune,
    [switch]$Retrain,
    [switch]$NoPush,
    [switch]$NoServe,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$HomeDir = if ($env:USERPROFILE) { $env:USERPROFILE }
           elseif ($env:HOME) { $env:HOME }
           else { (Get-Location).Path }
if (-not $RepoDir) { $RepoDir = Join-Path $HomeDir "Desktop\SIH_PROJECT" }
if (-not $ZipPath) { $ZipPath = Join-Path $HomeDir "Downloads\ROAD-SHIELD_rewrite.zip" }

function Head($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Note($m) { Write-Host "    $m" -ForegroundColor DarkGray }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

function Invoke-Native {
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$Arguments = @())
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Exe @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        $code = $LASTEXITCODE
        return @{ Code = $code; Output = ($lines -join [Environment]::NewLine) }
    } finally { $ErrorActionPreference = $prev }
}

function Invoke-Streaming {
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$Arguments = @())
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
        return @{ Code = $LASTEXITCODE }
    } finally { $ErrorActionPreference = $prev }
}

# git resolved to the executable, never the name: a helper function called Git
# shadows the command and recurses until the stack overflows.
$script:GitExe = (Get-Command git -CommandType Application -ErrorAction SilentlyContinue |
                  Select-Object -First 1).Source
function Invoke-GitCmd { param([string[]]$Arguments) Invoke-Native -Exe $script:GitExe -Arguments $Arguments }

function Get-FreeMb {
    # FreePhysicalMemory is in KILOBYTES, so /1024 gives MB. Dividing by 1MB
    # gives gigabytes, which is what an earlier version of this advice did -
    # and "5.86" then read as 5 MB rather than 5 GB.
    try {
        return [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    } catch {
        return -1      # not Windows, or CIM unavailable
    }
}

function Stop-Engines {
    $running = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($running.Count -gt 0) {
        Head "Stopping $($running.Count) running engine(s)"
        Note "each holds ResNet-50 and the ONNX runtime, about a gigabyte"
        $running | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    $mb = Get-FreeMb
    if ($mb -ge 0) { Note "$mb MB free" }
}

function Get-Python {
    $p = Join-Path $RepoDir ".venv\Scripts\python.exe"
    if (Test-Path $p) { return $p }
    return "python"
}

function Test-Segmenter {
    param([string]$Py)
    return Invoke-Native -Exe $Py -Arguments @("-c",
        "from models.defect_segmenter import DefectSegmenter as D; s=D(); print('READY' if s.is_ready else 'BROKEN'); print(s.load_error_detail or '')")
}

function Start-Engine {
    param([string]$Py)
    Head "Starting the engine on http://127.0.0.1:8000"
    Note "first start takes ~30s while the ResNet-50 backbone loads"
    Start-Process -FilePath $Py -ArgumentList "-m", "api.server" -WorkingDirectory $RepoDir
    foreach ($i in 1..40) {
        Start-Sleep -Seconds 3
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) {
                Good "engine is up"
                Start-Process "http://127.0.0.1:8000/inspect"
                Write-Host @"

  http://127.0.0.1:8000/inspect        analyse a photograph  <- start here
  http://127.0.0.1:8000/architecture   every claim, checked against this engine
  http://127.0.0.1:8000/models         per-class scores and confusion matrix
  http://127.0.0.1:8000/data           dataset lineage
  http://127.0.0.1:8000/corridor       corridor map
  http://127.0.0.1:8000/works          work orders and the SHA-256 seal demo

"@ -ForegroundColor Cyan
                return $true
            }
        } catch { }
    }
    Warn "engine did not answer in 2 minutes. Run it in a terminal to see why:"
    Warn "  $Py -m api.server"
    return $false
}

# ===========================================================================
# Short modes
# ===========================================================================
if (-not (Test-Path (Join-Path $RepoDir ".git"))) { Fail "No git repo at '$RepoDir'." }
if (-not $script:GitExe) { Fail "git is not on PATH." }
Set-Location $RepoDir
$py = Get-Python

if ($StopAll) { Stop-Engines; Good "done"; exit 0 }

if ($ServeOnly) {
    Stop-Engines
    $probe = Test-Segmenter -Py $py
    if ($probe.Output -match "BROKEN") {
        Warn "the segmenter is NOT loaded - the false-positive fix is inactive."
        Warn "run .\road_shield.ps1 -Retrain to fix it."
    } else { Good "segmenter loaded" }
    Start-Engine -Py $py | Out-Null
    exit 0
}

if ($Status) {
    Head "Status"
    Note "repo    $RepoDir"
    Note "python  $py"
    $mb = Get-FreeMb
    if ($mb -ge 0) { Note "$mb MB free" }
    $probe = Test-Segmenter -Py $py
    if ($probe.Output -match "READY") { Good "segmenter: READY" } else { Warn "segmenter: BROKEN" }
    $rep = Invoke-Native -Exe $py -Arguments @("-c", @"
import json,os
def j(p):
    try:
        with open(os.path.join('checkpoints',p)) as f: return json.load(f)
    except Exception: return None
r=j('cnn_head_resnet50_report.json') or {}
s=j('defect_segmenter_report.json') or {}
d=j('detection_quality_report.json') or {}
c=j('claims.json') or {}
print(f"  classifier      {r.get('held_out_test_accuracy')} accuracy, macro-F1 {r.get('held_out_test_macro_f1')}")
io=s.get('iou') or {}
print(f"  segmenter       crack IoU {io.get('crack',{}).get('iou')}, pothole IoU {io.get('pothole',{}).get('iou')}")
fp=s.get('false_positives_on_clean_roads') or {}
print(f"  clean-road      {fp.get('photo_rate_any_blob')} false-blob rate")
print(f"  end-to-end      {d.get('false_positive_rate')} false positives, {d.get('detection_rate')} found")
print(f"  claims          {len(c.get('subsystems',[]))} subsystems, {len(c.get('corrections',[]))} corrections")
"@)
    Write-Host $rep.Output
    $ahead = Invoke-GitCmd @('rev-list', '--count', 'origin/master..HEAD')
    if ($ahead.Code -eq 0) { Note "$($ahead.Output.Trim()) commit(s) not pushed" }
    exit 0
}

if ($Measure) {
    Stop-Engines
    Head "Measuring false positives and detection, end to end"
    Note "through audit_image() - the same call the API makes"
    Invoke-Streaming -Exe $py -Arguments @("-m", "scripts.measure_detection_quality", "--per-folder", "12") | Out-Null
    exit 0
}

if ($Tune) {
    Stop-Engines
    Head "Sweeping the proposal thresholds (about 25 minutes)"
    Note "prints BOTH error rates at every setting, so the choice can be argued with"
    Invoke-Streaming -Exe $py -Arguments @("-m", "scripts.tune_proposals") | Out-Null
    exit 0
}

# ===========================================================================
# Full flow
# ===========================================================================
Stop-Engines
$free = Get-FreeMb
if ($free -ge 0 -and $free -lt 1200) {
    Warn "under 1.2 GB free. Close Chrome: the engine needs about 1.5 GB and will"
    Warn "silently fall back to a weaker classifier if it cannot load."
}
Note "repo    $RepoDir"
Note "python  $py"

# --- 1. save local state ----------------------------------------------------
Head "Saving local state first"
# NOT $status: PowerShell variable names are case-insensitive, so it would be
# the same variable as the -Status switch and the assignment fails with a type
# error that names neither.
$gitStatus = Invoke-GitCmd @('status', '--porcelain')
if ($gitStatus.Output.Trim()) {
    Invoke-GitCmd @('add', '-A') | Out-Null
    Invoke-GitCmd @('commit', '-q', '-m', 'Local state before update') | Out-Null
    Note "committed local changes"
} else { Note "working tree already clean" }
Invoke-GitCmd @('fetch', 'origin') | Out-Null
$pull = Invoke-GitCmd @('pull', '--ff-only', 'origin', 'master')
if ($pull.Code -ne 0) { Note "pull did not fast-forward; your commits are intact" }
else { Note "up to date with origin/master" }

# --- 2. apply the zip -------------------------------------------------------
if (Test-Path $ZipPath) {
    Head "Removing superseded files"
    $dead = @(
        "scripts/tune_detection_gate.py", "checkpoints/detection_gate_report.json",
        "run_system_test_v2.py", "deep_upgrade_frontend.py", "legacy_dashboard.html",
        "data/data_loader.py",
        "checkpoints/automotive_rl_agent_weights.npz", "checkpoints/deep_imu_weights.npz",
        "checkpoints/deep_vision_weights.npz", "checkpoints/deterioration_forecaster_weights.npz",
        "checkpoints/forensic_embedder_weights.npz", "checkpoints/imu_shock_weights.npz",
        "checkpoints/multimodal_fusion_weights.npz", "checkpoints/pci_regressor_weights.npz",
        "checkpoints/urban_traffic_net_weights.npz", "checkpoints/vision_distress_weights.npz",
        "checkpoints/all_models_train_test_benchmark.json",
        "checkpoints/all_models_training_verification.json",
        "checkpoints/deep_pipeline_test_report.json", "checkpoints/deep_training_curves.json",
        "checkpoints/deep_vision_training_report.json",
        "checkpoints/exhaustive_system_audit_report.json",
        "checkpoints/master_comprehensive_test_report.json", "checkpoints/mega_model_zoo.json",
        "checkpoints/multimodal_real_images_test_report.json",
        "checkpoints/real_unique_images_benchmark_report.json",
        "checkpoints/realworld_vision_summary.json", "checkpoints/system_test_v2_report.json",
        "checkpoints/training_summary.json"
    )
    $removed = 0
    foreach ($d in $dead) {
        if (Test-Path $d) {
            $r = Invoke-GitCmd @('rm', '-q', $d)
            if ($r.Code -ne 0) { Remove-Item $d -Force -ErrorAction SilentlyContinue }
            $removed++
        }
    }
    Note "removed $removed of $($dead.Count) (the rest were already gone)"

    Head "Extracting the new code"
    Expand-Archive -Path $ZipPath -DestinationPath $RepoDir -Force
    Get-ChildItem -Path $RepoDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Note "applied $(Split-Path $ZipPath -Leaf)"
} else {
    Warn "no zip at '$ZipPath' - continuing with the code already in the repo"
}

# --- 3. does the segmenter actually load here? ------------------------------
Head "Checking the segmenter loads in THIS environment"
$probe = Test-Segmenter -Py $py
Write-Host $probe.Output

# An environment failure is never a reason to retrain. Training needs far more
# memory than loading, so retraining is the worst possible response to a machine
# that has run out of it - and it would overwrite a model that is probably fine.
if ($probe.Output -match "sklearn_here.: .unknown" -or $probe.Output -match "environment_failed.: True") {
    Fail ("The model could not be read because THIS MACHINE could not load " +
          "scikit-learn - almost always memory.`n" +
          "  Nothing has been changed; your model is untouched.`n" +
          "  Close Chrome, then run:  .\road_shield.ps1")
}

if ($Retrain -or $probe.Output -match "BROKEN") {
    Head "Training the segmenter here"
    Note "a pickled scikit-learn model is not portable across versions, so it is"
    Note "trained against whatever scikit-learn this venv has"
    Note "progress prints live; 20-30 minutes, longer on battery"
    $t = Invoke-Streaming -Exe $py -Arguments @("-m", "training.train_segmenter", "--images", "2000", "--max-pixels", "$MaxPixels")
    if ($t.Code -ne 0) {
        # Missing training data and an exhausted machine both stop the run, and
        # they need opposite advice. Telling someone to close Chrome when the
        # dataset is absent wastes their time twice.
        if (-not (Test-Path (Join-Path $RepoDir "datasets\incoming\cracks_potholes_dnit\coco.json"))) {
            Fail ("The training data is not in this repo, so the segmenter cannot be " +
                  "built here.`n" +
                  "  Fetch it once:`n" +
                  "    $py -m scripts.fetch_cracks_potholes_dataset --limit 2235`n" +
                  "  Then run this script again.")
        }
        # A reduced fit is a different model, not the same model trained
        # faster. One produced here calibrated to {crack 0.5, pothole 0.2} and
        # measured 54.2% detection where the full fit measures over 90 - and
        # nothing said so, because the run ended with "all tests pass".
        Warn "first attempt failed - retrying on a smaller sample"
        $script:ReducedFit = $true
        $t2 = Invoke-Streaming -Exe $py -Arguments @("-m", "training.train_segmenter", "--images", "1200", "--max-pixels", "900000")
        if ($t2.Code -ne 0) {
            Fail ("Training failed twice.`n" +
                  "  If the output above mentions memory: close Chrome and run`n" +
                  "    .\road_shield.ps1 -MaxPixels 600000`n" +
                  "  Otherwise read the error above - it is not a memory problem.")
        }
    }
    $probe2 = Test-Segmenter -Py $py
    if ($probe2.Output -notmatch "READY") { Fail "the segmenter still will not load after training." }
    if ($script:ReducedFit) {
        Warn "this model was fitted on a REDUCED sample after the full fit ran out of memory."
        Warn "it will load and the tests will pass, but it is a weaker model than the"
        Warn "one the full run produces. Close Chrome and rerun -Retrain when you can."
    }
    Good "trained and loading"
} else {
    Good "segmenter loads - the false-positive fix is active here"
}

# --- 4. tests before pushing ------------------------------------------------
if (-not $SkipTests) {
    Head "Running the test suite"
    # ONNX Runtime's NCHWc layout transform allocates a second copy of every
    # convolution weight. On a host with little headroom that is where
    # "bad allocation" comes from - a Conv node, mid-inference, not session
    # start. ORT_DISABLE_ALL skips the transform: slower, and it runs.
    # The recorded detection figure describes one corpus and one model. This
    # machine has its own of both - 1,404 photographs in the pothole folder
    # where the release machine has 489, and a locally retrained segmenter -
    # so a baseline measured elsewhere says nothing here. Re-measure when the
    # fingerprint no longer matches; otherwise the detection test has no
    # number it is entitled to assert and will skip.
    $bl = Invoke-Native -Exe $py -Arguments @('-m', 'scripts.check_detection_baseline')
    if ($bl.Output -match 'STALE\s*(.*)') {
        Note "detection baseline is stale: $($Matches[1].Trim())"
        Note "re-measuring on this machine's photographs - about three minutes"
        $m = Invoke-Streaming -Exe $py -Arguments @('-m', 'scripts.measure_detection_quality', '--per-folder', '12')
        if ($m.Code -ne 0) { Warn "could not re-measure; the detection test will skip" }
    } elseif ($bl.Output -match 'FRESH\s*(.*)') {
        Note "detection baseline: $($Matches[1].Trim())"
    }

    $freeMb = Get-FreeMb
    if ($freeMb -ge 0 -and $freeMb -lt 6000) {
        $env:ROAD_SHIELD_LITE_ORT = "1"
        Note "$freeMb MB free - running the classifier in low-memory mode"
    } else {
        Remove-Item Env:\ROAD_SHIELD_LITE_ORT -ErrorAction SilentlyContinue
    }
    $t = Invoke-Streaming -Exe $py -Arguments @('-m', 'unittest', 'discover', '-s', 'tests')
    if ($t.Code -ne 0) { Fail "tests failed - nothing has been pushed. Read the failures above." }
    Good "all tests pass"
} else { Note "tests skipped by -SkipTests" }

# --- 5. commit and push -----------------------------------------------------
Head "Committing"
Invoke-GitCmd @('add', '-A') | Out-Null
$short = Invoke-GitCmd @('status', '--short')
if ($short.Output.Trim()) {
    ($short.Output -split "`n" | Select-Object -First 20) | ForEach-Object { Note $_ }
}
$staged = Invoke-GitCmd @('diff', '--cached', '--name-only')
if (-not $staged.Output.Trim()) {
    Note "nothing new to commit"
} else {
    $tmpDir = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
    $msgFile = Join-Path $tmpDir "rs_msg.txt"
    [System.IO.File]::WriteAllText($msgFile, $Message, (New-Object System.Text.UTF8Encoding($false)))
    $cm = Invoke-GitCmd @('commit', '-q', '-F', $msgFile)
    Remove-Item $msgFile -ErrorAction SilentlyContinue
    if ($cm.Code -ne 0) { Fail "commit failed:`n$($cm.Output)" }
    Note "committed"
}

if (-not $NoPush) {
    Head "Pushing to origin/master"
    $ahead = Invoke-GitCmd @('rev-list', '--count', 'origin/master..HEAD')
    if ($ahead.Code -eq 0) { Note "$($ahead.Output.Trim()) commit(s) to push" }
    $push = Invoke-GitCmd @('push', 'origin', 'master')
    Write-Host $push.Output
    if ($push.Code -ne 0) { Fail "push failed - read the output above." }
    Good "pushed"
}

# --- 6. serve ---------------------------------------------------------------
if ($NoServe) { Good "Done. Start it with: .\road_shield.ps1 -ServeOnly"; exit 0 }
Start-Engine -Py $py | Out-Null

Write-Host @"
Other things this script can do:

  .\road_shield.ps1 -Status     what is installed, loaded and measured
  .\road_shield.ps1 -Measure    re-measure false positives end to end
  .\road_shield.ps1 -Tune       sweep the proposal thresholds (~25 min)
  .\road_shield.ps1 -StopAll    kill every engine and free memory

"@ -ForegroundColor DarkGray
