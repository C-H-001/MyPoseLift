param(
    [Guid]$BitsJobId = 'cbbf49a5-149b-45fd-a57c-8ea19b95dbd8',
    [string]$CocoRoot = 'E:\BaiduNetdiskDownload\COCOBody',
    [string]$H3wbRoot = 'E:\BaiduNetdiskDownload\H36Wdataset\h36m',
    [string]$H3wbAnn = 'C:\Users\CH\Documents\MyPoseLift\data\raw\h3wb\h3wb_train_bbox.npz',
    [string]$RepoRoot = 'C:\Users\CH\Documents\MyPoseLift'
)

$ErrorActionPreference = 'Stop'
$watchLog = Join-Path $RepoRoot 'work_dirs\coco_mixed_download_watcher.log'
$trainDir = Join-Path $RepoRoot 'work_dirs\rtmw3d-m_68_h3wb-coco-body-256x192_run03_accum4_02'
$trainLog = Join-Path $trainDir 'automatic_train.log'

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -LiteralPath $watchLog -Value $line
    Write-Output $line
}

New-Item -ItemType Directory -Force -Path (Split-Path $watchLog) | Out-Null
Write-Log "watching BITS job $BitsJobId"

while ($true) {
    $job = Get-BitsTransfer -JobId $BitsJobId -ErrorAction Stop
    $percent = if ($job.BytesTotal -gt 0) {
        [math]::Round(100 * $job.BytesTransferred / $job.BytesTotal, 2)
    } else { 0 }
    Write-Log "download state=$($job.JobState) progress=$percent%"

    if ($job.JobState -eq 'Transferred') {
        Complete-BitsTransfer -BitsJob $job
        break
    }
    if ($job.JobState -in @('Error', 'TransientError', 'Cancelled', 'AckNotNeeded')) {
        throw "COCO download failed with BITS state $($job.JobState): $($job.ErrorDescription)"
    }
    Start-Sleep -Seconds 60
}

$zipPath = Join-Path $CocoRoot 'train2017.zip'
$annotationPath = Join-Path $CocoRoot 'annotations\person_keypoints_train2017.json'
$imageDir = Join-Path $CocoRoot 'train2017'
if (-not (Test-Path -LiteralPath $zipPath)) { throw "missing $zipPath" }
if (-not (Test-Path -LiteralPath $annotationPath)) { throw "missing $annotationPath" }

if (-not (Test-Path -LiteralPath $imageDir)) {
    Write-Log 'extracting train2017.zip'
    Expand-Archive -LiteralPath $zipPath -DestinationPath $CocoRoot -Force
}

$imageCount = @(Get-ChildItem -LiteralPath $imageDir -Filter '*.jpg' -File).Count
Write-Log "COCO train images found: $imageCount"
if ($imageCount -lt 100000) {
    throw "COCO image validation failed: expected at least 100000 JPEG files, found $imageCount"
}

$python = Join-Path $RepoRoot '.venv-rtmw3d\Scripts\python.exe'
$config = Join-Path $RepoRoot 'configs\rtmw3d\rtmw3d-m_68_h3wb-256x192.py'
$trainScript = Join-Path $RepoRoot 'external\mmpose\tools\train.py'
$env:H3WB_ROOT = $H3wbRoot
$env:H3WB_ANN = $H3wbAnn
$env:COCO_BODY_ROOT = $CocoRoot
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$env:PYTHONPATH = [string]::Join(';', @(
    $RepoRoot,
    (Join-Path $RepoRoot 'tools\compat'),
    (Join-Path $RepoRoot 'external\mmpose'),
    (Join-Path $RepoRoot 'external\mmpose\projects\rtmpose3d')
))

New-Item -ItemType Directory -Force -Path $trainDir | Out-Null
Write-Log "starting mixed training in $trainDir"
& $python $trainScript $config --work-dir $trainDir `
    --cfg-options `
    train_dataloader.batch_size=4 `
    val_dataloader.batch_size=4 `
    train_dataloader.num_workers=2 `
    val_dataloader.num_workers=2 `
    optim_wrapper.accumulative_counts=4 `
    param_scheduler.0.end=100 2>&1 | Tee-Object -FilePath $trainLog
$exitCode = $LASTEXITCODE
Write-Log "training exited with code $exitCode"
exit $exitCode
