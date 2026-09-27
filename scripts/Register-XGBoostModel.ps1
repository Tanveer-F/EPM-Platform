[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$null = Get-EpmAccount
$target = @('--subscription', $env:AZURE_SUBSCRIPTION_ID, '--resource-group', $env:AZURE_RESOURCE_GROUP, '--workspace-name', $env:AZURE_ML_WORKSPACE)
Write-Host 'Verifying completed source job.'
$jobOutput = & az ml job show --subscription $env:AZURE_SUBSCRIPTION_ID --resource-group $env:AZURE_RESOURCE_GROUP --workspace-name $env:AZURE_ML_WORKSPACE --name epm-baseline-de82ea3141be --only-show-errors --output json
if ($LASTEXITCODE -ne 0) { throw 'The completed source job could not be verified.' }
$job = $jobOutput | ConvertFrom-Json
if ($job.name -ne 'epm-baseline-de82ea3141be' -or $job.status -ne 'Completed' -or
    $job.outputs.baseline.path -ne 'azureml://datastores/workspaceblobstore/paths/baseline/epm-baseline-de82ea3141be/') {
    throw 'The approved source baseline job/output is not complete; registration refused.'
}

$source = Join-Path $script:ProjectRoot 'artifacts\baseline\epm-baseline-de82ea3141be'
$manifestPath = Join-Path $source 'artifact-manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.artifact_type -ne 'xgboost-rul-baseline' -or
    $manifest.provenance.ml_ready_manifest_sha256 -ne '2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd' -or
    (Get-FileHash -LiteralPath (Join-Path $source 'model.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe') {
    throw 'Local source bundle failed model, source-dataset, or artifact identity checks.'
}
$metricFile = Get-Content -LiteralPath (Join-Path $source 'metrics.json') -Raw | ConvertFrom-Json
if ([math]::Abs($metricFile.test.overall.rmse - 30.80716678557037) -gt 0.000000001 -or
    [math]::Abs($metricFile.test.overall.nasa_score - 51555.97851104609) -gt 0.000001) {
    throw 'The reviewed evaluation differs from the approved model selection.'
}

$workspaceOutput = & az ml workspace show --subscription $env:AZURE_SUBSCRIPTION_ID --resource-group $env:AZURE_RESOURCE_GROUP --name $env:AZURE_ML_WORKSPACE --only-show-errors --output json
if ($LASTEXITCODE -ne 0) { throw 'The Azure ML workspace could not be verified.' }
$workspace = $workspaceOutput | ConvertFrom-Json
$env:MLFLOW_TRACKING_URI = $workspace.mlflow_tracking_uri
if (-not $env:MLFLOW_TRACKING_URI.StartsWith('azureml://')) { throw 'Azure MLflow tracking is unavailable.' }
$isolatedPython = Join-Path $script:ProjectRoot '.venv-torch\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $isolatedPython)) { throw 'The isolated Azure MLflow client runtime is missing.' }
$backfillReceipt = Join-Path $script:ProjectRoot '.azure\baseline-mlflow-backfill.json'
Write-Host 'Verifying existing MLflow source and backfilling verified run evidence without retraining.'
$env:PYTHONUTF8 = '1'
& $isolatedPython (Join-Path $script:ProjectRoot 'scripts\backfill_baseline_mlflow.py') --source $source --receipt $backfillReceipt
if ($LASTEXITCODE -ne 0) { throw 'The source run metadata/artifact backfill was not verified; model registration refused.' }
$backfillRunId = (Get-Content -LiteralPath $backfillReceipt -Raw | ConvertFrom-Json).backfill_run_id
if (-not $backfillRunId) { throw 'The verified MLflow metadata backfill run ID is missing.' }

$modelName = 'epm-cmapss-rul-xgboost'
Write-Host 'Checking workspace model versions.'
$listArgs = @('ml', 'model', 'list') + $target + @('--only-show-errors', '--output', 'json')
$listOutput = & az @listArgs
if ($LASTEXITCODE -ne 0) { throw 'The workspace model inventory could not be safely checked.' }
$listJson = $listOutput | Out-String
if ($listJson.Trim() -eq '[]') {
    $existing = @()
} else {
    $models = ConvertFrom-Json -InputObject $listJson
    $existing = @($models | Where-Object { $_.name -eq $modelName })
}
if (@($existing).Count -gt 0) {
    if (@($existing).Count -ne 1 -or [string]$existing[0].'latest version' -ne '1') {
        throw 'The selected model already has unexpected versions; no version was overwritten.'
    }
    $registered = Invoke-AzureJson -Arguments (@('ml', 'model', 'show', '--name', $modelName, '--version', '1') + $target)
} else {
    $modelDescription = 'Validated pooled XGBoost CPU RUL model, selected over the PyTorch MLP on identical test engines by lower RMSE and substantially lower asymmetric NASA score. Native model artifact; no endpoint or failure-probability claim.'
    $modelTags = @(
        'framework=xgboost',
        'model_family=gradient_boosted_trees',
        'model_format=native_xgboost_json',
        'target=uncapped_remaining_useful_life_cycles',
        'selection=test_rmse_and_asymmetric_nasa_score',
        'source_training_job=epm-baseline-de82ea3141be',
        'source_mlflow_run_id=epm-baseline-de82ea3141be',
        'source_mlflow_experiment=epm-baseline-rul',
        "metadata_mlflow_backfill_run_id=$backfillRunId",
        'ml_ready_asset_name=epm-cmapss-ml-ready',
        'ml_ready_asset_version=d-f4ueae6u7g4c5islgehonqveey',
        'ml_ready_manifest_sha256=2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd',
        'baseline_model_sha256=e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe',
        'baseline_artifact_manifest_sha256=c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09',
        'code_sha256=fca4f48ea125278a9b6089701970b6d5a150a4921bfaff4a896202b366bea2c5',
        'environment=azureml-curated-sklearn-1.5-54-python312-runtime',
        'runtime=python-3.12.10-xgboost-cpu-3.4.1',
        'test_rmse_cycles=30.80716678557037',
        'test_mae_cycles=25.481298365721102',
        'test_r2=0.6361424319583523',
        'test_bias_cycles=11.393868892836942',
        'test_nasa_score_sum=51555.97851104609',
        'test_nasa_score_mean=72.92217611180493',
        'validation_rmse_cycles=31.3611565432057',
        'validation_mae_cycles=25.51895052950147',
        'validation_r2=-0.5336506176801772',
        'comparison_model_job=epm-pytorch-d97b75fd1f65',
        'comparison_model_test_rmse_cycles=31.932091950899398',
        'comparison_model_test_mae_cycles=23.842123510696123',
        'comparison_model_test_nasa_score_sum=316736.2852433281'
    )
    $modelPath = 'azureml://jobs/epm-baseline-de82ea3141be/outputs/baseline/paths/'
    $registered = Invoke-AzureJson -Arguments (@(
        'ml', 'model', 'create',
        '--name', $modelName,
        '--version', '1',
        '--type', 'custom_model',
        '--path', $modelPath,
        '--description', $modelDescription,
        '--tags'
    ) + $modelTags + $target)
}
if ($registered.name -ne $modelName -or [string]$registered.version -ne '1' -or $registered.type -ne 'custom_model' -or
    $registered.tags.source_training_job -ne 'epm-baseline-de82ea3141be' -or
    $registered.tags.metadata_mlflow_backfill_run_id -ne $backfillRunId -or
    $registered.tags.ml_ready_manifest_sha256 -ne '2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd' -or
    $registered.tags.baseline_model_sha256 -ne 'e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe') {
    throw 'The retrieved registry asset/version metadata does not match the source lineage and metrics.'
}
if ($registered.job_name -ne 'epm-baseline-de82ea3141be' -or
    $registered.path -notmatch '/datastores/workspaceblobstore/paths/baseline/epm-baseline-de82ea3141be/$') {
    throw 'The model asset no longer references the original Azure ML job output.'
}

$downloadRoot = Join-Path $script:ProjectRoot '.azure\registered-model-verification'
if (Test-Path -LiteralPath $downloadRoot) { throw 'Model verification destination already exists; inspect it before retrying.' }
New-Item -ItemType Directory -Path $downloadRoot | Out-Null
$null = Invoke-AzureJson -Arguments (@('ml', 'model', 'download', '--name', $modelName, '--version', '1', '--download-path', $downloadRoot) + $target)
$downloadedManifest = @(Get-ChildItem -LiteralPath $downloadRoot -Filter 'artifact-manifest.json' -Recurse -File)
if ($downloadedManifest.Count -ne 1) { throw 'The registered model download did not contain one artifact manifest.' }
$downloadedRoot = $downloadedManifest[0].Directory.FullName
$downloadManifest = Get-Content -LiteralPath $downloadedManifest[0].FullName -Raw | ConvertFrom-Json
if ($downloadManifest.artifact_type -ne 'xgboost-rul-baseline' -or
    $downloadManifest.provenance.ml_ready_manifest_sha256 -ne '2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd') {
    throw 'The downloaded registry asset has invalid model or dataset provenance.'
}
foreach ($file in $downloadManifest.files) {
    $path = Join-Path $downloadedRoot $file.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Item -LiteralPath $path).Length -ne $file.size_bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw 'A downloaded registered-model artifact failed its size or SHA-256 check.'
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $downloadedRoot 'artifact-manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    'c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09') {
    throw 'The registered artifact manifest differs from the validated Phase 4 output.'
}

$receiptPath = Join-Path $script:ProjectRoot '.azure\registered-xgboost-model.json'
$receipt = [ordered]@{
    status = 'verified'
    model_name = $modelName
    model_version = '1'
    model_type = 'custom_model'
    source_training_job = 'epm-baseline-de82ea3141be'
    source_mlflow_run_id = 'epm-baseline-de82ea3141be'
    metadata_mlflow_run_id = $backfillRunId
    ml_ready_asset = 'epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey'
    ml_ready_manifest_sha256 = '2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd'
    model_sha256 = 'e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe'
    artifact_manifest_sha256 = 'c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09'
    test_rmse_cycles = 30.80716678557037
    test_mae_cycles = 25.481298365721102
    test_r2 = 0.6361424319583523
    test_nasa_score_sum = 51555.97851104609
    downloaded_files_verified = $downloadManifest.files.Count + 1
    registration_uri_verified = $true
    source_run_uri_verified = $true
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Host "PASS: $modelName version 1 registered from the completed Azure ML run and downloaded with SHA-256 verification."
