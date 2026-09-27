[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$JobName,
    [Parameter(Mandatory=$true)][string]$ArtifactsPath
)
. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$uri=Invoke-AzureJson -Arguments @('ml','workspace','show','--subscription',$env:AZURE_SUBSCRIPTION_ID,'--resource-group',$env:AZURE_RESOURCE_GROUP,'--name',$env:AZURE_ML_WORKSPACE,'--query','mlflow_tracking_uri')
if (-not $uri.StartsWith('azureml://')) { throw 'Azure ML tracking endpoint unavailable.' }
$env:MLFLOW_TRACKING_URI=$uri
$env:MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR='false'
$python=Join-Path $script:ProjectRoot '.venv-torch\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Install the isolated PyTorch/MLflow runtime first.' }
& $python (Join-Path $script:ProjectRoot 'scripts\verify_pytorch_tracking.py') --job-name $JobName --artifacts (Resolve-Path $ArtifactsPath).Path --receipt (Join-Path $script:ProjectRoot '.azure\pytorch-mlflow-verification.json')
if ($LASTEXITCODE -ne 0) { throw 'Independent MLflow verification failed; no cloud writes were performed.' }
