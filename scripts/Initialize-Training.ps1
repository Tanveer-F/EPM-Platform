[CmdletBinding()]
param([ValidateSet('WhatIf','Deploy')][string]$Action='WhatIf', [switch]$ApproveCosts)
. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$null = Get-EpmAccount
if ($Action -eq 'Deploy' -and -not $ApproveCosts) { throw 'Explicit runtime permission approval required.' }
$target = @('--subscription',$env:AZURE_SUBSCRIPTION_ID,'--resource-group',$env:AZURE_RESOURCE_GROUP)
$ws = Invoke-AzureJson -Arguments (@('ml','workspace','show','--name',$env:AZURE_ML_WORKSPACE) + $target)
$store = Invoke-AzureJson -Arguments (@('ml','datastore','show','--name','workspaceblobstore','--workspace-name',$env:AZURE_ML_WORKSPACE) + $target)
$publisher = Invoke-AzureJson -Arguments @('ad','signed-in-user','show','--query','id')
$template = Join-Path $script:ProjectRoot '.azure\training-access.json'
$params = Join-Path $script:ProjectRoot '.azure\training-access.parameters.json'
& az bicep build --file (Join-Path $script:ProjectRoot 'infra\training-access.bicep') --outfile $template --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Training Bicep failed.' }
@{ '$schema'='https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#'; contentVersion='1.0.0.0'; parameters=@{workspaceName=@{value=$env:AZURE_ML_WORKSPACE};computeName=@{value=$env:AZURE_ML_COMPUTE};storageAccountName=@{value=($ws.storage_account -split '/')[-1]};publisherObjectId=@{value=$publisher};jobContainerName=@{value=$store.container_name}}} | ConvertTo-Json -Depth 8 | Set-Content $params -Encoding UTF8
try {
    $args = $target + @('--name','epm-training-access','--template-file',$template,'--parameters',('@'+$params))
    $preview = Invoke-AzureJson -Arguments (@('deployment','group','what-if','--no-pretty-print')+$args)
    $preview | ConvertTo-Json -Depth 100 | Set-Content (Join-Path $script:ProjectRoot '.azure\training-what-if.json') -Encoding UTF8
    if ($preview.status -ne 'Succeeded') { throw 'Training preview failed.' }
    foreach ($change in $preview.changes) { Write-Host ($change.changeType + ': ' + ($change.resourceId -replace '/subscriptions/[^/]+/','<subscription>/')) }
    if (@($preview.changes | Where-Object { $_.changeType -eq 'Delete' }).Count) { throw 'Destructive change refused.' }
    if ($Action -eq 'Deploy') {
        $result = Invoke-AzureJson -Arguments (@('deployment','group','create','--mode','Incremental')+$args)
        if ($result.properties.provisioningState -ne 'Succeeded') { throw 'Training setup failed.' }
        Write-Host 'PASS: scoped runtime grants configured; no registry or infrastructure created.'
    }
} finally { if (Test-Path $params) { Remove-Item -LiteralPath $params } }
