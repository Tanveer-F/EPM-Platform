[CmdletBinding()]
param(
    [ValidateSet('WhatIf', 'Deploy')][string]$Action = 'WhatIf',
    [switch]$ApproveAccessChanges
)

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$account = Get-EpmAccount
if ($env:EPM_AUTH_MODE -ne 'azure-cli' -or $account.user.type -ne 'user') {
    throw 'This provisioning script grants only the current approved interactive publishing user. Review another identity separately.'
}
if ($Action -eq 'Deploy' -and -not $ApproveAccessChanges) {
    throw 'Container creation and scoped RBAC grants require explicit -ApproveAccessChanges.'
}
$workspace = Invoke-AzureJson -Arguments @('ml', 'workspace', 'show', '--subscription', $env:AZURE_SUBSCRIPTION_ID, '--resource-group', $env:AZURE_RESOURCE_GROUP, '--name', $env:AZURE_ML_WORKSPACE)
$storageId = $workspace.storage_account
$prefix = "/subscriptions/$env:AZURE_SUBSCRIPTION_ID/resourceGroups/$env:AZURE_RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/"
if (-not $storageId.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Workspace storage is outside the approved existing subscription/resource group.'
}
$storageName = ($storageId -split '/')[-1]
$storage = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com${storageId}?api-version=2025-01-01")
if ($storage.properties.allowSharedKeyAccess -ne $false -or $storage.properties.allowBlobPublicAccess -ne $false) {
    throw 'Existing storage no longer meets the approved keyless/private-blob foundation.'
}
$publisher = Invoke-AzureJson -Arguments @('ad', 'signed-in-user', 'show', '--query', 'id')
$state = Join-Path $script:ProjectRoot '.azure'
New-Item -ItemType Directory -Path $state -Force | Out-Null
$template = Join-Path $state 'data-access.json'
$parametersPath = Join-Path $state 'data-access.parameters.json'
& az bicep build --file (Join-Path $script:ProjectRoot 'infra\data-access.bicep') --outfile $template --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Phase 2 Bicep compilation failed.' }
@{
    '$schema' = 'https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#'
    contentVersion = '1.0.0.0'
    parameters = @{
        storageAccountName = @{ value = $storageName }
        publisherObjectId = @{ value = $publisher }
    }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $parametersPath -Encoding UTF8
$target = @('--subscription', $env:AZURE_SUBSCRIPTION_ID, '--resource-group', $env:AZURE_RESOURCE_GROUP, '--name', 'epm-phase2-data-access', '--template-file', $template, '--parameters', ('@' + $parametersPath))
Write-Host "Target: $($account.name) / $env:AZURE_RESOURCE_GROUP / existing storage $storageName"
try {
    $null = Invoke-AzureJson -Arguments (@('deployment', 'group', 'validate') + $target)
    $preview = Invoke-AzureJson -Arguments (@('deployment', 'group', 'what-if', '--no-pretty-print') + $target)
    $preview | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $state 'data-access-what-if.json') -Encoding UTF8
    if ($preview.status -ne 'Succeeded') { throw 'Data access preview did not succeed.' }
    foreach ($change in $preview.changes) {
        Write-Host ($change.changeType + ': ' + ($change.resourceId -replace '/subscriptions/[^/]+/', '<subscription>/'))
    }
    if (@($preview.changes | Where-Object { $_.changeType -eq 'Delete' }).Count -gt 0) { throw 'Destructive resource changes refused.' }
    if ($Action -eq 'Deploy') {
        $result = Invoke-AzureJson -Arguments (@('deployment', 'group', 'create', '--mode', 'Incremental') + $target)
        if ($result.properties.provisioningState -ne 'Succeeded') { throw 'Data access deployment failed.' }
        Write-Host 'PASS: two private containers and container-scoped publisher grants configured; no account, compute or service created.'
    }
}
finally {
    # The resolved principal ID is needed only for this invocation, not durable project state.
    if (Test-Path -LiteralPath $parametersPath) { Remove-Item -LiteralPath $parametersPath }
}
