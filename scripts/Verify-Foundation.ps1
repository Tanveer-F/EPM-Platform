[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$null = Get-EpmAccount
$checks = [System.Collections.Generic.List[object]]::new()
function Assert-Foundation {
    param([string]$Name, [bool]$Condition)
    $checks.Add([pscustomobject]@{ check = $Name; passed = $Condition })
    if ($Condition) { Write-Host "PASS: $Name" } else { Write-Host "FAIL: $Name" }
}
$target = @('--subscription', $env:AZURE_SUBSCRIPTION_ID, '--resource-group', $env:AZURE_RESOURCE_GROUP)
$workspace = Invoke-AzureJson -Arguments (@('ml', 'workspace', 'show', '--name', $env:AZURE_ML_WORKSPACE) + $target)
$compute = Invoke-AzureJson -Arguments (@('ml', 'compute', 'show', '--name', $env:AZURE_ML_COMPUTE, '--workspace-name', $env:AZURE_ML_WORKSPACE) + $target)
$workspaceArm = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($workspace.id)?api-version=2025-06-01")
$computeArm = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($compute.id)?api-version=2025-06-01")
$properties = $workspaceArm.properties
$cpu = $computeArm.properties.properties
$storage = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($properties.storageAccount)?api-version=2025-01-01")
$vault = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($properties.keyVault)?api-version=2024-11-01")
$appInsights = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($properties.applicationInsights)?api-version=2020-02-02", '--query', '{id:id,properties:{workspaceResourceId:properties.WorkspaceResourceId,disableLocalAuth:properties.DisableLocalAuth}}')
$logs = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($appInsights.properties.workspaceResourceId)?api-version=2025-02-01")
Assert-Foundation 'Workspace telemetry dependencies linked with Entra authentication' ($appInsights.properties.disableLocalAuth -eq $true -and $logs.properties.features.disableLocalAuth -eq $true)
Assert-Foundation 'Telemetry development retention and ingestion safeguard' ($logs.properties.retentionInDays -eq 30 -and $logs.properties.workspaceCapping.dailyQuotaGb -eq 1)
Assert-Foundation 'Workspace provisioned' ($properties.provisioningState -eq 'Succeeded')
Assert-Foundation 'CPU compute provisioned' ($computeArm.properties.provisioningState -eq 'Succeeded')
Assert-Foundation 'Identity-based system datastores' ($properties.systemDatastoresAuthMode -eq 'Identity')
Assert-Foundation 'Separate system-assigned managed identities' (
    $workspaceArm.identity.type -eq 'SystemAssigned' -and $computeArm.identity.type -eq 'SystemAssigned' -and
    $workspaceArm.identity.principalId -and $computeArm.identity.principalId -and
    $workspaceArm.identity.principalId -ne $computeArm.identity.principalId)
Assert-Foundation 'Managed outbound network configured' ($properties.managedNetwork.isolationMode -eq 'AllowInternetOutbound')
Assert-Foundation 'Approved public authenticated development service endpoints' (
    $properties.publicNetworkAccess -eq 'Enabled' -and $storage.properties.publicNetworkAccess -eq 'Enabled' -and $vault.properties.publicNetworkAccess -eq 'Enabled')
Assert-Foundation 'Storage keys and anonymous blobs disabled' ($storage.properties.allowSharedKeyAccess -eq $false -and $storage.properties.allowBlobPublicAccess -eq $false)
Assert-Foundation 'HTTPS and TLS 1.2 storage' ($storage.properties.supportsHttpsTrafficOnly -eq $true -and $storage.properties.minimumTlsVersion -eq 'TLS1_2')
Assert-Foundation 'Key Vault RBAC, soft delete and purge protection enabled' ($vault.properties.enableRbacAuthorization -eq $true -and $vault.properties.enableSoftDelete -eq $true -and $vault.properties.enablePurgeProtection -eq $true)
Assert-Foundation 'CPU-only two-core development SKU' ($computeArm.properties.computeType -eq 'AmlCompute' -and $cpu.vmSize -eq 'Standard_D2s_v3' -and $cpu.osType -eq 'Linux' -and $cpu.vmPriority -eq 'Dedicated')
Assert-Foundation 'Compute scales to zero with a one-node cap' ($cpu.scaleSettings.minNodeCount -eq 0 -and $cpu.scaleSettings.maxNodeCount -eq 1 -and [System.Xml.XmlConvert]::ToTimeSpan($cpu.scaleSettings.nodeIdleTimeBeforeScaleDown).TotalSeconds -eq 300)
Assert-Foundation 'Node public IP and public SSH disabled' ($cpu.enableNodePublicIp -eq $false -and $cpu.remoteLoginPortPublicAccess -eq 'Disabled')
Assert-Foundation 'Compute local authentication disabled' ($computeArm.properties.disableLocalAuth -eq $true)
$roles = @(Invoke-AzureJson -Arguments @('role', 'assignment', 'list', '--assignee-object-id', $workspaceArm.identity.principalId, '--all', '--subscription', $env:AZURE_SUBSCRIPTION_ID))
$storageRoles = @($roles | Where-Object { $_.scope -eq $storage.id })
$vaultRoles = @($roles | Where-Object { $_.scope -eq $vault.id })
Assert-Foundation 'Workspace identity dependency grants present' ($storageRoles.Count -gt 0 -and $vaultRoles.Count -gt 0)
Write-Host ('Workspace storage role names: ' + (($storageRoles | ForEach-Object { $_.roleDefinitionName }) -join ', '))
Write-Host ('Workspace vault role names: ' + (($vaultRoles | ForEach-Object { $_.roleDefinitionName }) -join ', '))
$inventory = @(Invoke-AzureJson -Arguments (@('resource', 'list') + $target))
$unexpected = @($inventory | Where-Object { $_.type -notin @('Microsoft.Storage/storageAccounts', 'Microsoft.KeyVault/vaults', 'Microsoft.MachineLearningServices/workspaces', 'Microsoft.MachineLearningServices/workspaces/computes', 'Microsoft.Insights/components', 'Microsoft.OperationalInsights/workspaces', 'Microsoft.Insights/actionGroups') })
Assert-Foundation 'No unexpected application services in the project resource group' ($unexpected.Count -eq 0)
$defaultGroups = @($inventory | Where-Object { $_.type -eq 'Microsoft.Insights/actionGroups' -and $_.name -eq 'Application Insights Smart Detection' })
Assert-Foundation 'Only the managed disabled default action group exists' ($defaultGroups.Count -eq 1 -and @($inventory | Where-Object { $_.type -eq 'Microsoft.Insights/actionGroups' }).Count -eq 1)
if ($defaultGroups.Count -eq 1) {
    $actionGroup = Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com$($defaultGroups[0].id)?api-version=2023-01-01")
    Assert-Foundation 'Default action group disabled and recipient-free' ($actionGroup.properties.enabled -eq $false -and $actionGroup.properties.armRoleReceivers.Count -eq 0 -and $actionGroup.properties.emailReceivers.Count -eq 0 -and $actionGroup.properties.smsReceivers.Count -eq 0 -and $actionGroup.properties.webhookReceivers.Count -eq 0)
}
$python = Join-Path $script:ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the project-local Python environment before SDK verification.' }
& $python -m epm_platform.verify
Assert-Foundation 'Azure ML SDK v2 verification succeeded' ($LASTEXITCODE -eq 0)
$report = [pscustomobject]@{
    verifiedAtUtc = [DateTime]::UtcNow.ToString('o')
    workspace = $env:AZURE_ML_WORKSPACE
    compute = $env:AZURE_ML_COMPUTE
    checks = $checks.ToArray()
    workloadsSubmitted = $false
    note = 'Read-only foundation validation. Zero-node success does not prove future job execution, node capacity or data access.'
}
New-Item -ItemType Directory -Path (Join-Path $script:ProjectRoot '.azure') -Force | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $script:ProjectRoot '.azure\verification.json') -Encoding UTF8
if (@($checks | Where-Object { -not $_.passed }).Count -gt 0) { throw 'Foundation verification failed; see .azure\verification.json.' }
Write-Host 'PASS: read-only foundation verification. No data, models, jobs or experiments were created.'
