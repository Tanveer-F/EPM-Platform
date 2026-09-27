[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Common.ps1')
$build = Build-EpmInfrastructure
$template = Get-Content -LiteralPath $build.Template -Raw | ConvertFrom-Json
function Get-TemplateResources {
    param($Template)
    foreach ($resource in $Template.resources) {
        $resource
        if ($resource.type -eq 'Microsoft.Resources/deployments') {
            Get-TemplateResources -Template $resource.properties.template
        }
    }
}
$resources = @(Get-TemplateResources -Template $template)
$assertions = 0
function Assert-Local {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "Infrastructure contract failed: $Message" }
    $script:assertions += 1
}
$allowed = @('Microsoft.Resources/resourceGroups', 'Microsoft.Resources/deployments', 'Microsoft.Storage/storageAccounts', 'Microsoft.Storage/storageAccounts/blobServices', 'Microsoft.KeyVault/vaults', 'Microsoft.MachineLearningServices/workspaces', 'Microsoft.MachineLearningServices/workspaces/computes', 'Microsoft.Insights/components', 'Microsoft.OperationalInsights/workspaces', 'Microsoft.Insights/actionGroups')
Assert-Local (@($resources | Where-Object { $_.type -notin $allowed }).Count -eq 0) 'Only Phase 1 resource types may be created.'
foreach ($type in $allowed | Where-Object { $_ -ne 'Microsoft.Resources/deployments' }) {
    Assert-Local (@($resources | Where-Object { $_.type -eq $type }).Count -eq 1) "Exactly one $type is expected."
}
$storage = ($resources | Where-Object { $_.type -eq 'Microsoft.Storage/storageAccounts' }).properties
$vault = ($resources | Where-Object { $_.type -eq 'Microsoft.KeyVault/vaults' }).properties
$workspace = $resources | Where-Object { $_.type -eq 'Microsoft.MachineLearningServices/workspaces' }
$compute = $resources | Where-Object { $_.type -eq 'Microsoft.MachineLearningServices/workspaces/computes' }
$cpu = $compute.properties.properties
Assert-Local ($storage.allowSharedKeyAccess -eq $false) 'Shared keys disabled.'
Assert-Local ($storage.allowBlobPublicAccess -eq $false) 'Anonymous blob access disabled.'
Assert-Local ($storage.minimumTlsVersion -eq 'TLS1_2' -and $storage.supportsHttpsTrafficOnly -eq $true) 'TLS and HTTPS enforced.'
Assert-Local ($storage.isHnsEnabled -eq $false -and $storage.allowCrossTenantReplication -eq $false) 'Default workspace storage compatibility and tenant isolation.'
Assert-Local ($vault.enableRbacAuthorization -and $vault.enableSoftDelete -and $vault.enablePurgeProtection) 'Vault authorization and retention protections.'
Assert-Local ($vault.accessPolicies.Count -eq 0) 'No legacy Key Vault access policies.'
Assert-Local ($workspace.identity.type -eq 'SystemAssigned' -and $compute.identity.type -eq 'SystemAssigned') 'Managed identities enabled.'
Assert-Local ($workspace.properties.systemDatastoresAuthMode -eq 'Identity') 'Identity-based system storage.'
Assert-Local ($workspace.properties.managedNetwork.isolationMode -eq 'AllowInternetOutbound') 'Approved managed compute network.'
Assert-Local ($workspace.properties.v1LegacyMode -eq $false) 'No Azure ML v1 legacy mode.'
$appInsights = ($resources | Where-Object { $_.type -eq 'Microsoft.Insights/components' }).properties
$logs = ($resources | Where-Object { $_.type -eq 'Microsoft.OperationalInsights/workspaces' }).properties
Assert-Local ([bool]$workspace.properties.applicationInsights -and [bool]$appInsights.WorkspaceResourceId) 'Approved workspace telemetry dependencies linked.'
Assert-Local ($appInsights.DisableLocalAuth -eq $true -and $logs.features.disableLocalAuth -eq $true) 'Telemetry dependencies require Entra authentication.'
Assert-Local ($logs.retentionInDays -eq 30 -and $logs.workspaceCapping.dailyQuotaGb -eq 1) 'Telemetry development retention and ingestion safeguard.'
$defaultAlerts = ($resources | Where-Object { $_.type -eq 'Microsoft.Insights/actionGroups' }).properties
Assert-Local ($defaultAlerts.enabled -eq $false -and $defaultAlerts.armRoleReceivers.Count -eq 0 -and $defaultAlerts.emailReceivers.Count -eq 0 -and $defaultAlerts.smsReceivers.Count -eq 0 -and $defaultAlerts.webhookReceivers.Count -eq 0) 'Platform-default action group disabled and recipient-free.'
Assert-Local ($compute.properties.computeType -eq 'AmlCompute' -and $cpu.vmSize -eq 'Standard_D2s_v3') 'CPU cluster, not GPU or compute instance.'
Assert-Local ($cpu.scaleSettings.minNodeCount -eq 0 -and $cpu.scaleSettings.maxNodeCount -eq 1) 'Zero idle nodes, one-node maximum.'
Assert-Local ($cpu.scaleSettings.nodeIdleTimeBeforeScaleDown -eq 'PT5M') 'Five-minute idle scale-down.'
Assert-Local ($cpu.enableNodePublicIp -eq $false -and $cpu.remoteLoginPortPublicAccess -eq 'Disabled') 'No node public IP or public SSH.'
Assert-Local ($compute.properties.disableLocalAuth -eq $true) 'No compute local authentication.'
Assert-Local ($template.parameters.acknowledgePublicDevelopmentEndpoints.allowedValues[0] -eq $true) 'Development public-access acknowledgement required.'
Assert-Local (-not ($template.parameters.acknowledgePublicDevelopmentEndpoints.PSObject.Properties.Name -contains 'defaultValue')) 'No implicit public-access acknowledgement.'
foreach ($script in Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1') {
    $tokens = $null; $errors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile($script.FullName, [ref]$tokens, [ref]$errors)
    Assert-Local ($errors.Count -eq 0) "Valid PowerShell syntax: $($script.Name)"
}
Write-Host "PASS: $assertions infrastructure and script checks; Bicep compilation exit 0."
