[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Common.ps1')
$state = Join-Path $script:ProjectRoot '.azure'
New-Item -ItemType Directory -Path $state -Force | Out-Null
$templatePath = Join-Path $state 'data-access.json'
& az bicep build --file (Join-Path $script:ProjectRoot 'infra\data-access.bicep') --outfile $templatePath --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Phase 2 Bicep compilation failed.' }
$template = Get-Content -LiteralPath $templatePath -Raw | ConvertFrom-Json
$resources = @($template.resources)
$checks = 0
function Assert-DataInfra {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "Phase 2 infrastructure contract failed: $Message" }
    $script:checks += 1
}
$containers = @($resources | Where-Object { $_.type -eq 'Microsoft.Storage/storageAccounts/blobServices/containers' })
$assignments = @($resources | Where-Object { $_.type -eq 'Microsoft.Authorization/roleAssignments' })
Assert-DataInfra ($resources.Count -eq 4 -and $containers.Count -eq 2 -and $assignments.Count -eq 2) 'Only two containers and two scoped grants may be created.'
foreach ($container in $containers) {
    Assert-DataInfra ($container.properties.publicAccess -eq 'None') 'No public container access.'
    Assert-DataInfra ($container.properties.metadata.dataset -eq 'nasa-cmapss') 'Dataset-specific namespace.'
}
Assert-DataInfra (@($containers | Where-Object { $_.properties.metadata.zone -eq 'raw' }).Count -eq 1) 'One raw container.'
Assert-DataInfra (@($containers | Where-Object { $_.properties.metadata.zone -eq 'curated' }).Count -eq 1) 'One curated container.'
foreach ($assignment in $assignments) {
    Assert-DataInfra ($assignment.scope -match "resourceId\('Microsoft.Storage/storageAccounts/blobServices/containers'" -and $assignment.scope -match 'epm-cmapss-') 'Publisher grant scoped to a dataset container, not the account or resource group.'
    Assert-DataInfra ($assignment.properties.principalId -eq "[parameters('publisherObjectId')]" -and $assignment.properties.principalType -eq 'User') 'Only the approved runtime-resolved publishing user receives grants.'
    Assert-DataInfra ($assignment.properties.roleDefinitionId -match 'blobContributorRole') 'Only Blob Data Contributor is assigned.'
}
Assert-DataInfra ($template.variables.blobContributorRole -match 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') 'Correct built-in data role.'
Assert-DataInfra ($template.parameters.publisherObjectId.type -eq 'string' -and -not ($template.parameters.publisherObjectId.PSObject.Properties.Name -contains 'defaultValue')) 'Non-secret principal identifier is required at runtime, never hardcoded.'
foreach ($path in @('Initialize-DataStorage.ps1', 'Test-DataInfrastructure.ps1')) {
    $tokens = $null; $errors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $path), [ref]$tokens, [ref]$errors)
    Assert-DataInfra ($errors.Count -eq 0) "Valid PowerShell: $path"
}
Write-Host "PASS: $checks Phase 2 infrastructure checks; no new Azure service or compute."
