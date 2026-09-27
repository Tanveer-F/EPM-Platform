[CmdletBinding()]
param(
    [ValidateSet('Validate', 'WhatIf', 'Deploy')][string]$Action = 'WhatIf',
    [switch]$ApproveCosts
)

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$account = Get-EpmAccount
$build = Build-EpmInfrastructure
Assert-EpmConfiguration -ParametersPath $build.Parameters
if ($Action -eq 'Deploy' -and -not $ApproveCosts) {
    throw 'Deployment requires -ApproveCosts after reviewing WhatIf, the development network exception, and ongoing charges.'
}
$arguments = @(
    'deployment', 'sub',
    '--subscription', $env:AZURE_SUBSCRIPTION_ID,
    '--location', $env:AZURE_LOCATION,
    '--name', 'epm-phase1-dev',
    '--template-file', $build.Template,
    '--parameters', ('@' + $build.Parameters)
)
Write-Host "Target: $($account.name) / $env:AZURE_RESOURCE_GROUP / $env:AZURE_LOCATION"
$null = Invoke-AzureJson -Arguments ($arguments[0..1] + @('validate') + $arguments[2..($arguments.Length - 1)])
Write-Host 'PASS: Azure Resource Manager validation.'
if ($Action -eq 'Validate') { return }
$preview = Invoke-AzureJson -Arguments ($arguments[0..1] + @('what-if', '--no-pretty-print') + $arguments[2..($arguments.Length - 1)])
$preview | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $script:ProjectRoot '.azure\what-if.json') -Encoding UTF8
if ($preview.status -ne 'Succeeded') { throw 'Azure WhatIf did not succeed; deployment was not started.' }
$changes = @($preview.changes)
foreach ($change in $changes) {
    $relativeId = $change.resourceId -replace '^/subscriptions/[^/]+', '<subscription>'
    Write-Host "$($change.changeType): $relativeId"
}
if (@($changes | Where-Object { $_.changeType -eq 'Delete' }).Count -gt 0) { throw 'Deletion detected. This Phase 1 workflow refuses destructive deployment.' }
if ($Action -eq 'WhatIf') { return }
$result = Invoke-AzureJson -Arguments ($arguments[0..1] + @('create') + $arguments[2..($arguments.Length - 1)])
if ($result.properties.provisioningState -ne 'Succeeded') { throw 'Deployment did not report Succeeded.' }
$result.properties.outputs | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $script:ProjectRoot '.azure\deployment-outputs.json') -Encoding UTF8
Write-Host 'PASS: deployment succeeded. Run scripts\Verify-Foundation.ps1 before declaring Phase 1 complete.'
