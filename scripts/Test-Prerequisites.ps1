[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
foreach ($command in @('git', 'py', 'az')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Required command missing: $command" }
}
& git --version
if ($LASTEXITCODE -ne 0) { throw 'Git verification failed.' }
& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 with: py install 3.12' }
& az bicep version
if ($LASTEXITCODE -ne 0) { throw 'Install Bicep with: az bicep install' }
$versions = Invoke-AzureJson -Arguments @('version')
Write-Host "Azure CLI: $($versions.'azure-cli')"
$extension = Invoke-AzureJson -Arguments @('extension', 'show', '--name', 'ml')
if ([version]$extension.version -lt [version]'2.0.0') { throw 'Azure ML CLI v2 is required.' }
Write-Host "Azure ML CLI v2: $($extension.version)"
$account = Get-EpmAccount
$expiry = Invoke-AzureJson -Arguments @('account', 'get-access-token', '--subscription', $env:AZURE_SUBSCRIPTION_ID, '--resource', 'https://management.azure.com/', '--query', 'expires_on')
if (-not $expiry) { throw 'Azure authentication did not return token metadata.' }
Write-Host "Authenticated subscription: $($account.name)"
foreach ($provider in @('Microsoft.MachineLearningServices', 'Microsoft.Storage', 'Microsoft.KeyVault', 'Microsoft.ManagedIdentity', 'Microsoft.Network', 'Microsoft.Compute', 'Microsoft.Insights', 'Microsoft.OperationalInsights')) {
    $state = Invoke-AzureJson -Arguments @('provider', 'show', '--namespace', $provider, '--subscription', $env:AZURE_SUBSCRIPTION_ID, '--query', 'registrationState')
    if ($state -ne 'Registered') { throw "Register provider $provider with an authorized administrator before deployment." }
}
$build = Build-EpmInfrastructure
Assert-EpmConfiguration -ParametersPath $build.Parameters
Write-Host 'PASS: tooling, authentication, provider registration and configuration.'
