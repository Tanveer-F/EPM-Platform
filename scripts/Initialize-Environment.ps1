[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Common.ps1')
$allowed = @('AZURE_SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP', 'AZURE_ML_WORKSPACE', 'AZURE_ML_COMPUTE', 'AZURE_LOCATION', 'EPM_AUTH_MODE', 'AZURE_CLIENT_ID')
$configuration = @{}
foreach ($path in @((Join-Path $script:ProjectRoot 'config\.env.example'), (Join-Path $script:ProjectRoot '.env.local'))) {
    if (-not (Test-Path -LiteralPath $path)) { continue }
    foreach ($line in Get-Content -LiteralPath $path) {
        $line = $line.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        if ($line -notmatch '^([A-Z][A-Z0-9_]*)=(.*)$' -or $Matches[1] -notin $allowed) {
            throw 'Invalid or unsupported configuration entry. Use simple KEY=value lines from config\.env.example.'
        }
        $configuration[$Matches[1]] = $Matches[2].Trim()
    }
}
foreach ($key in $configuration.Keys) {
    if (-not [Environment]::GetEnvironmentVariable($key, 'Process') -and $configuration[$key]) {
        [Environment]::SetEnvironmentVariable($key, $configuration[$key], 'Process')
    }
}
if (-not $env:AZURE_SUBSCRIPTION_ID) {
    if ($env:EPM_AUTH_MODE -ne 'azure-cli') { throw 'Managed-identity mode requires AZURE_SUBSCRIPTION_ID in the process environment.' }
    $account = Invoke-AzureJson -Arguments @('account', 'show')
    $env:AZURE_SUBSCRIPTION_ID = $account.id
}
$subscriptionGuid = [guid]::Empty
if (-not [guid]::TryParse($env:AZURE_SUBSCRIPTION_ID, [ref]$subscriptionGuid) -or $subscriptionGuid -eq [guid]::Empty) {
    throw 'AZURE_SUBSCRIPTION_ID must be a valid nonempty UUID.'
}
Write-Host 'Environment loaded; subscription identifier retained only in process environment.'
