Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-AzureJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & az @Arguments --only-show-errors --output json
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI operation failed: $($Arguments[0]). Review the preceding Azure error."
    }
    if ($output) { return ($output | Out-String | ConvertFrom-Json) }
}

function Get-EpmAccount {
    if (-not $env:AZURE_SUBSCRIPTION_ID) { throw 'Initialize the environment first.' }
    $account = Invoke-AzureJson -Arguments @('account', 'show', '--subscription', $env:AZURE_SUBSCRIPTION_ID)
    if ($account.state -ne 'Enabled') { throw 'The selected subscription is not enabled.' }
    if ($account.environmentName -ne 'AzureCloud') { throw 'Phase 1 is configured for Azure public cloud only.' }
    return $account
}

function Build-EpmInfrastructure {
    $stateDirectory = Join-Path $script:ProjectRoot '.azure'
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    $template = Join-Path $stateDirectory 'main.json'
    $parameters = Join-Path $stateDirectory 'parameters.json'
    & az bicep build --file (Join-Path $script:ProjectRoot 'infra\main.bicep') --outfile $template --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Bicep template compilation failed.' }
    & az bicep build-params --file (Join-Path $script:ProjectRoot 'infra\main.dev.bicepparam') --outfile $parameters --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Bicep parameter compilation failed.' }
    return @{ Template = $template; Parameters = $parameters }
}

function Assert-EpmConfiguration {
    param([Parameter(Mandatory = $true)][string]$ParametersPath)
    $parameters = (Get-Content -LiteralPath $ParametersPath -Raw | ConvertFrom-Json).parameters
    $project = $parameters.projectName.value
    $environment = $parameters.environmentName.value
    $location = $parameters.location.value
    if ($project -notmatch '^[a-z][a-z0-9]{1,7}$') { throw 'Project name must be 2-8 lowercase letters/digits, starting with a letter.' }
    if ($environment -ne 'dev' -or -not $parameters.acknowledgePublicDevelopmentEndpoints.value) {
        throw 'Only the explicitly approved public development foundation is supported.'
    }
    if ($env:AZURE_RESOURCE_GROUP -ne "rg-$project-$environment-$location" -or
        $env:AZURE_ML_WORKSPACE -ne "mlw-$project-$environment-$location" -or
        $env:AZURE_ML_COMPUTE -ne 'cpu-dev' -or $env:AZURE_LOCATION -ne $location) {
        throw 'Environment configuration and infra\main.dev.bicepparam do not match. No deployment was started.'
    }
    if ($env:AZURE_ML_WORKSPACE.Length -gt 33) { throw 'Generated workspace name exceeds 33 characters.' }
}
