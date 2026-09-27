[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'Common.ps1')
$python = Join-Path $script:ProjectRoot '.tools\iac\Scripts\python.exe'
$entryPoint = Join-Path $script:ProjectRoot '.tools\iac\Scripts\checkov'
if (-not (Test-Path -LiteralPath $entryPoint)) {
    throw 'Install isolated IaC tooling first: py -3.12 -m venv .tools\iac; .tools\iac\Scripts\python.exe -m pip install -r infra\requirements-tools.lock.txt'
}
$env:BC_API_KEY = $null
$env:PRISMA_API_URL = $null
New-Item -ItemType Directory -Path (Join-Path $script:ProjectRoot '.azure') -Force | Out-Null
$reportPath = Join-Path $script:ProjectRoot '.azure\checkov.json'
$output = & $python $entryPoint -d (Join-Path $script:ProjectRoot 'infra') --framework bicep --skip-path 'training-access.bicep$' --skip-download --skip-results-upload --output json
$scanExit = $LASTEXITCODE
$output | Set-Content -LiteralPath $reportPath -Encoding UTF8
if ($scanExit -ne 0) { throw 'Checkov failed. Review the local .azure\checkov.json report; results were not uploaded.' }
$report = $output | Out-String | ConvertFrom-Json
if ($report.summary.parsing_errors -gt 0 -or $report.summary.passed -eq 0) { throw 'Checkov did not successfully analyze the infrastructure.' }
$approvedExceptions = @('CKV_AZURE_35', 'CKV_AZURE_43', 'CKV_AZURE_109', 'CKV_AZURE_189', 'CKV_AZURE_206', 'CKV_AZURE_243')
$actualExceptions = @($report.results.skipped_checks | ForEach-Object { $_.check_id })
if (@(Compare-Object -ReferenceObject $approvedExceptions -DifferenceObject $actualExceptions).Count -gt 0) {
    throw 'Security exceptions changed. Re-review docs\security-exceptions.md rather than silently accepting new suppressions.'
}
Write-Host "Checkov: $($report.summary.passed) passed, $($report.summary.failed) failed, $($report.summary.skipped) documented exceptions; exit 0."
# Checkov 3.3.19 cannot parse these existing-resource identity references; inspect compiled scopes.
& (Join-Path $PSScriptRoot 'Test-TrainingInfrastructure.ps1')
