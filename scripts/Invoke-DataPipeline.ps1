[CmdletBinding()]
param(
    [ValidateSet('Acquire', 'Curate', 'Publish', 'Verify')][string]$Action = 'Curate',
    [string]$Version,
    [string]$ArchivePath,
    [switch]$ApproveAzureWrites
)

. (Join-Path $PSScriptRoot 'Common.ps1')
if ($Action -in @('Publish', 'Verify')) {
    . (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
    if (-not $Version -or $Version -notmatch '^sha256-[a-f0-9]{64}$') {
        throw 'An explicit verified curated version is required. Use the version printed by Curate.'
    }
}
if ($Action -eq 'Publish' -and -not $ApproveAzureWrites) {
    throw 'Publishing uploads data and registers a version; explicit -ApproveAzureWrites is required.'
}
if ($ArchivePath -and $Action -ne 'Acquire') { throw 'ArchivePath is only supported by Acquire.' }
$python = Join-Path $script:ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the locked project environment first.' }
$arguments = @(
    '-m', 'epm_platform.data',
    '--spec', (Join-Path $script:ProjectRoot 'config\cmapss-source.json'),
    '--data-root', (Join-Path $script:ProjectRoot 'data'),
    '--state-dir', (Join-Path $script:ProjectRoot '.azure'),
    $Action.ToLowerInvariant()
)
if ($ArchivePath) { $arguments += @('--archive', (Resolve-Path -LiteralPath $ArchivePath).Path) }
if ($Action -in @('Publish', 'Verify')) { $arguments += @('--version', $Version) }
if ($ApproveAzureWrites -and $Action -eq 'Publish') { $arguments += '--approve-azure-writes' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Phase 2 operation failed: $Action. No fallback to keys or another dataset is permitted." }
