[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('Publish','Submit','Status','Download')][string]$Action,
  [string]$DataPath,
  [string]$AssetVersion,
  [string]$ManifestSha256,
  [string]$EnvironmentVersion='54',
  [string]$JobName,
  [string]$Destination,
  [switch]$ApproveCosts
)
. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$arguments=@('-m','epm_platform.baseline.azure',$Action.ToLowerInvariant())
switch ($Action) {
  'Publish' { if (-not $DataPath) { throw 'DataPath is required.' }; $arguments+=@('--data',(Resolve-Path $DataPath).Path) }
  'Submit' { if (-not $ApproveCosts) { throw 'ApproveCosts is required.' }; $arguments+=@('--asset-version',$AssetVersion,'--manifest-sha256',$ManifestSha256,'--environment-version',$EnvironmentVersion,'--approve-costs') }
  'Status' { $arguments+=@('--job-name',$JobName) }
  'Download' { $arguments+=@('--job-name',$JobName,'--destination',$Destination) }
}
& (Join-Path $script:ProjectRoot '.venv\Scripts\python.exe') @arguments
if ($LASTEXITCODE -ne 0) { throw "Baseline $Action failed." }
