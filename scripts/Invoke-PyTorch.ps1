[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Submit','Status','Download')][string]$Action,
    [string]$JobName,
    [string]$Destination,
    [switch]$ApproveCosts
)
. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$arguments=@('-m','epm_platform.deep_learning.azure',$Action.ToLowerInvariant())
switch ($Action) {
    'Submit' { if (-not $ApproveCosts) { throw 'Explicit CPU job and tracking cost approval required.' }; $arguments+='--approve-costs' }
    'Status' { if (-not $JobName) { throw 'JobName required.' }; $arguments+=@('--job-name',$JobName) }
    'Download' { if (-not $JobName -or -not $Destination) { throw 'JobName and Destination required.' }; $arguments+=@('--job-name',$JobName,'--destination',$Destination) }
}
& (Join-Path $script:ProjectRoot '.venv\Scripts\python.exe') @arguments
if ($LASTEXITCODE -ne 0) { throw "PyTorch $Action failed; inspect the saved run before any retry." }
