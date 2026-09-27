[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'Common.ps1')
$path=Join-Path $script:ProjectRoot '.azure\training-access.json'
& az bicep build --file (Join-Path $script:ProjectRoot 'infra\training-access.bicep') --outfile $path --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Training Bicep compilation failed.' }
$t=Get-Content $path -Raw | ConvertFrom-Json
$r=@($t.resources)
if ($r.Count -ne 5 -or @($r | Where-Object { $_.type -ne 'Microsoft.Authorization/roleAssignments' }).Count) { throw 'Only five scoped role assignments are allowed; no registry or compute creation.' }
foreach ($assignment in $r) {
  if ($assignment.scope -notmatch 'Microsoft.Storage/storageAccounts/blobServices/containers') { throw 'Role assignment exceeds container scope.' }
  if ($assignment.properties.principalType -notin @('User','ServicePrincipal')) { throw 'Unexpected principal type.' }
  if ($assignment.properties.roleDefinitionId -notmatch 'blobRead|blobWrite') { throw 'Unexpected runtime role.' }
}
$inputs=@($r | Where-Object { $_.scope -match 'epm-cmapss-curated' })
if ($inputs.Count -ne 1 -or $inputs[0].properties.roleDefinitionId -notmatch 'blobRead') { throw 'Training inputs must be read-only.' }
if ($t.variables.blobRead -notmatch '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1' -or $t.variables.blobWrite -notmatch 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') { throw 'Incorrect built-in role IDs.' }
Write-Host 'PASS: compiled training template contains five container-scoped grants only; curated access is read-only; no ACR, service, or compute resource.'
