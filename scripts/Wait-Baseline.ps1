[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$JobName,[ValidateRange(1,90)][int]$TimeoutMinutes=80)
. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
$deadline=[DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
$previous=''
do {
    $job=Invoke-AzureJson -Arguments @('ml','job','show','--subscription',$env:AZURE_SUBSCRIPTION_ID,'--resource-group',$env:AZURE_RESOURCE_GROUP,'--workspace-name',$env:AZURE_ML_WORKSPACE,'--name',$JobName,'--query','{name:name,status:status}')
    if ($job.status -ne $previous) { Write-Host ((Get-Date -Format o)+' '+$job.status); $previous=$job.status }
    if ($job.status -in @('Completed','Failed','Canceled','NotResponding')) { break }
    Start-Sleep -Seconds 30
} while ([DateTime]::UtcNow -lt $deadline)
if ($job.status -ne 'Completed') { throw "Job state: $($job.status). No replacement job was submitted; inspect this run before retrying." }
Write-Host 'PASS: Azure ML command job completed.'
