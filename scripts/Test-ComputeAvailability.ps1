[CmdletBinding()]
param(
    [switch]$ApproveCosts,
    [ValidateRange(5, 30)][int]$TimeoutMinutes = 20
)

. (Join-Path $PSScriptRoot 'Initialize-Environment.ps1')
if (-not $ApproveCosts) { throw 'This optional test allocates a billable CPU node. Explicit -ApproveCosts is required.' }
$null = Get-EpmAccount
$computeId = "/subscriptions/$env:AZURE_SUBSCRIPTION_ID/resourceGroups/$env:AZURE_RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$env:AZURE_ML_WORKSPACE/computes/$env:AZURE_ML_COMPUTE"
function Get-CpuState {
    return (Invoke-AzureJson -Arguments @('rest', '--method', 'get', '--url', "https://management.azure.com${computeId}?api-version=2025-06-01")).properties
}
function Set-CpuMinimum {
    param([ValidateSet(0, 1)][int]$Minimum)
    $null = Invoke-AzureJson -Arguments @(
        'ml', 'compute', 'update', '--subscription', $env:AZURE_SUBSCRIPTION_ID,
        '--resource-group', $env:AZURE_RESOURCE_GROUP, '--workspace-name', $env:AZURE_ML_WORKSPACE,
        '--name', $env:AZURE_ML_COMPUTE, '--min-instances', "$Minimum", '--max-instances', '1',
        '--idle-time-before-scale-down', '300', '--no-wait'
    )
}
function Wait-CpuCount {
    param([ValidateSet(0, 1)][int]$Count)
    $deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
    $previous = ''
    do {
        $state = Get-CpuState
        $properties = $state.properties
        $summary = "state=$($state.provisioningState); allocated=$($properties.currentNodeCount); target=$($properties.targetNodeCount)"
        if ($summary -ne $previous) { Write-Host $summary; $previous = $summary }
        if ($state.provisioningState -eq 'Failed') { throw 'CPU resource provisioning failed.' }
        if ($state.provisioningState -eq 'Succeeded') {
            $counts = $properties.nodeStateCounts
            $idleReady = $null -ne $counts -and $counts.PSObject.Properties.Name -contains 'idleNodeCount' -and $counts.idleNodeCount -eq 1
            if ($properties.scaleSettings.minNodeCount -eq $Count -and $properties.currentNodeCount -eq $Count -and $properties.targetNodeCount -eq $Count) {
                if ($Count -eq 0 -or $idleReady) { return }
            }
            if ($Count -eq 1 -and $null -ne $counts -and $counts.PSObject.Properties.Name -contains 'unusableNodeCount' -and $counts.unusableNodeCount -gt 0) {
                throw 'Azure reported an unusable CPU node; no workload will be submitted.'
            }
        }
        Start-Sleep -Seconds 20
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for CPU node count $Count."
}
$initial = Get-CpuState
if ($initial.provisioningState -ne 'Succeeded' -or $initial.properties.vmSize -ne 'Standard_D2s_v3' -or
    $initial.properties.scaleSettings.minNodeCount -ne 0 -or $initial.properties.scaleSettings.maxNodeCount -ne 1 -or
    $initial.properties.currentNodeCount -ne 0 -or $initial.properties.targetNodeCount -ne 0) {
    throw 'Availability test requires the approved healthy, idle, zero-node baseline; no scaling was attempted.'
}
$allocated = $false
$restored = $false
try {
    Write-Host 'Allocating one approved CPU node, without submitting a job.'
    Set-CpuMinimum -Minimum 1
    Wait-CpuCount -Count 1
    $allocated = $true
    Write-Host 'PASS: one CPU node reached idle/available state.'
}
finally {
    try {
        Write-Host 'Restoring min=0, max=1 and five-minute idle scale-down.'
        Set-CpuMinimum -Minimum 0
        Wait-CpuCount -Count 0
        $restored = $true
        Write-Host 'PASS: zero allocated and target nodes confirmed.'
    }
    finally {
        $report = [pscustomobject]@{
            checkedAtUtc = [DateTime]::UtcNow.ToString('o')
            compute = $env:AZURE_ML_COMPUTE
            oneIdleNodeVerified = $allocated
            zeroNodesRestored = $restored
            workloadsSubmitted = $false
        }
        New-Item -ItemType Directory -Path (Join-Path $script:ProjectRoot '.azure') -Force | Out-Null
        $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $script:ProjectRoot '.azure\compute-availability.json') -Encoding UTF8
        if (-not $restored) { Write-Warning 'URGENT: zero-node restoration is unverified. Inspect this exact cluster in Azure immediately; charges may continue.' }
    }
}
