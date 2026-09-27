@description('Compute location, matching the workspace.')
param location string

@description('Existing parent Azure ML workspace name.')
param workspaceName string

@description('Common ownership and cost-attribution tags.')
param tags object

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-06-01' existing = {
  name: workspaceName
}

resource cpu 'Microsoft.MachineLearningServices/workspaces/computes@2025-06-01' = {
  parent: workspace
  name: 'cpu-dev'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    disableLocalAuth: true
    properties: {
      vmSize: 'Standard_D2s_v3'
      vmPriority: 'Dedicated'
      osType: 'Linux'
      enableNodePublicIp: false
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT5M'
      }
    }
  }
}

output computeName string = cpu.name
