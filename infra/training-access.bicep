targetScope = 'resourceGroup'

@description('Existing workspace name.')
param workspaceName string
@description('Existing CPU cluster.')
param computeName string = 'cpu-dev'
@description('Existing workspace storage name.')
param storageAccountName string
@description('Runtime-resolved authorized publishing user.')
param publisherObjectId string
@description('Existing workspace job blob container.')
param jobContainerName string

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-06-01' existing = {
  name: workspaceName
}
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2025-06-01' existing = {
  parent: workspace
  name: computeName
}
resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' existing = {
  name: storageAccountName
}
resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2025-01-01' existing = {
  parent: storage
  name: 'default'
}
resource inputs 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' existing = {
  parent: blobs
  name: 'epm-cmapss-curated'
}
resource jobBlobs 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' existing = {
  parent: blobs
  name: jobContainerName
}
resource artifacts 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' existing = {
  parent: blobs
  name: 'azureml'
}
var blobRead = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
var blobWrite = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
)
resource computeInputs 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(inputs.id, compute.id, blobRead)
  scope: inputs
  properties: {
    principalId: compute.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobRead
  }
}
resource computeJob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(jobBlobs.id, compute.id, blobWrite)
  scope: jobBlobs
  properties: {
    principalId: compute.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobWrite
  }
}
resource publisherJob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(jobBlobs.id, publisherObjectId, blobWrite)
  scope: jobBlobs
  properties: {
    principalId: publisherObjectId
    principalType: 'User'
    roleDefinitionId: blobWrite
  }
}
resource computeArtifacts 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(artifacts.id, compute.id, blobWrite)
  scope: artifacts
  properties: {
    principalId: compute.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobWrite
  }
}
resource publisherArtifacts 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(artifacts.id, publisherObjectId, blobWrite)
  scope: artifacts
  properties: {
    principalId: publisherObjectId
    principalType: 'User'
    roleDefinitionId: blobWrite
  }
}
