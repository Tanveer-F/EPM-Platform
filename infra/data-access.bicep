targetScope = 'resourceGroup'

@description('Existing storage account associated with the Phase 1 Azure ML workspace.')
param storageAccountName string

@description('Non-secret object identifier for the approved publishing user, resolved at runtime; never hardcoded.')
@minLength(36)
@maxLength(36)
param publisherObjectId string

resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' existing = {
  name: storageAccountName
}

resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2025-01-01' existing = {
  parent: storage
  name: 'default'
}

resource raw 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobs
  name: 'epm-cmapss-raw'
  properties: {
    publicAccess: 'None'
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    metadata: {
      project: 'epm'
      dataset: 'nasa-cmapss'
      zone: 'raw'
    }
  }
}

resource curated 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = {
  parent: blobs
  name: 'epm-cmapss-curated'
  properties: {
    publicAccess: 'None'
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    metadata: {
      project: 'epm'
      dataset: 'nasa-cmapss'
      zone: 'curated'
    }
  }
}

var blobContributorRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')

resource rawPublisher 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(raw.id, publisherObjectId, blobContributorRole)
  scope: raw
  properties: {
    roleDefinitionId: blobContributorRole
    principalId: publisherObjectId
    principalType: 'User'
    description: 'Approved Phase 2 publisher access limited to the C-MAPSS raw container.'
  }
}

resource curatedPublisher 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(curated.id, publisherObjectId, blobContributorRole)
  scope: curated
  properties: {
    roleDefinitionId: blobContributorRole
    principalId: publisherObjectId
    principalType: 'User'
    description: 'Approved Phase 2 publisher access limited to the C-MAPSS curated container.'
  }
}

output rawContainerName string = raw.name
output curatedContainerName string = curated.name
