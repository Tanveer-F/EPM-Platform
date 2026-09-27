targetScope = 'subscription'

@description('Azure region for this development foundation.')
param location string = 'eastus'

@description('Short lowercase project prefix used for deterministic resource names.')
@minLength(2)
@maxLength(8)
param projectName string = 'epm'

@description('Only the approved development environment is supported in Phase 1.')
@allowed(['dev'])
param environmentName string = 'dev'

@description('Explicit acknowledgement that service endpoints are public and authenticated, not private production endpoints.')
@allowed([true])
param acknowledgePublicDevelopmentEndpoints bool

var resourceGroupName = 'rg-${projectName}-${environmentName}-${location}'
var workspaceName = 'mlw-${projectName}-${environmentName}-${location}'
var tags = {
  project: projectName
  environment: environmentName
  phase: '1'
  managedBy: 'bicep'
}

resource resourceGroup 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module foundation 'modules/foundation.bicep' = {
  name: 'epm-foundation'
  scope: resourceGroup
  params: {
    location: location
    projectName: projectName
    environmentName: environmentName
    workspaceName: workspaceName
    publicDevelopmentAccess: acknowledgePublicDevelopmentEndpoints
    tags: tags
  }
}

output resourceGroupName string = resourceGroup.name
output workspaceName string = foundation.outputs.workspaceName
output computeName string = foundation.outputs.computeName
output storageAccountName string = foundation.outputs.storageAccountName
output keyVaultName string = foundation.outputs.keyVaultName
output applicationInsightsName string = foundation.outputs.applicationInsightsName
output logAnalyticsName string = foundation.outputs.logAnalyticsName
