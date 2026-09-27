@description('Resource location.')
param location string

@description('Project naming prefix.')
param projectName string

@description('Environment naming segment.')
param environmentName string

@description('Azure ML workspace name.')
param workspaceName string

@description('Approved development-only public service access; this module is not a private-network profile.')
@allowed([true])
param publicDevelopmentAccess bool

@description('Common ownership and cost-attribution tags.')
param tags object

var suffix = uniqueString(resourceGroup().id)
var publicNetworkAccess = publicDevelopmentAccess ? 'Enabled' : 'Disabled'

resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' = {
  // checkov:skip=CKV_AZURE_35:Approved authenticated public development access; production requires private networking (docs/architecture-decisions.md).
  // checkov:skip=CKV_AZURE_206:Standard_LRS is the approved single-region development cost tradeoff, not production disaster recovery.
  // checkov:skip=CKV_AZURE_43:Scanner cannot evaluate take/uniqueString; generated lowercase alphanumeric names are validated by ARM validation and WhatIf.
  name: take('st${projectName}${environmentName}${suffix}', 24)
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    allowCrossTenantReplication: false
    defaultToOAuthAuthentication: true
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    isHnsEnabled: false
    isSftpEnabled: false
    isNfsV3Enabled: false
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      bypass: 'None'
      defaultAction: publicDevelopmentAccess ? 'Allow' : 'Deny'
    }
    encryption: {
      keySource: 'Microsoft.Storage'
      services: {
        blob: { enabled: true, keyType: 'Account' }
        file: { enabled: true, keyType: 'Account' }
      }
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2025-01-01' = {
  parent: storage
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
      allowPermanentDelete: false
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  // checkov:skip=CKV_AZURE_109:Approved public development network; Entra RBAC still required. No corporate VPN is available.
  // checkov:skip=CKV_AZURE_189:Public authenticated development access explicitly approved; private endpoints are a documented production prerequisite.
  name: take('kv-${projectName}-${environmentName}-${suffix}', 24)
  location: location
  tags: tags
  properties: {
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    accessPolicies: []
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      bypass: 'None'
      defaultAction: publicDevelopmentAccess ? 'Allow' : 'Deny'
    }
  }
}

module workspaceTelemetry 'workspace-telemetry.bicep' = {
  name: 'epm-workspace-telemetry'
  params: {
    location: location
    projectName: projectName
    environmentName: environmentName
    tags: tags
  }
}

// Azure ML manages dependency grants for its system-assigned identity.
resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-06-01' = {
  // checkov:skip=CKV_AZURE_243:Public authenticated development workspace explicitly approved; private client networking is deferred, not claimed complete.
  name: workspaceName
  location: location
  tags: tags
  kind: 'Default'
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'EPM Platform development'
    description: 'Phase 1 predictive maintenance foundation; no data or training workloads.'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: workspaceTelemetry.outputs.applicationInsightsId
    systemDatastoresAuthMode: 'Identity'
    publicNetworkAccess: publicNetworkAccess
    v1LegacyMode: false
    managedNetwork: {
      isolationMode: 'AllowInternetOutbound'
      managedNetworkKind: 'V1'
    }
  }
}

module compute 'compute.bicep' = {
  name: 'epm-cpu'
  params: {
    location: location
    workspaceName: workspace.name
    tags: tags
  }
}

output workspaceName string = workspace.name
output computeName string = compute.outputs.computeName
output storageAccountName string = storage.name
output keyVaultName string = keyVault.name
output applicationInsightsName string = workspaceTelemetry.outputs.applicationInsightsName
output logAnalyticsName string = workspaceTelemetry.outputs.logAnalyticsName
