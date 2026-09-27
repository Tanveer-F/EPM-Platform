@description('Same region as the Azure ML workspace.')
param location string

@description('Project naming prefix.')
param projectName string

@description('Development environment naming segment.')
param environmentName string

@description('Common ownership and cost-attribution tags.')
param tags object

resource logs 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: 'log-${projectName}-${environmentName}-${location}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    workspaceCapping: {
      dailyQuotaGb: 1
    }
    features: {
      disableLocalAuth: true
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${projectName}-${environmentName}-${location}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
    IngestionMode: 'LogAnalytics'
    DisableLocalAuth: true
    DisableIpMasking: false
    RetentionInDays: 30
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Azure creates this default asynchronously; explicitly prevent notification recipients in Phase 1.
resource defaultSmartDetection 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'Application Insights Smart Detection'
  location: 'global'
  tags: tags
  properties: {
    groupShortName: 'SmartDetect'
    enabled: false
    armRoleReceivers: []
    emailReceivers: []
    smsReceivers: []
    webhookReceivers: []
  }
  dependsOn: [appInsights]
}

output applicationInsightsId string = appInsights.id
output applicationInsightsName string = appInsights.name
output logAnalyticsName string = logs.name
