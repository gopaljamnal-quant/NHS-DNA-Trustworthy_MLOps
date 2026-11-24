@description('Azure region restricted to UK South for NHS data residency')
param location string = 'uksouth'
param prefix string = 'nhsdna'
param uniqueSuffix string = uniqueString(resourceGroup().id)

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: '${prefix}st${uniqueSuffix}'
  location: location
  sku: { name: 'Standard_GRS' }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${prefix}-kv-${uniqueSuffix}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-appi-${uniqueSuffix}'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${prefix}acr${uniqueSuffix}'
  location: location
  sku: { name: 'Standard' }
  properties: { adminUserEnabled: false }
}

resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: '${prefix}-amlw-${uniqueSuffix}'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: 'NHS Outpatient DNA Trustworthy MLOps'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
    v1LegacyMode: false
  }
}

output workspaceName string = amlWorkspace.name
output acrLoginServer string = acr.properties.loginServer
