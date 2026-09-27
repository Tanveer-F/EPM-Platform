"""Explicit Microsoft Entra credentials and Azure ML SDK v2 client creation."""

from azure.ai.ml import MLClient
from azure.core.credentials import TokenCredential
from azure.identity import AzureCliCredential, ManagedIdentityCredential

from epm_platform.config import AzureConfig


def create_credential(config: AzureConfig) -> TokenCredential:
    """Use exactly the chosen credential, never a fallback credential chain."""
    if config.auth_mode == "azure-cli":
        return AzureCliCredential(process_timeout=30)
    if config.auth_mode == "managed-identity":
        if config.managed_identity_client_id is not None:
            return ManagedIdentityCredential(client_id=config.managed_identity_client_id)
        return ManagedIdentityCredential()
    raise ValueError("Unsupported authentication mode.")


def create_ml_client(config: AzureConfig, credential: TokenCredential) -> MLClient:
    """Create an explicitly scoped client without performing any Azure operations."""
    return MLClient(
        credential=credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
        enable_telemetry=False,
        show_progress=False,
        logging_enable=False,
    )
