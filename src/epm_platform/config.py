"""Validated configuration from the process environment (no dotenv loading)."""

import os
import re
from dataclasses import dataclass, field

_UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_RESOURCE_GROUP = re.compile(r"[A-Za-z0-9_().-]{1,90}")
_WORKSPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,31}[A-Za-z0-9]")
_COMPUTE = re.compile(r"[A-Za-z][A-Za-z0-9-]{0,14}[A-Za-z0-9]")


class ConfigurationError(ValueError):
    """A configuration failure whose message never includes environment values."""


@dataclass(frozen=True, slots=True)
class AzureConfig:
    subscription_id: str = field(repr=False)
    resource_group: str = "rg-epm-dev-eastus"
    workspace_name: str = "mlw-epm-dev-eastus"
    compute_name: str = "cpu-dev"
    auth_mode: str = "azure-cli"
    managed_identity_client_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate("AZURE_SUBSCRIPTION_ID", self.subscription_id, _UUID)
        _validate("AZURE_RESOURCE_GROUP", self.resource_group, _RESOURCE_GROUP)
        if self.resource_group.endswith("."):
            raise ConfigurationError("AZURE_RESOURCE_GROUP has an invalid format.")
        _validate("AZURE_ML_WORKSPACE", self.workspace_name, _WORKSPACE)
        _validate("AZURE_ML_COMPUTE", self.compute_name, _COMPUTE)
        if self.auth_mode not in ("azure-cli", "managed-identity"):
            raise ConfigurationError("EPM_AUTH_MODE must be azure-cli or managed-identity.")
        if self.managed_identity_client_id is not None:
            _validate("AZURE_CLIENT_ID", self.managed_identity_client_id, _UUID)


def _validate(name: str, value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ConfigurationError(f"{name} is missing or has an invalid format.")


def load_config() -> AzureConfig:
    """Read environment variables only; blank values are errors, not defaults."""
    return AzureConfig(
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP", "rg-epm-dev-eastus"),
        workspace_name=os.environ.get("AZURE_ML_WORKSPACE", "mlw-epm-dev-eastus"),
        compute_name=os.environ.get("AZURE_ML_COMPUTE", "cpu-dev"),
        auth_mode=os.environ.get("EPM_AUTH_MODE", "azure-cli"),
        managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID"),
    )
