from uuid import UUID

import pytest

from epm_platform.config import AzureConfig, ConfigurationError, load_config


def test_defaults_from_environment(monkeypatch, subscription_id):
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", subscription_id)
    config = load_config()
    assert config.subscription_id == subscription_id
    assert config.resource_group == "rg-epm-dev-eastus"
    assert config.workspace_name == "mlw-epm-dev-eastus"
    assert config.compute_name == "cpu-dev"
    assert config.auth_mode == "azure-cli"
    assert config.managed_identity_client_id is None
    assert subscription_id not in repr(config)


def test_environment_overrides(monkeypatch, subscription_id):
    client_id = str(UUID(int=4))
    for key, value in {
        "AZURE_SUBSCRIPTION_ID": subscription_id,
        "AZURE_RESOURCE_GROUP": "rg-local",
        "AZURE_ML_WORKSPACE": "mlw-local",
        "AZURE_ML_COMPUTE": "cpu-local",
        "EPM_AUTH_MODE": "managed-identity",
        "AZURE_CLIENT_ID": client_id,
    }.items():
        monkeypatch.setenv(key, value)
    config = load_config()
    assert config.resource_group == "rg-local"
    assert config.workspace_name == "mlw-local"
    assert config.compute_name == "cpu-local"
    assert config.auth_mode == "managed-identity"
    assert config.managed_identity_client_id == client_id
    assert client_id not in repr(config)


def test_subscription_is_required():
    with pytest.raises(ConfigurationError, match="AZURE_SUBSCRIPTION_ID"):
        load_config()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AZURE_SUBSCRIPTION_ID", ""),
        ("AZURE_SUBSCRIPTION_ID", "not-a-uuid"),
        ("AZURE_SUBSCRIPTION_ID", "0" * 32),
        ("AZURE_RESOURCE_GROUP", ""),
        ("AZURE_RESOURCE_GROUP", "rg."),
        ("AZURE_RESOURCE_GROUP", "rg with spaces"),
        ("AZURE_RESOURCE_GROUP", "r" * 91),
        ("AZURE_ML_WORKSPACE", ""),
        ("AZURE_ML_WORKSPACE", "ab"),
        ("AZURE_ML_WORKSPACE", "w" * 34),
        ("AZURE_ML_WORKSPACE", "-workspace"),
        ("AZURE_ML_COMPUTE", ""),
        ("AZURE_ML_COMPUTE", "c"),
        ("AZURE_ML_COMPUTE", "c" * 17),
        ("AZURE_ML_COMPUTE", "cpu_1"),
        ("AZURE_ML_COMPUTE", "1cpu"),
        ("AZURE_ML_COMPUTE", "cpu-"),
        ("EPM_AUTH_MODE", ""),
        ("EPM_AUTH_MODE", "default"),
        ("EPM_AUTH_MODE", "Azure-CLI"),
        ("AZURE_CLIENT_ID", ""),
        ("AZURE_CLIENT_ID", "bad-client-id"),
    ],
)
def test_invalid_environment(monkeypatch, subscription_id, key, value):
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", subscription_id)
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigurationError, match=key):
        load_config()


@pytest.mark.parametrize("key", ["AZURE_SUBSCRIPTION_ID", "AZURE_ML_WORKSPACE", "EPM_AUTH_MODE"])
def test_errors_do_not_include_environment_values(monkeypatch, subscription_id, key):
    secret = 'private-value"\n\x1b[31m'
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", subscription_id)
    monkeypatch.setenv(key, secret)
    with pytest.raises(ConfigurationError) as error:
        load_config()
    assert secret not in str(error.value)
    assert "private-value" not in str(error.value)
    assert subscription_id not in str(error.value)


def test_direct_configuration_also_validates(subscription_id):
    with pytest.raises(ConfigurationError, match="EPM_AUTH_MODE"):
        AzureConfig(subscription_id=subscription_id, auth_mode="default")


def test_uppercase_uuid_allowed(monkeypatch):
    value = str(UUID(int=0xABCDEF)).upper()
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", value)
    assert load_config().subscription_id == value
