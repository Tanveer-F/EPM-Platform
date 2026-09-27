from dataclasses import replace
from unittest.mock import Mock
from uuid import UUID

import pytest

from epm_platform import client


def test_cli_credential_is_explicit(monkeypatch, config, fake_credential):
    cli = Mock(return_value=fake_credential)
    managed = Mock()
    monkeypatch.setattr(client, "AzureCliCredential", cli)
    monkeypatch.setattr(client, "ManagedIdentityCredential", managed)
    selected = replace(config, managed_identity_client_id=str(UUID(int=4)))
    assert client.create_credential(selected) is fake_credential
    cli.assert_called_once_with(process_timeout=30)
    managed.assert_not_called()
    fake_credential.get_token.assert_not_called()


@pytest.mark.parametrize("use_client_id", [False, True])
def test_managed_identity_credential(monkeypatch, config, fake_credential, use_client_id):
    managed = Mock(return_value=fake_credential)
    cli = Mock()
    monkeypatch.setattr(client, "ManagedIdentityCredential", managed)
    monkeypatch.setattr(client, "AzureCliCredential", cli)
    client_id = str(UUID(int=4)) if use_client_id else None
    selected = replace(config, auth_mode="managed-identity", managed_identity_client_id=client_id)
    assert client.create_credential(selected) is fake_credential
    if use_client_id:
        managed.assert_called_once_with(client_id=client_id)
    else:
        managed.assert_called_once_with()
    cli.assert_not_called()
    fake_credential.get_token.assert_not_called()


def test_client_is_scoped_and_telemetry_disabled(monkeypatch, config, fake_credential):
    factory = Mock()
    monkeypatch.setattr(client, "MLClient", factory)
    assert client.create_ml_client(config, fake_credential) is factory.return_value
    factory.assert_called_once_with(
        credential=fake_credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
        enable_telemetry=False,
        show_progress=False,
        logging_enable=False,
    )
    fake_credential.get_token.assert_not_called()
