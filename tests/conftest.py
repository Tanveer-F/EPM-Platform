from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from epm_platform.config import AzureConfig

ENVIRONMENT_KEYS = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
    "AZURE_ML_COMPUTE",
    "EPM_AUTH_MODE",
    "AZURE_CLIENT_ID",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def subscription_id():
    return str(UUID(int=1))


@pytest.fixture
def config(subscription_id):
    return AzureConfig(subscription_id=subscription_id)


@pytest.fixture
def fake_credential():
    return Mock(spec_set=["get_token", "close"])


@pytest.fixture
def fake_client():
    workspace = SimpleNamespace(
        identity=SimpleNamespace(type="SystemAssigned", principal_id=str(UUID(int=2))),
        system_datastores_auth_mode="Identity",
        managed_network=SimpleNamespace(isolation_mode="AllowInternetOutbound"),
    )
    compute = SimpleNamespace(
        provisioning_state="Succeeded",
        identity=SimpleNamespace(type="SystemAssigned", principal_id=str(UUID(int=3))),
        type="AmlCompute",
        size="Standard_D2s_v3",
        tier="Dedicated",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=300.0,
        enable_node_public_ip=False,
        ssh_public_access_enabled=False,
    )
    client = Mock(spec_set=["workspaces", "compute", "datastores"])
    client.workspaces = Mock(spec_set=["get"])
    client.compute = Mock(spec_set=["get"])
    client.datastores = Mock(spec_set=["list"])
    client.workspaces.get.return_value = workspace
    client.compute.get.return_value = compute
    # Opaque objects ensure verification cannot assume or inspect datastore content.
    client.datastores.list.return_value = iter([object(), object()])
    return client
