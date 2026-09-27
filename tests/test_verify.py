import json
import logging
import subprocess
import sys
from enum import Enum
from unittest.mock import Mock, call

import pytest
from azure.ai.ml.entities import AmlCompute, IdentityConfiguration, ManagedNetwork, Workspace
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from epm_platform import verify


def checks_by_name(client, config):
    return {check["name"]: check for check in verify.verify_foundation(client, config)}


def test_success_uses_only_three_metadata_operations(fake_client, config):
    checks = verify.verify_foundation(fake_client, config)
    assert len(checks) == 17
    assert all(check["status"] == "passed" for check in checks)
    assert fake_client.mock_calls == [
        call.workspaces.get(name=config.workspace_name),
        call.compute.get(name=config.compute_name),
        call.datastores.list(include_secrets=False),
    ]


def test_sdk_enum_and_lowercase_values(fake_client, config):
    workspace = fake_client.workspaces.get.return_value
    compute = fake_client.compute.get.return_value
    for resource, field, value in [
        (workspace, "system_datastores_auth_mode", "Identity"),
        (workspace.managed_network, "isolation_mode", "AllowInternetOutbound"),
        (workspace.identity, "type", "system_assigned"),
        (compute.identity, "type", "system_assigned"),
        (compute, "provisioning_state", "Succeeded"),
        (compute, "type", "amlcompute"),
        (compute, "tier", "dedicated"),
    ]:
        setattr(resource, field, Enum("SdkValue", {"VALUE": value}).VALUE)
    assert all(
        check["status"] == "passed" for check in verify.verify_foundation(fake_client, config)
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("provisioning_state", "Failed"),
        ("provisioning_state", "Creating"),
        ("type", "ComputeInstance"),
        ("size", "Standard_D4s_v3"),
        ("tier", "LowPriority"),
        ("min_instances", 1),
        ("max_instances", 2),
        ("idle_time_before_scale_down", 120),
        ("enable_node_public_ip", True),
        ("ssh_public_access_enabled", True),
    ],
)
def test_invalid_cluster_policy(fake_client, config, field, invalid):
    setattr(fake_client.compute.get.return_value, field, invalid)
    check = checks_by_name(fake_client, config)[f"compute.{field}"]
    assert check["status"] == "failed"
    assert check["code"] == "policy_mismatch"


@pytest.mark.parametrize(
    "field",
    [
        "provisioning_state",
        "type",
        "size",
        "tier",
        "min_instances",
        "max_instances",
        "idle_time_before_scale_down",
        "enable_node_public_ip",
        "ssh_public_access_enabled",
        "identity",
    ],
)
def test_missing_cluster_fields_fail_closed(fake_client, config, field):
    delattr(fake_client.compute.get.return_value, field)
    check = checks_by_name(fake_client, config)[f"compute.{field}"]
    assert check["status"] == "failed"
    assert check["code"] == "unknown_or_missing"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("min_instances", False),
        ("max_instances", True),
        ("ssh_public_access_enabled", 0),
        ("enable_node_public_ip", "false"),
        ("idle_time_before_scale_down", None),
    ],
)
def test_malformed_cluster_values_are_not_coerced(fake_client, config, field, invalid):
    setattr(fake_client.compute.get.return_value, field, invalid)
    assert checks_by_name(fake_client, config)[f"compute.{field}"]["status"] == "failed"


@pytest.mark.parametrize("field", ["system_datastores_auth_mode", "managed_network", "identity"])
def test_missing_supported_workspace_fields_fail_closed(fake_client, config, field):
    delattr(fake_client.workspaces.get.return_value, field)
    check = checks_by_name(fake_client, config)[f"workspace.{field}"]
    assert check["status"] == "failed"
    assert check["code"] == "unknown_or_missing"


@pytest.mark.parametrize(
    ("field", "invalid", "check_name"),
    [
        ("system_datastores_auth_mode", "AccessKey", "workspace.system_datastores_auth_mode"),
        ("managed_network", None, "workspace.managed_network"),
        ("identity", None, "workspace.identity"),
    ],
)
def test_workspace_policy_failure(fake_client, config, field, invalid, check_name):
    setattr(fake_client.workspaces.get.return_value, field, invalid)
    assert checks_by_name(fake_client, config)[check_name]["status"] == "failed"


def test_wrong_network_isolation(fake_client, config):
    fake_client.workspaces.get.return_value.managed_network.isolation_mode = "Disabled"
    assert checks_by_name(fake_client, config)["workspace.managed_network"]["status"] == "failed"


@pytest.mark.parametrize("scope", ["workspaces", "compute"])
@pytest.mark.parametrize(("field", "invalid"), [("type", "UserAssigned"), ("principal_id", " ")])
def test_system_identity_required(fake_client, config, scope, field, invalid):
    identity = getattr(fake_client, scope).get.return_value.identity
    setattr(identity, field, invalid)
    name = "workspace" if scope == "workspaces" else "compute"
    assert checks_by_name(fake_client, config)[f"{name}.identity"]["status"] == "failed"


def test_principals_must_be_distinct(fake_client, config):
    principal = fake_client.workspaces.get.return_value.identity.principal_id
    fake_client.compute.get.return_value.identity.principal_id = principal.upper()
    assert checks_by_name(fake_client, config)["identities.separate"]["status"] == "failed"


def test_cli_supported_sdk_entities_pass_without_arm_only_fields(cli_setup, config, capsys):
    client, _ = cli_setup
    workspace_identity = client.workspaces.get.return_value.identity
    compute_identity = client.compute.get.return_value.identity
    workspace = Workspace(
        name=config.workspace_name,
        identity=IdentityConfiguration(
            type="SystemAssigned", principal_id=workspace_identity.principal_id
        ),
        system_datastores_auth_mode="Identity",
        managed_network=ManagedNetwork(isolation_mode="AllowInternetOutbound"),
    )
    compute = AmlCompute(
        name=config.compute_name,
        provisioning_state="Succeeded",
        identity=IdentityConfiguration(
            type="SystemAssigned", principal_id=compute_identity.principal_id
        ),
        size="Standard_D2s_v3",
        tier="dedicated",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=300.0,
        enable_node_public_ip=False,
        ssh_public_access_enabled=False,
    )
    assert not hasattr(workspace, "provisioning_state")
    assert not hasattr(compute, "os_type")
    client.workspaces.get.return_value = workspace
    client.compute.get.return_value = compute

    assert verify.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert all(check["status"] == "passed" for check in result["checks"])
    assert {limit["name"]: limit["status"] for limit in result["verification_limits"]} == {
        "workspace.provisioning_state": "delegated_to_arm",
        "compute.os_type": "delegated_to_arm",
    }
    assert not {"workspace.provisioning_state", "compute.os_type"} & {
        check["name"] for check in result["checks"]
    }


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ClientAuthenticationError("sensitive diagnostic"), "authentication_failed"),
        (ResourceNotFoundError("sensitive diagnostic"), "not_found"),
        (ServiceRequestError("sensitive diagnostic"), "connection_failed"),
        (RuntimeError("sensitive diagnostic"), "read_failed"),
    ],
)
def test_sdk_errors_are_sanitized(fake_client, config, error, code):
    fake_client.workspaces.get.side_effect = error
    checks = checks_by_name(fake_client, config)
    assert checks["workspace.read"]["code"] == code
    assert "sensitive diagnostic" not in json.dumps(checks)
    fake_client.compute.get.assert_called_once()
    fake_client.datastores.list.assert_called_once()


@pytest.mark.parametrize(
    ("status", "code"), [(401, "authentication_failed"), (403, "access_denied")]
)
def test_http_status_diagnostics(fake_client, config, status, code):
    error = HttpResponseError("sensitive diagnostic")
    error.status_code = status
    fake_client.compute.get.side_effect = error
    assert checks_by_name(fake_client, config)["compute.read"]["code"] == code


def test_paged_datastore_failure_is_detected(fake_client, config):
    def pages():
        yield object()
        raise ClientAuthenticationError("private paged response")

    fake_client.datastores.list.return_value = pages()
    check = checks_by_name(fake_client, config)["datastores.list"]
    assert check["status"] == "failed"
    assert check["code"] == "authentication_failed"


@pytest.fixture
def cli_setup(monkeypatch, subscription_id, fake_client, fake_credential):
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", subscription_id)
    monkeypatch.setattr(verify, "create_credential", Mock(return_value=fake_credential))
    monkeypatch.setattr(verify, "create_ml_client", Mock(return_value=fake_client))
    return fake_client, fake_credential


def test_module_entrypoint_without_subscription_is_offline():
    result = subprocess.run(
        [sys.executable, "-m", "epm_platform.verify"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["checks"][0]["code"] == "invalid_configuration"


def test_cli_json_success(cli_setup, capsys):
    assert verify.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result["status"] == "passed"
    assert all(check["status"] == "passed" for check in result["checks"])
    cli_setup[1].close.assert_called_once()


def test_cli_failure_is_nonzero(cli_setup, capsys):
    cli_setup[0].compute.get.return_value.max_instances = 4
    assert verify.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize(
    ("scope", "field"),
    [("workspaces", "system_datastores_auth_mode"), ("compute", "provisioning_state")],
)
def test_cli_missing_supported_field_is_nonzero(cli_setup, capsys, scope, field):
    delattr(getattr(cli_setup[0], scope).get.return_value, field)
    assert verify.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert any(check.get("code") == "unknown_or_missing" for check in result["checks"])


def test_cli_missing_config_does_not_construct_credential(monkeypatch, capsys):
    factory = Mock()
    monkeypatch.setattr(verify, "create_credential", factory)
    assert verify.main() == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["code"] == "invalid_configuration"
    factory.assert_not_called()


def test_cli_untrusted_values_cannot_escape_json(cli_setup, capsys, subscription_id):
    sensitive = 'private-value"\n\x1b[31m' + subscription_id

    def fail(**kwargs):
        logging.getLogger("azure.identity").critical(sensitive)
        raise ClientAuthenticationError(sensitive)

    cli_setup[0].workspaces.get.side_effect = fail
    cli_setup[0].compute.get.return_value.size = sensitive
    previous_threshold = logging.root.manager.disable
    assert verify.main() == 1
    assert logging.root.manager.disable == previous_threshold
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "failed"
    assert "private-value" not in captured.out
    assert subscription_id not in captured.out
    assert "\x1b" not in captured.out


def test_cli_configuration_injection_is_sanitized(cli_setup, monkeypatch, capsys):
    monkeypatch.setenv("EPM_AUTH_MODE", 'private-value"\n\x1b[31m')
    assert verify.main() == 1
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert "private-value" not in captured.out
    assert json.loads(captured.out)["checks"][0]["code"] == "invalid_configuration"


def test_cli_credential_failure_is_sanitized(cli_setup, monkeypatch, capsys):
    monkeypatch.setattr(
        verify, "create_credential", Mock(side_effect=RuntimeError("private-value"))
    )
    assert verify.main() == 1
    assert "private-value" not in capsys.readouterr().out


def test_cli_closes_credential_after_client_failure(cli_setup, monkeypatch, capsys):
    monkeypatch.setattr(verify, "create_ml_client", Mock(side_effect=RuntimeError("private-value")))
    assert verify.main() == 1
    assert "private-value" not in capsys.readouterr().out
    cli_setup[1].close.assert_called_once()


def test_cli_cleanup_error_is_sanitized(cli_setup, capsys):
    cli_setup[1].close.side_effect = RuntimeError("private-value")
    assert verify.main() == 1
    captured = capsys.readouterr()
    assert "private-value" not in captured.out
    assert json.loads(captured.out)["checks"][-1]["code"] == "cleanup_failed"
