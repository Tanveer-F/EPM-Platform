"""Read-only checks; run with ``python -m epm_platform.verify``.

Only supported public SDK entity fields are required; missing fields fail closed.
Workspace provisioning state and compute OS are not exposed by azure-ai-ml 1.35.0.
They are reported as delegated_to_arm verification limits, not SDK assertions.
The parent's ARM verifier must check them; SDK success alone is not full validation.
No private SDK or REST fallback is used.
Datastore listing excludes secrets and checks metadata access, not stored data or
credential absence. Public SSH status is the SDK's interpretation of the ARM response.
"""

import json
import logging
from enum import Enum
from typing import Any

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from epm_platform.client import create_credential, create_ml_client
from epm_platform.config import AzureConfig, ConfigurationError, load_config

Check = dict[str, str]
_MISSING = object()
_VERIFICATION_LIMITS = (
    {
        "name": "workspace.provisioning_state",
        "status": "delegated_to_arm",
        "message": "ARM verification must confirm Succeeded; this SDK field is not exposed.",
    },
    {
        "name": "compute.os_type",
        "status": "delegated_to_arm",
        "message": "ARM verification must confirm Linux; this SDK field is not exposed.",
    },
)


def _field(entity: object, *path: str) -> Any:
    value = entity
    for name in path:
        value = getattr(value, name, _MISSING)
        if value is _MISSING or value is None:
            return _MISSING
    return value


def _text(value: object) -> str | None:
    if isinstance(value, Enum):
        value = value.value
    return value.casefold() if isinstance(value, str) else None


def _check(name: str, valid: bool | None, expectation: str) -> Check:
    result = {"name": name, "status": "passed" if valid is True else "failed"}
    if valid is not True:
        result["code"] = "unknown_or_missing" if valid is None else "policy_mismatch"
        result["message"] = expectation
    return result


def _equals(name: str, value: object, expected: str | int | bool, expectation: str) -> Check:
    if value is _MISSING:
        valid = None
    elif isinstance(expected, str):
        normalized = _text(value)
        valid = normalized == expected.casefold() if normalized is not None else None
    elif isinstance(expected, bool):
        valid = value is expected if isinstance(value, bool) else None
    else:
        # SDK durations are returned as float seconds, but booleans are not counts.
        valid = value == expected if type(value) in (int, float) else None
    return _check(name, valid, expectation)


def _identity_check(name: str, resource: object) -> Check:
    kind = _text(_field(resource, "identity", "type"))
    principal = _field(resource, "identity", "principal_id")
    valid = (
        kind in ("systemassigned", "system_assigned")
        and isinstance(principal, str)
        and bool(principal.strip())
    )
    if kind is None or principal is _MISSING:
        valid = None
    return _check(name, valid, "A system-assigned identity with a nonempty principal is required.")


def _failure(name: str, error: Exception) -> Check:
    code = "read_failed"
    message = "The read-only Azure operation failed. Check connectivity and configuration."
    if isinstance(error, ClientAuthenticationError):
        code, message = (
            "authentication_failed",
            "Authentication failed for the selected credential.",
        )
    elif isinstance(error, ResourceNotFoundError):
        code, message = "not_found", "The configured Azure resource was not found."
    elif isinstance(error, HttpResponseError):
        if error.status_code == 403:
            code, message = "access_denied", "The selected identity lacks access to this resource."
        elif error.status_code == 401:
            code, message = (
                "authentication_failed",
                "Authentication failed for the selected credential.",
            )
    elif isinstance(error, ServiceRequestError):
        code, message = "connection_failed", "The Azure service could not be reached."
    return {"name": name, "status": "failed", "code": code, "message": message}


def verify_foundation(client: Any, config: AzureConfig) -> list[Check]:
    """Read workspace, compute, and datastore metadata without any mutations."""
    checks: list[Check] = []
    workspace = compute = None
    try:
        workspace = client.workspaces.get(name=config.workspace_name)
        checks.append(
            _check("workspace.read", workspace is not None, "Workspace metadata is required.")
        )
    except Exception as error:
        checks.append(_failure("workspace.read", error))

    if workspace is not None:
        checks.extend(
            [
                _identity_check("workspace.identity", workspace),
                _equals(
                    "workspace.system_datastores_auth_mode",
                    _field(workspace, "system_datastores_auth_mode"),
                    "Identity",
                    "System datastores must use Identity authentication.",
                ),
                _equals(
                    "workspace.managed_network",
                    _field(workspace, "managed_network", "isolation_mode"),
                    "AllowInternetOutbound",
                    "The managed network must use AllowInternetOutbound isolation.",
                ),
            ]
        )

    try:
        compute = client.compute.get(name=config.compute_name)
        checks.append(_check("compute.read", compute is not None, "Compute metadata is required."))
    except Exception as error:
        checks.append(_failure("compute.read", error))

    if compute is not None:
        policies = (
            ("provisioning_state", "Succeeded", "Compute provisioning must be Succeeded."),
            ("type", "AmlCompute", "Compute must be an AmlCompute cluster."),
            ("size", "Standard_D2s_v3", "Compute size must be Standard_D2s_v3."),
            ("tier", "Dedicated", "Compute must use dedicated nodes."),
            ("min_instances", 0, "Compute minimum node count must be zero."),
            ("max_instances", 1, "Compute maximum node count must be one."),
            ("idle_time_before_scale_down", 300, "Idle scale-down must be 300 seconds."),
            ("enable_node_public_ip", False, "Compute node public IPs must be disabled."),
            ("ssh_public_access_enabled", False, "Compute public SSH must be disabled."),
        )
        checks.extend(
            _equals(f"compute.{name}", _field(compute, name), expected, expectation)
            for name, expected, expectation in policies
        )
        checks.append(_identity_check("compute.identity", compute))

    if workspace is not None and compute is not None:
        workspace_principal = _field(workspace, "identity", "principal_id")
        compute_principal = _field(compute, "identity", "principal_id")
        valid = None
        if all(
            isinstance(value, str) and value.strip()
            for value in (workspace_principal, compute_principal)
        ):
            valid = workspace_principal.strip().casefold() != compute_principal.strip().casefold()
        checks.append(
            _check(
                "identities.separate", valid, "Workspace and compute principals must be distinct."
            )
        )

    try:
        # Exhaust the iterator so pagination/authorization failures are not hidden.
        for _ in client.datastores.list(include_secrets=False):
            pass
        checks.append(_check("datastores.list", True, ""))
    except Exception as error:
        checks.append(_failure("datastores.list", error))
    return checks


def main() -> int:
    """Write one sanitized JSON document, returning 0 on success or 1 on failure."""
    checks: list[Check] = []
    credential = None
    previous_logging_threshold = logging.root.manager.disable
    # Azure exceptions/logging can contain tokens, endpoints and resource IDs.
    logging.disable(logging.CRITICAL)
    try:
        config = load_config()
        checks.append(_check("configuration", True, ""))
        credential = create_credential(config)
        client = create_ml_client(config, credential)
        checks.extend(verify_foundation(client, config))
    except ConfigurationError as error:
        checks.append(
            {
                "name": "configuration",
                "status": "failed",
                "code": "invalid_configuration",
                "message": str(error),
            }
        )
    except Exception as error:
        checks.append(_failure("verification", error))
    finally:
        if credential is not None:
            try:
                credential.close()
            except Exception:
                checks.append(
                    {
                        "name": "credential.close",
                        "status": "failed",
                        "code": "cleanup_failed",
                        "message": "Credential cleanup failed.",
                    }
                )
        logging.disable(previous_logging_threshold)

    status = "passed" if all(check["status"] == "passed" for check in checks) else "failed"
    print(
        json.dumps(
            {"status": status, "checks": checks, "verification_limits": _VERIFICATION_LIMITS},
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
