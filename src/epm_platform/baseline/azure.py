"""Explicit, bounded Azure ML v2 publication and CPU baseline job operations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import shutil
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from uuid import uuid4

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import (
    AzureBlobDatastore,
    CommandJobLimits,
    Data,
    ManagedIdentityConfiguration,
    NoneCredentialConfiguration,
)
from azure.core import MatchConditions
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from epm_platform.client import create_credential, create_ml_client
from epm_platform.config import AzureConfig, load_config
from epm_platform.data import publishing
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint
from epm_platform.features.pipeline import FEATURE_COLUMNS, verify_features

ASSET_NAME = "epm-cmapss-ml-ready"
ENVIRONMENT_NAME = "sklearn-1.5"
ENVIRONMENT_VERSION = "54"
ENVIRONMENT_REF = "azureml://registries/azureml/environments/sklearn-1.5/versions/54"
SOURCE = {
    "asset_name": "epm-cmapss-curated",
    "asset_version": "d-xjsfezqpozevgso26sct6tpodm",
    "manifest_sha256": "ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640",
}
_CONTENT = frozenset(
    {"train.parquet", "validation.parquet", "test.parquet", "splits.json", "feature-summary.json"}
)
_SHA256 = re.compile(r"[a-f0-9]{64}")
_JOB_NAME = re.compile(r"epm-baseline-[a-f0-9]{12}")
_CODE_PATHS = {
    name: name
    for name in (
        "src/epm_platform/__init__.py",
        "src/epm_platform/features/__init__.py",
        "src/epm_platform/features/pipeline.py",
        "src/epm_platform/baseline/__init__.py",
        "src/epm_platform/baseline/training.py",
        "src/epm_platform/data/__init__.py",
        "src/epm_platform/data/curation.py",
        "src/epm_platform/data/errors.py",
        "src/epm_platform/data/manifest.py",
        "src/epm_platform/data/source.py",
        "src/epm_platform/data/spec.py",
        "src/epm_platform/data/validation.py",
        "config/baseline.json",
        "config/features.json",
        "config/cmapss-source.json",
    )
} | {
    "config/runtime-requirements.txt": "environments/baseline/requirements.txt",
    "config/run-baseline.sh": "scripts/run-baseline.sh",
}
_MAX_CODE_FILE = 1024 * 1024
_MAX_CODE_BUNDLE = 4 * 1024 * 1024
_STATUSES = frozenset(
    {
        "NotStarted",
        "Starting",
        "Provisioning",
        "Preparing",
        "Queued",
        "Running",
        "Finalizing",
        "Completed",
        "Failed",
        "Canceled",
        "CancelRequested",
        "Paused",
    }
)


@contextmanager
def _operation(name: str):
    try:
        yield
    except DataError:
        raise
    except Exception as error:
        status = getattr(error, "status_code", None)
        code = f"HTTP {status}" if type(status) is int and 100 <= status <= 599 else "no status"
        raise DataError(
            f"{name} failed ({code}). Check the configured identity, access and resource state; "
            "no automatic resubmission or credential fallback is performed."
        ) from None


@contextmanager
def _client(config: AzureConfig):
    with _operation("Azure operation"), ExitStack() as stack:
        credential = create_credential(config)
        stack.callback(credential.close)
        yield create_ml_client(config, credential), credential


def _no_links(path: Path) -> None:
    if ".." in path.parts:
        raise DataError("Parent traversal is not allowed in local paths.")
    for candidate in (path, *path.absolute().parents):
        if candidate.is_symlink() or candidate.is_junction():
            raise DataError("Links and junctions are not allowed in local paths.")


def _project_root(project_root: Path | None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).absolute().parents[3]
    _no_links(root)
    if not root.is_dir():
        raise DataError("The project directory is missing.")
    return root.absolute()


def _receipt(root: Path, name: str, value: dict) -> None:
    path = root / ".azure" / name
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory replacement avoids truncated recovery receipts on interruption.
    pending = path.with_name(f".{name}-{uuid4().hex}")
    try:
        with pending.open("xb") as stream:
            stream.write(canonical_json(value))
        _no_links(path)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _prepare(data: Path) -> tuple[publishing._Bundle, str]:
    root = Path(data)
    manifest = verify_features(root)
    if (
        manifest.get("source") != SOURCE
        or manifest.get("feature_columns") != list(FEATURE_COLUMNS)
        or len(FEATURE_COLUMNS) != 35
        or manifest.get("target") != "rul"
    ):
        raise DataError("Feature source, columns or target differ from the approved recipe.")
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or len(entries) != len(_CONTENT)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in _CONTENT
            for entry in entries
        )
        or {entry["path"] for entry in entries} != _CONTENT
    ):
        raise DataError("Feature publication inventory must contain exactly five content files.")
    files = {
        entry["path"]: {key: entry[key] for key in ("sha256", "size_bytes")} for entry in entries
    }
    publishing._check_fingerprints(files)
    publishing._check_inventory(root, set(files) | {"manifest.json", "_SUCCESS.json"})
    if (root / "manifest.json").read_bytes() != canonical_json(manifest):
        raise DataError("Feature manifest changed after local verification.")
    files["manifest.json"] = fingerprint(root / "manifest.json")
    digest = files["manifest.json"]["sha256"]
    if (root / "_SUCCESS.json").read_bytes() != canonical_json(
        {"manifest_sha256": digest, "version": "sha256-" + digest}
    ):
        raise DataError("Feature completion marker changed after local verification.")
    files["_SUCCESS.json"] = fingerprint(root / "_SUCCESS.json")
    return publishing._Bundle(
        root, f"ml-ready/cmapss/sha256-{digest}/", files, "_SUCCESS.json"
    ), digest


def _asset_path(digest: str) -> str:
    return (
        f"azureml://datastores/{publishing.DATASTORE_NAME}/paths/ml-ready/cmapss/sha256-{digest}/"
    )


def _tags(digest: str) -> dict[str, str]:
    return {
        "manifest_sha256": digest,
        "source_asset_name": SOURCE["asset_name"],
        "source_asset_version": SOURCE["asset_version"],
        "source_manifest_sha256": SOURCE["manifest_sha256"],
        "dataset": "nasa-cmapss-ml-ready",
        "recipe_version": "1",
    }


def _get_asset(client, version: str):
    try:
        return client.data.get(name=ASSET_NAME, version=version)
    except ResourceNotFoundError:
        return None


def _check_asset(asset, version: str, digest: str, config: AzureConfig) -> None:
    full_path = (
        f"azureml://subscriptions/{config.subscription_id}/resourcegroups/{config.resource_group}"
        f"/workspaces/{config.workspace_name}/datastores/{publishing.DATASTORE_NAME}"
        f"/paths/ml-ready/cmapss/sha256-{digest}/"
    )
    if (
        not isinstance(asset, Data)
        or asset.name != ASSET_NAME
        or asset.version != version
        or version != publishing._asset_version(digest)
        or asset.type != "uri_folder"
        or not isinstance(asset.path, str)
        or asset.path.removesuffix("/")
        not in {_asset_path(digest).removesuffix("/"), full_path.removesuffix("/")}
        or not isinstance(asset.tags, dict)
        or any(asset.tags.get(key) != value for key, value in _tags(digest).items())
    ):
        raise DataError("Feature asset has conflicting identity, full hash, path or source tags.")


def _check_source(client, config: AzureConfig) -> None:
    source = client.data.get(name=SOURCE["asset_name"], version=SOURCE["asset_version"])
    publishing._check_asset(
        source,
        SOURCE["asset_version"],
        "sha256-" + SOURCE["manifest_sha256"],
        {"manifest_sha256": SOURCE["manifest_sha256"]},
        config,
    )


def _metadata(client, config: AzureConfig, digest: str, *, allow_missing: bool):
    _check_source(client, config)
    account = publishing._storage_account(client.workspaces.get(name=config.workspace_name), config)
    publishing._check_datastore(
        client.datastores.get(publishing.DATASTORE_NAME, include_secrets=False), account, config
    )
    asset = _get_asset(client, publishing._asset_version(digest))
    if asset is None:
        if not allow_missing:
            raise DataError("The exact feature asset version must already be published.")
    else:
        _check_asset(asset, publishing._asset_version(digest), digest, config)
    return account, asset


def publish(data: Path, config: AzureConfig, *, project_root: Path | None = None) -> dict:
    """Publish only the seven verified feature objects to the existing curated datastore."""
    with _operation("Feature publication"):
        root = _project_root(project_root)
        bundle, digest = _prepare(data)
        version = publishing._asset_version(digest)
        with _client(config) as (client, credential), ExitStack() as stack:
            account, _ = _metadata(client, config, digest, allow_missing=True)
            service = BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net",
                credential=credential,
                max_single_get_size=1024 * 1024,
                max_chunk_get_size=1024 * 1024,
                max_single_put_size=4 * 1024 * 1024,
                max_block_size=4 * 1024 * 1024,
                logging_enable=False,
            )
            stack.callback(service.close)
            container = service.get_container_client(publishing.CURATED_CONTAINER)
            if container.get_container_properties(logging_enable=False).public_access is not None:
                raise DataError("The existing curated container must be private.")
            uploaded, reused, etags = publishing._transfer(container, bundle, allow_write=True)
            _, asset = _metadata(client, config, digest, allow_missing=True)
            publishing._remote_inventory(container, bundle, etags)
            registered = asset is None
            if registered:
                client.data.create_or_update(
                    Data(
                        name=ASSET_NAME,
                        version=version,
                        type="uri_folder",
                        path=_asset_path(digest),
                        tags=_tags(digest),
                        description="Verified C-MAPSS baseline feature bundle.",
                    )
                )
            _metadata(client, config, digest, allow_missing=False)
        result = {
            "schema_version": 1,
            "asset_name": ASSET_NAME,
            "asset_version": version,
            "asset_ref": f"azureml:{ASSET_NAME}:{version}",
            "manifest_sha256": digest,
            "source": SOURCE.copy(),
            "datastore_name": publishing.DATASTORE_NAME,
            "blob_prefix": bundle.prefix,
            "objects_verified": len(bundle.files),
            "uploaded_objects": uploaded,
            "reused_objects": reused,
            "registered_new_version": registered,
        }
        _receipt(root, "ml-ready-publication.json", result)
        return result


def _stage_inventory(root: Path, expected: dict[str, bytes]) -> None:
    _no_links(root)
    found = set()
    directories = {
        str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."
    }
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            _no_links(path)
            relative = path.relative_to(root)
            if path.is_dir() and str(relative) in directories:
                pending.append(path)
            elif path.is_file() and relative.as_posix() in expected:
                payload = expected[relative.as_posix()]
                if path.stat().st_size != len(payload) or path.read_bytes() != payload:
                    raise DataError("The existing code staging content has changed.")
                found.add(relative.as_posix())
            else:
                raise DataError("Code staging contains an unexpected filesystem entry.")
    if found != set(expected):
        raise DataError("Code staging inventory is incomplete.")


def _stage_code(root: Path) -> tuple[Path, str]:
    snapshots = {}
    total = 0
    for name in sorted(_CODE_PATHS):
        path = root.joinpath(*_CODE_PATHS[name].split("/"))
        _no_links(path)
        if not path.is_file() or not 0 <= path.stat().st_size <= _MAX_CODE_FILE:
            raise DataError("An allowlisted code file is missing, nonregular or too large.")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_CODE_FILE + 1)
        total += len(payload)
        if len(payload) > _MAX_CODE_FILE or total > _MAX_CODE_BUNDLE:
            raise DataError("Code staging exceeds the approved file or bundle size limit.")
        if name.endswith(".sh") and b"\r" in payload:
            raise DataError("Linux job bootstrap must use LF, not Windows CRLF line endings.")
        snapshots[name] = payload
    digest = hashlib.sha256(
        canonical_json(
            [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in snapshots.items()
            ]
        )
    ).hexdigest()
    stage = root / ".azure" / "job-code" / digest
    _no_links(stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    if stage.exists():
        _stage_inventory(stage, snapshots)
        return stage, digest
    stage.mkdir()
    try:
        for name, payload in snapshots.items():
            destination = stage.joinpath(*name.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            _no_links(destination)
            with destination.open("xb") as stream:
                stream.write(payload)
        _stage_inventory(stage, snapshots)
    except Exception:
        shutil.rmtree(stage)
        raise
    return stage, digest


def _check_compute(compute, config: AzureConfig) -> None:
    text = {
        "name": "cpu-dev",
        "type": "amlcompute",
        "size": "standard_d2s_v3",
        "tier": "dedicated",
        "provisioning_state": "succeeded",
    }
    counts = {"min_instances": 0, "max_instances": 1, "idle_time_before_scale_down": 300}
    identity = getattr(compute, "identity", None)
    kind = getattr(identity, "type", None)
    principal = getattr(identity, "principal_id", None)
    if (
        config.compute_name != "cpu-dev"
        or any(
            str(getattr(getattr(compute, key, ""), "value", getattr(compute, key, ""))).casefold()
            != value
            for key, value in text.items()
        )
        or any(
            type(getattr(compute, key, None)) not in (int, float) or getattr(compute, key) != value
            for key, value in counts.items()
        )
        or getattr(compute, "enable_node_public_ip", None) is not False
        or getattr(compute, "ssh_public_access_enabled", None) is not False
        or not isinstance(kind, str)
        or kind.casefold() not in ("systemassigned", "system_assigned")
        or not isinstance(principal, str)
        or not principal.strip()
    ):
        raise DataError(
            "Compute must match the approved private CPU, scale-to-zero and SAI policy."
        )


def _selectors(asset_version: str, digest: str, environment_version: str) -> None:
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise DataError("A full lowercase SHA-256 feature manifest digest is required.")
    if asset_version != publishing._asset_version(digest):
        raise DataError("The feature asset selector does not match the full manifest digest.")
    if environment_version != ENVIRONMENT_VERSION or ENVIRONMENT_REF != (
        f"azureml://registries/azureml/environments/{ENVIRONMENT_NAME}/versions/{environment_version}"
    ):
        raise DataError("Only public curated sklearn-1.5 environment version 54 is approved.")


def _status(job, name: str) -> dict:
    if getattr(job, "name", None) != name:
        raise DataError("Returned job identity does not match the requested name.")
    value = getattr(job, "status", None)
    result = {"job_name": name, "status": value if value in _STATUSES else "Unknown"}
    if result["status"] == "Failed":
        result["error_code"] = "job_failed"
    elif result["status"] == "Canceled":
        result["error_code"] = "job_canceled"
    return result


def submit(
    asset_version: str,
    manifest_sha256: str,
    environment_version: str,
    config: AzureConfig,
    *,
    approve_costs: bool = False,
    project_root: Path | None = None,
) -> dict:
    """Submit once, explicitly approved, recording the job name before the uncertain write."""
    if approve_costs is not True:
        raise DataError(
            "Submission requires explicit --approve-costs for one CPU job up to 60 minutes."
        )
    _selectors(asset_version, manifest_sha256, environment_version)
    with _operation("Baseline submission"):
        root = _project_root(project_root)
        with _client(config) as (client, _):
            _metadata(client, config, manifest_sha256, allow_missing=False)
            _check_compute(client.compute.get(name=config.compute_name), config)
            code, codehash = _stage_code(root)
            name = "epm-baseline-" + uuid4().hex[:12]
            asset_ref = f"azureml:{ASSET_NAME}:{asset_version}"
            env_ref = ENVIRONMENT_REF
            job = command(
                name=name,
                experiment_name="epm-baseline-rul",
                code=str(code),
                command=(
                    "bash config/run-baseline.sh --data ${{inputs.ml_ready}} "
                    "--config config/baseline.json --output ${{outputs.baseline}}"
                ),
                environment=env_ref,
                compute="cpu-dev",
                instance_count=1,
                identity=ManagedIdentityConfiguration(),
                limits=CommandJobLimits(timeout=3600),
                inputs={"ml_ready": Input(type="uri_folder", path=asset_ref, mode="download")},
                outputs={
                    "baseline": Output(
                        type="uri_folder",
                        mode="upload",
                        path=f"azureml://datastores/workspaceblobstore/paths/baseline/{name}/",
                    )
                },
                environment_variables={
                    "PYTHONPATH": "./src",
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "PYTHONHASHSEED": "42",
                },
                tags={**_tags(manifest_sha256), "code_sha256": codehash},
            )
            receipt = {
                "schema_version": 1,
                "job_name": name,
                "status": "SubmissionPending",
                "asset_ref": asset_ref,
                "env_ref": env_ref,
                "codehash": codehash,
                "manifest_sha256": manifest_sha256,
            }
            _receipt(root, "baseline-job.json", receipt)
            try:
                submitted = client.jobs.create_or_update(job)
                receipt.update(_status(submitted, name))
            except Exception:
                receipt["status"] = "SubmissionUnknown"
                _receipt(root, "baseline-job.json", receipt)
                raise
            _receipt(root, "baseline-job.json", receipt)
            return receipt


def _job_name(name: str) -> None:
    if not isinstance(name, str) or not _JOB_NAME.fullmatch(name):
        raise DataError("A generated epm-baseline job name is required.")


def status(job_name: str, config: AzureConfig) -> dict:
    """Read only the named job; never return service messages, URLs or resource IDs."""
    _job_name(job_name)
    with _client(config) as (client, _):
        return _status(client.jobs.get(name=job_name), job_name)


def download(job_name: str, destination: Path, config: AzureConfig) -> dict:
    """Retrieve and verify the exact completed output directly, never trusting an SDK no-op."""
    _job_name(job_name)
    marker = "artifact-manifest.json"
    content_names = {
        "evaluation.md",
        "feature-importance.json",
        "metrics.json",
        "model.json",
        "predictions_test.parquet",
        "predictions_validation.parquet",
        "run-metadata.json",
    }
    expected_names = content_names | {marker}
    prefix = f"baseline/{job_name}/"
    expected_uri = f"azureml://datastores/workspaceblobstore/paths/{prefix}"
    stage = None
    published = []

    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise DataError("Artifact manifest contains duplicate keys.")
            value[key] = item
        return value

    with _operation("Baseline download"):
        destination = Path(destination)
        _no_links(destination)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise DataError("Download destination must be a new or empty directory.")
        try:
            with _client(config) as (client, credential), ExitStack() as stack:
                job = client.jobs.get(name=job_name)
                result = _status(job, job_name)
                if result["status"] != "Completed":
                    raise DataError("Only Completed baseline jobs may be downloaded.")
                outputs = getattr(job, "outputs", None)
                output = outputs.get("baseline") if isinstance(outputs, dict) else None
                if (
                    getattr(output, "type", None) != "uri_folder"
                    or getattr(output, "path", None) != expected_uri
                ):
                    raise DataError("The baseline output must match the exact approved job path.")
                account = publishing._storage_account(
                    client.workspaces.get(name=config.workspace_name), config
                )
                datastore = client.datastores.get("workspaceblobstore", include_secrets=False)
                default = client.datastores.get_default(include_secrets=False)
                expected_id = (
                    f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}"
                    "/providers/Microsoft.MachineLearningServices/workspaces/"
                    f"{config.workspace_name}/datastores/workspaceblobstore"
                )
                for store in (datastore, default):
                    if (
                        not isinstance(store, AzureBlobDatastore)
                        or store.name != "workspaceblobstore"
                        or store.account_name != account
                        or store.protocol != "https"
                        or store.endpoint != "core.windows.net"
                        or not isinstance(store.container_name, str)
                        or not 3 <= len(store.container_name) <= 63
                        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", store.container_name)
                        or not (
                            store.credentials is None
                            or isinstance(store.credentials, NoneCredentialConfiguration)
                        )
                        or (
                            store.id is not None
                            and store.id.rstrip("/").casefold() != expected_id.casefold()
                        )
                    ):
                        raise DataError(
                            "Output datastore must match the keyless workspace storage."
                        )
                if datastore.container_name != default.container_name:
                    raise DataError("Output container must match the workspace default datastore.")
                service = BlobServiceClient(
                    account_url=f"https://{account}.blob.core.windows.net",
                    credential=credential,
                    max_single_get_size=1024 * 1024,
                    max_chunk_get_size=1024 * 1024,
                    logging_enable=False,
                )
                stack.callback(service.close)
                container = service.get_container_client(datastore.container_name)
                if (
                    container.get_container_properties(logging_enable=False).public_access
                    is not None
                ):
                    raise DataError("The workspace output container must be private.")
                inventory = {}
                for item in container.list_blobs(name_starts_with=prefix, logging_enable=False):
                    if (
                        item.name not in {prefix + name for name in expected_names}
                        or item.name in inventory
                        or type(item.size) is not int
                        or not 0 < item.size <= 64 * 1024 * 1024
                        or not isinstance(item.etag, str)
                        or not item.etag.strip('"')
                    ):
                        raise DataError(
                            "Remote baseline inventory has an unexpected or invalid object."
                        )
                    inventory[item.name] = item
                if set(inventory) != {prefix + name for name in expected_names}:
                    raise DataError(
                        "Remote baseline inventory must contain exactly eight artifacts."
                    )
                if sum(item.size for item in inventory.values()) > 256 * 1024 * 1024:
                    raise DataError("Baseline output exceeds the bounded download size.")
                if inventory[prefix + marker].size > 1024 * 1024:
                    raise DataError("Artifact manifest exceeds the bounded download size.")
                _no_links(destination)
                destination.mkdir(parents=True, exist_ok=True)
                if any(destination.iterdir()):
                    raise DataError("Download destination changed before staging.")
                stage = destination / (".download-" + uuid4().hex)
                stage.mkdir()
                files, etags = {}, {}

                def fetch(name):
                    listed = inventory[prefix + name]
                    blob = container.get_blob_client(prefix + name)
                    properties = blob.get_blob_properties(logging_enable=False)
                    if (
                        properties.size != listed.size
                        or not isinstance(properties.etag, str)
                        or properties.etag.strip('"') != listed.etag.strip('"')
                    ):
                        raise DataError("Output blob changed after inventory listing.")
                    condition = {
                        "etag": properties.etag,
                        "match_condition": MatchConditions.IfNotModified,
                    }
                    stream = blob.download_blob(
                        **condition, max_concurrency=1, logging_enable=False
                    )
                    local = stage / name
                    _no_links(local)
                    digest, size = hashlib.sha256(), 0
                    with local.open("xb") as target:
                        for chunk in stream.chunks():
                            size += len(chunk)
                            if size > listed.size:
                                raise DataError("Output stream exceeds its declared size.")
                            target.write(chunk)
                            digest.update(chunk)
                    if size != listed.size:
                        raise DataError("Output stream is shorter than its declared size.")
                    blob.get_blob_properties(**condition, logging_enable=False)
                    etags[prefix + name] = properties.etag
                    files[name] = {"sha256": digest.hexdigest(), "size_bytes": size}

                fetch(marker)
                manifest = json.loads((stage / marker).read_bytes(), object_pairs_hook=unique_keys)
                if (
                    not isinstance(manifest, dict)
                    or set(manifest)
                    != {
                        "schema_version",
                        "artifact_type",
                        "files",
                        "provenance",
                        "completion_marker",
                    }
                    or type(manifest["schema_version"]) is not int
                    or manifest["schema_version"] != 1
                    or manifest["artifact_type"] != "xgboost-rul-baseline"
                    or manifest["completion_marker"] != marker
                    or not isinstance(manifest["provenance"], dict)
                    or not isinstance(manifest["files"], list)
                    or len(manifest["files"]) != len(content_names)
                ):
                    raise DataError("Artifact manifest does not match the baseline schema.")
                fingerprints = {}
                for entry in manifest["files"]:
                    if (
                        not isinstance(entry, dict)
                        or set(entry) != {"path", "sha256", "size_bytes"}
                        or not isinstance(entry["path"], str)
                        or entry["path"] not in content_names
                        or entry["path"] in fingerprints
                    ):
                        raise DataError("Artifact manifest has an unsafe or duplicate file entry.")
                    fingerprints[entry["path"]] = {
                        key: entry[key] for key in ("sha256", "size_bytes")
                    }
                publishing._check_fingerprints(fingerprints)
                if set(fingerprints) != content_names:
                    raise DataError("Artifact manifest content inventory is incomplete.")
                for name in sorted(content_names):
                    if fingerprints[name]["size_bytes"] != inventory[prefix + name].size:
                        raise DataError("Artifact manifest size differs from the remote output.")
                    fetch(name)
                    if files[name] != fingerprints[name]:
                        raise DataError("Downloaded artifact checksum differs from its manifest.")
                publishing._remote_inventory(
                    container, publishing._Bundle(stage, prefix, files, marker), etags
                )
                publishing._check_inventory(stage, expected_names)
                _no_links(destination)
                if set(destination.iterdir()) != {stage}:
                    raise DataError("Download destination changed before finalization.")
                for name in [*sorted(content_names), marker]:
                    source, target = stage / name, destination / name
                    _no_links(source)
                    _no_links(target)
                    with source.open("rb") as input_stream, target.open("xb") as output_stream:
                        published.append(target)
                        shutil.copyfileobj(input_stream, output_stream)
                    if fingerprint(target) != files[name]:
                        raise DataError("Downloaded artifact changed during finalization.")
                shutil.rmtree(stage)
                stage = None
                publishing._check_inventory(destination, expected_names)
                return {
                    **result,
                    "output_name": "baseline",
                    "downloaded": True,
                    "objects_verified": len(files),
                    "artifact_manifest_sha256": files[marker]["sha256"],
                }
        except Exception:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
        finally:
            if stage is not None:
                shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("publish").add_argument("--data", type=Path, required=True)
    submission = commands.add_parser("submit")
    for option in ("--asset-version", "--manifest-sha256"):
        submission.add_argument(option, required=True)
    submission.add_argument("--environment-version", default=ENVIRONMENT_VERSION)
    submission.add_argument("--approve-costs", action="store_true")
    commands.add_parser("status").add_argument("--job-name", required=True)
    downloading = commands.add_parser("download")
    downloading.add_argument("--job-name", required=True)
    downloading.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    previous_logging_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        # SDK progress output is not a safe receipt and can contain cloud identifiers.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            config = load_config()
            if args.operation == "publish":
                result = publish(args.data, config)
            elif args.operation == "submit":
                result = submit(
                    args.asset_version,
                    args.manifest_sha256,
                    args.environment_version,
                    config,
                    approve_costs=args.approve_costs,
                )
            elif args.operation == "status":
                result = status(args.job_name, config)
            else:
                result = download(args.job_name, args.destination, config)
        exit_code = 0
    except DataError as error:
        result, exit_code = (
            {"status": "failed", "code": "baseline_operation_failed", "message": str(error)},
            1,
        )
    except Exception:
        result, exit_code = (
            {
                "status": "failed",
                "code": "baseline_operation_failed",
                "message": (
                    "Baseline operation failed. Check configuration and the local recovery receipt."
                ),
            },
            1,
        )
    finally:
        logging.disable(previous_logging_threshold)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
