"""Explicit, bounded Azure ML operations for the approved CPU PyTorch experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import shutil
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from uuid import uuid4

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import (
    AzureBlobDatastore,
    CommandJobLimits,
    ManagedIdentityConfiguration,
    NoneCredentialConfiguration,
)
from azure.core import MatchConditions
from azure.storage.blob import BlobServiceClient

from epm_platform.baseline import azure as safe
from epm_platform.config import AzureConfig, load_config
from epm_platform.data import publishing
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint

ASSET_NAME = "epm-cmapss-ml-ready"
ASSET_VERSION = "d-f4ueae6u7g4c5islgehonqveey"
MANIFEST_SHA256 = "2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
ENVIRONMENT_VERSION = "54"
ENVIRONMENT_REF = safe.ENVIRONMENT_REF
_JOB_NAME = re.compile(r"epm-pytorch-[a-f0-9]{12}")
_MARKER = "artifact-manifest.json"
_CONTENT = frozenset(
    {
        "model.pt",
        "model-spec.json",
        "preprocessing.json",
        "metrics.json",
        "predictions_validation.parquet",
        "predictions_test.parquet",
        "run-metadata.json",
        "training-history.json",
        "evaluation.md",
        "comparison.json",
    }
)
_CODE_PATHS = (
    {
        name: source
        for name, source in safe._CODE_PATHS.items()
        if name
        not in {"config/baseline.json", "config/run-baseline.sh", "config/runtime-requirements.txt"}
    }
    | {
        name: name
        for name in (
            "src/epm_platform/deep_learning/__init__.py",
            "src/epm_platform/deep_learning/training.py",
            "src/epm_platform/deep_learning/tracking.py",
            "config/pytorch.json",
            "config/baseline-reference.json",
        )
    }
    | {
        "config/run-pytorch.sh": "scripts/run-pytorch.sh",
        "config/runtime-requirements.txt": "environments/pytorch/requirements-linux.txt",
    }
)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("JSON contains duplicate keys.")
        result[key] = value
    return result


def _stage_code(root: Path) -> tuple[Path, str]:
    """Snapshot only reviewed code, never a repository or data-directory upload."""
    snapshots, total = {}, 0
    for name, source in sorted(_CODE_PATHS.items()):
        path = root.joinpath(*source.split("/"))
        safe._no_links(path)
        if not path.is_file() or not 0 < path.stat().st_size <= safe._MAX_CODE_FILE:
            raise DataError("An allowlisted PyTorch code file is missing or too large.")
        with path.open("rb") as stream:
            payload = stream.read(safe._MAX_CODE_FILE + 1)
        total += len(payload)
        if len(payload) > safe._MAX_CODE_FILE or total > safe._MAX_CODE_BUNDLE:
            raise DataError("PyTorch code staging exceeds its bounded size.")
        if name.endswith(".sh") and b"\r" in payload:
            raise DataError("Linux job bootstrap must use LF line endings.")
        snapshots[name] = payload
    recipe = json.loads(snapshots["config/pytorch.json"], object_pairs_hook=_unique_keys)
    if (
        recipe.get("device") != "cpu"
        or type(recipe.get("threads")) is not int
        or recipe["threads"] != 2
        or type(recipe.get("max_epochs")) is not int
        or not 1 <= recipe["max_epochs"] <= 100
        or type(recipe.get("patience")) is not int
        or not 1 <= recipe["patience"] <= min(12, recipe["max_epochs"])
        or type(recipe.get("seed")) is not int
        or recipe["seed"] != 42
    ):
        raise DataError("PyTorch configuration exceeds the approved CPU training budget.")
    reference = json.loads(
        snapshots["config/baseline-reference.json"], object_pairs_hook=_unique_keys
    )
    if reference.get("ml_ready_manifest_sha256") != MANIFEST_SHA256:
        raise DataError("The baseline reference must identify the approved feature digest.")
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
    stage = root / ".azure" / "pytorch-code" / digest
    safe._no_links(stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    if stage.exists():
        safe._stage_inventory(stage, snapshots)
        return stage, digest
    stage.mkdir()
    try:
        for name, payload in snapshots.items():
            target = stage.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            safe._no_links(target)
            with target.open("xb") as stream:
                stream.write(payload)
        safe._stage_inventory(stage, snapshots)
    except Exception:
        shutil.rmtree(stage)
        raise
    return stage, digest


def submit(
    asset_version: str,
    manifest_sha256: str,
    environment_version: str,
    config: AzureConfig,
    *,
    approve_costs: bool = False,
    project_root: Path | None = None,
) -> dict:
    """Submit once; persist the generated name before the potentially ambiguous write."""
    if approve_costs is not True:
        raise DataError("Submission requires explicit --approve-costs for one 60-minute CPU job.")
    if (asset_version, manifest_sha256, environment_version) != (
        ASSET_VERSION,
        MANIFEST_SHA256,
        ENVIRONMENT_VERSION,
    ):
        raise DataError("Only the exact approved feature asset and pinned environment are allowed.")
    with safe._operation("PyTorch submission"):
        root = safe._project_root(project_root)
        code, codehash = _stage_code(root)
        with safe._client(config) as (client, _):
            safe._metadata(client, config, MANIFEST_SHA256, allow_missing=False)
            safe._check_compute(client.compute.get(name=config.compute_name), config)
            name = "epm-pytorch-" + uuid4().hex[:12]
            asset_ref = f"azureml:{ASSET_NAME}:{ASSET_VERSION}"
            job = command(
                name=name,
                experiment_name="epm-pytorch-rul",
                code=str(code),
                command=(
                    "bash config/run-pytorch.sh --data ${{inputs.ml_ready}} "
                    "--config config/pytorch.json "
                    "--baseline-reference config/baseline-reference.json "
                    "--output ${{outputs.pytorch}}"
                ),
                environment=ENVIRONMENT_REF,
                compute="cpu-dev",
                instance_count=1,
                identity=ManagedIdentityConfiguration(),
                limits=CommandJobLimits(timeout=3600),
                inputs={"ml_ready": Input(type="uri_folder", path=asset_ref, mode="download")},
                outputs={
                    "pytorch": Output(
                        type="uri_folder",
                        mode="upload",
                        path=f"azureml://datastores/workspaceblobstore/paths/pytorch/{name}/",
                    )
                },
                environment_variables={
                    "PYTHONPATH": "./src",
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "PYTHONHASHSEED": "42",
                    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
                    "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
                    "MLFLOW_ENABLE_TELEMETRY": "false",
                },
                tags={**safe._tags(MANIFEST_SHA256), "code_sha256": codehash},
            )
            receipt = {
                "schema_version": 1,
                "job_name": name,
                "status": "SubmissionPending",
                "asset_ref": asset_ref,
                "env_ref": ENVIRONMENT_REF,
                "codehash": codehash,
                "manifest_sha256": MANIFEST_SHA256,
            }
            safe._receipt(root, "pytorch-job.json", receipt)
            try:
                receipt.update(safe._status(client.jobs.create_or_update(job), name))
            except Exception:
                receipt["status"] = "SubmissionUnknown"
                safe._receipt(root, "pytorch-job.json", receipt)
                raise
            safe._receipt(root, "pytorch-job.json", receipt)
            return receipt


def _job_name(name: str) -> None:
    if not isinstance(name, str) or not _JOB_NAME.fullmatch(name):
        raise DataError("A generated epm-pytorch job name is required.")


def status(job_name: str, config: AzureConfig) -> dict:
    """Read a sanitized status, without service errors or automatic cancellation."""
    _job_name(job_name)
    with safe._client(config) as (client, _):
        return safe._status(client.jobs.get(name=job_name), job_name)


def _output_container(client, credential, config: AzureConfig, stack: ExitStack):
    account = publishing._storage_account(client.workspaces.get(name=config.workspace_name), config)
    datastore = client.datastores.get("workspaceblobstore", include_secrets=False)
    default = client.datastores.get_default(include_secrets=False)
    expected_id = (
        f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{config.workspace_name}"
        "/datastores/workspaceblobstore"
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
            or (store.id is not None and store.id.rstrip("/").casefold() != expected_id.casefold())
        ):
            raise DataError("Output datastore must match the keyless workspace storage.")
    if datastore.container_name != default.container_name:
        raise DataError("Output container differs from the workspace default datastore.")
    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=credential,
        max_single_get_size=1024 * 1024,
        max_chunk_get_size=1024 * 1024,
        logging_enable=False,
    )
    stack.callback(service.close)
    container = service.get_container_client(datastore.container_name)
    if container.get_container_properties(logging_enable=False).public_access is not None:
        raise DataError("The workspace output container must be private.")
    return container


def _inventory(container, prefix: str) -> dict:
    expected = {prefix + name for name in _CONTENT | {_MARKER}}
    inventory = {}
    for item in container.list_blobs(name_starts_with=prefix, logging_enable=False):
        if (
            item.name not in expected
            or item.name in inventory
            or type(item.size) is not int
            or not 0 < item.size <= 64 * 1024 * 1024
            or not isinstance(item.etag, str)
            or not item.etag.strip('"')
        ):
            raise DataError("Remote PyTorch inventory has an unexpected or invalid object.")
        inventory[item.name] = item
    if set(inventory) != expected:
        raise DataError("Remote PyTorch inventory must contain exactly eleven artifacts.")
    if (
        sum(item.size for item in inventory.values()) > 256 * 1024 * 1024
        or inventory[prefix + _MARKER].size > 1024 * 1024
    ):
        raise DataError("PyTorch artifacts exceed the bounded download size.")
    return inventory


def _fetch(container, prefix: str, name: str, listed, stage: Path) -> tuple[dict, str]:
    blob = container.get_blob_client(prefix + name)
    properties = blob.get_blob_properties(logging_enable=False)
    if (
        properties.size != listed.size
        or not isinstance(properties.etag, str)
        or properties.etag.strip('"') != listed.etag.strip('"')
    ):
        raise DataError("Output blob changed after inventory listing.")
    condition = {"etag": properties.etag, "match_condition": MatchConditions.IfNotModified}
    stream = blob.download_blob(**condition, max_concurrency=1, logging_enable=False)
    target = stage / name
    safe._no_links(target)
    digest, size = hashlib.sha256(), 0
    with target.open("xb") as output:
        for chunk in stream.chunks():
            size += len(chunk)
            if size > listed.size:
                raise DataError("Output stream exceeds its declared size.")
            output.write(chunk)
            digest.update(chunk)
    if size != listed.size:
        raise DataError("Output stream is shorter than its declared size.")
    blob.get_blob_properties(**condition, logging_enable=False)
    return {"sha256": digest.hexdigest(), "size_bytes": size}, properties.etag


def _manifest(path: Path) -> dict:
    manifest = json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "artifact_type",
            "model_sha256",
            "files",
            "provenance",
            "completion_marker",
        }
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["artifact_type"] != "pytorch-mlp-rul"
        or manifest["completion_marker"] != _MARKER
        or not isinstance(manifest["provenance"], dict)
        or manifest["provenance"].get("ml_ready_manifest_sha256") != MANIFEST_SHA256
        or manifest["provenance"].get("source") != safe.SOURCE
        or not isinstance(manifest["files"], list)
        or len(manifest["files"]) != len(_CONTENT)
    ):
        raise DataError("Artifact manifest does not match the approved PyTorch schema/source.")
    files = {}
    for entry in manifest["files"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in _CONTENT
            or entry["path"] in files
        ):
            raise DataError("Artifact manifest has an unsafe or duplicate file entry.")
        files[entry["path"]] = {key: entry[key] for key in ("sha256", "size_bytes")}
    publishing._check_fingerprints(files)
    if set(files) != _CONTENT or manifest["model_sha256"] != files["model.pt"]["sha256"]:
        raise DataError("Artifact manifest has conflicting content or model fingerprints.")
    return files


def download(job_name: str, destination: Path, config: AzureConfig) -> dict:
    """Conditionally stream eleven artifacts; validate bytes without loading a checkpoint."""
    _job_name(job_name)
    prefix = f"pytorch/{job_name}/"
    expected_uri = f"azureml://datastores/workspaceblobstore/paths/{prefix}"
    expected = _CONTENT | {_MARKER}
    stage, published = None, []
    with safe._operation("PyTorch download"):
        destination = Path(destination)
        safe._no_links(destination)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise DataError("Download destination must be a new or empty directory.")
        try:
            with safe._client(config) as (client, credential), ExitStack() as stack:
                job = client.jobs.get(name=job_name)
                result = safe._status(job, job_name)
                if result["status"] != "Completed":
                    raise DataError("Only Completed PyTorch jobs may be downloaded.")
                outputs = getattr(job, "outputs", None)
                output = outputs.get("pytorch") if isinstance(outputs, dict) else None
                if (
                    getattr(output, "type", None) != "uri_folder"
                    or getattr(output, "path", None) != expected_uri
                ):
                    raise DataError("The PyTorch output must match the exact approved job path.")
                container = _output_container(client, credential, config, stack)
                inventory = _inventory(container, prefix)
                safe._no_links(destination)
                destination.mkdir(parents=True, exist_ok=True)
                if any(destination.iterdir()):
                    raise DataError("Download destination changed before staging.")
                stage = destination / (".download-" + uuid4().hex)
                stage.mkdir()
                files, etags = {}, {}
                files[_MARKER], etags[prefix + _MARKER] = _fetch(
                    container, prefix, _MARKER, inventory[prefix + _MARKER], stage
                )
                declared = _manifest(stage / _MARKER)
                for name in sorted(_CONTENT):
                    if declared[name]["size_bytes"] != inventory[prefix + name].size:
                        raise DataError("Artifact manifest size differs from the remote output.")
                    files[name], etags[prefix + name] = _fetch(
                        container, prefix, name, inventory[prefix + name], stage
                    )
                    if files[name] != declared[name]:
                        raise DataError("Downloaded artifact checksum differs from its manifest.")
                publishing._remote_inventory(
                    container, publishing._Bundle(stage, prefix, files, _MARKER), etags
                )
                publishing._check_inventory(stage, expected)
                safe._no_links(destination)
                if set(destination.iterdir()) != {stage}:
                    raise DataError("Download destination changed before finalization.")
                for name in [*sorted(_CONTENT), _MARKER]:
                    source, target = stage / name, destination / name
                    safe._no_links(source)
                    safe._no_links(target)
                    with source.open("rb") as incoming, target.open("xb") as outgoing:
                        published.append(target)
                        shutil.copyfileobj(incoming, outgoing)
                    if fingerprint(target) != files[name]:
                        raise DataError("Downloaded artifact changed during finalization.")
                shutil.rmtree(stage)
                stage = None
                publishing._check_inventory(destination, expected)
                return {
                    **result,
                    "output_name": "pytorch",
                    "downloaded": True,
                    "objects_verified": len(files),
                    "artifact_manifest_sha256": files[_MARKER]["sha256"],
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
    submission = commands.add_parser("submit")
    submission.add_argument("--asset-version", default=ASSET_VERSION)
    submission.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    submission.add_argument("--environment-version", default=ENVIRONMENT_VERSION)
    submission.add_argument("--approve-costs", action="store_true")
    commands.add_parser("status").add_argument("--job-name", required=True)
    downloading = commands.add_parser("download")
    downloading.add_argument("--job-name", required=True)
    downloading.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            config = load_config()
            if args.operation == "submit":
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
    except Exception:
        result, exit_code = (
            {
                "status": "failed",
                "code": "pytorch_operation_failed",
                "message": "PyTorch operation failed; check configuration and the job receipt.",
            },
            1,
        )
    finally:
        logging.disable(previous)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
