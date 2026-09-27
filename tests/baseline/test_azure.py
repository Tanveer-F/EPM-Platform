"""Offline Azure contracts; all disposable test artifacts remain inside the project."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from azure.ai.ml.entities import AzureBlobDatastore, Data
from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)

from epm_platform.baseline import azure
from epm_platform.data import publishing
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint

ROOT = Path(__file__).absolute().parents[2]
SECRET = "SENSITIVE-BODY-https://example.invalid/?sig=secret-token"
JOB = "epm-baseline-012345abcdef"


@pytest.fixture
def workspace():
    path = ROOT / (".baseline-azure-test-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def project(workspace):
    project = workspace / "project"
    for name in azure._CODE_PATHS.values():
        target = project.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "scripts/run-baseline.sh":
            target.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
        else:
            shutil.copyfile(ROOT.joinpath(*name.split("/")), target)
    return project


@pytest.fixture
def bundle(workspace, monkeypatch):
    root = workspace / "arbitrary-mounted-folder"
    root.mkdir()
    for name in azure._CONTENT:
        (root / name).write_bytes((name + "\n").encode())
    manifest = {
        "source": azure.SOURCE.copy(),
        "feature_columns": list(azure.FEATURE_COLUMNS),
        "target": "rul",
        "row_counts": {"train": 20, "validation": 4, "test": 4},
        "files": [{"path": name, **fingerprint(root / name)} for name in sorted(azure._CONTENT)],
    }
    (root / "manifest.json").write_bytes(canonical_json(manifest))
    digest = fingerprint(root / "manifest.json")["sha256"]
    (root / "_SUCCESS.json").write_bytes(
        canonical_json({"manifest_sha256": digest, "version": "sha256-" + digest})
    )
    monkeypatch.setattr(azure, "verify_features", Mock(return_value=manifest))
    return SimpleNamespace(
        root=root, manifest=manifest, digest=digest, version=publishing._asset_version(digest)
    )


class Blob:
    def __init__(self, container, name):
        self.container, self.name = container, name

    def get_blob_properties(self, **kwargs):
        if self.name not in self.container.objects:
            raise ResourceNotFoundError(SECRET)
        content = self.container.objects[self.name]
        if "etag" in kwargs:
            assert kwargs["etag"] == '"revision-1"'
            assert kwargs["match_condition"] == MatchConditions.IfNotModified
        return SimpleNamespace(size=len(content), etag='"revision-1"')

    def upload_blob(self, stream, **kwargs):
        assert kwargs["overwrite"] is False
        assert kwargs["max_concurrency"] == 1
        assert kwargs["logging_enable"] is False
        if self.name in self.container.objects:
            raise ResourceExistsError(SECRET)
        payload = stream.read()
        assert len(payload) == kwargs["length"]
        assert hashlib.sha256(payload).hexdigest() == kwargs["metadata"]["sha256"]
        self.container.objects[self.name] = payload
        self.container.events.append(("upload", self.name))

    def download_blob(self, **kwargs):
        assert kwargs["etag"] == '"revision-1"'
        assert kwargs["match_condition"] == MatchConditions.IfNotModified
        return SimpleNamespace(chunks=lambda: [self.container.objects[self.name]])


class Container:
    def __init__(self, events):
        self.events, self.objects = events, {}
        self.public_access = None

    def get_container_properties(self, **kwargs):
        return SimpleNamespace(public_access=self.public_access)

    def get_blob_client(self, name):
        assert name.startswith("ml-ready/cmapss/sha256-")
        return Blob(self, name)

    def list_blobs(self, *, name_starts_with, **kwargs):
        self.events.append(("inventory", name_starts_with))
        return [
            SimpleNamespace(name=name, etag="revision-1")
            for name in self.objects
            if name.startswith(name_starts_with)
        ]


@pytest.fixture
def cloud(config, fake_credential, monkeypatch):
    events = []
    source = Data(
        name=azure.SOURCE["asset_name"],
        version=azure.SOURCE["asset_version"],
        type="uri_folder",
        path=publishing._asset_path("sha256-" + azure.SOURCE["manifest_sha256"]),
        tags={"manifest_sha256": azure.SOURCE["manifest_sha256"]},
    )
    assets = {(source.name, source.version): source}
    client = Mock()

    def get_asset(*, name, version):
        events.append(("get_asset", name))
        if (name, version) not in assets:
            raise ResourceNotFoundError(SECRET)
        return assets[name, version]

    def register(asset):
        events.append(("register", asset.name))
        assets[asset.name, asset.version] = asset
        return asset

    client.data.get.side_effect = get_asset
    client.data.create_or_update.side_effect = register
    client.workspaces.get.return_value = SimpleNamespace(
        storage_account=(
            f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}"
            "/providers/Microsoft.Storage/storageAccounts/epmstorage000"
        )
    )
    client.datastores.get.return_value = AzureBlobDatastore(
        name=publishing.DATASTORE_NAME,
        account_name="epmstorage000",
        container_name=publishing.CURATED_CONTAINER,
        protocol="https",
        endpoint="core.windows.net",
    )
    client.compute.get.return_value = SimpleNamespace(
        name="cpu-dev",
        type="AmlCompute",
        size="Standard_D2s_v3",
        tier="Dedicated",
        provisioning_state="Succeeded",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=300,
        enable_node_public_ip=False,
        ssh_public_access_enabled=False,
        identity=SimpleNamespace(type="SystemAssigned", principal_id="compute-sai"),
    )
    client.environments.get.side_effect = AssertionError("No workspace environment lookup allowed")
    client.jobs.create_or_update.side_effect = lambda job: SimpleNamespace(
        name=job.name, status="Queued"
    )
    client.jobs.get.return_value = SimpleNamespace(name=JOB, status="Completed")
    container = Container(events)
    service = Mock()
    service.get_container_client.return_value = container
    credential_factory = Mock(return_value=fake_credential)
    monkeypatch.setattr(azure, "create_credential", credential_factory)
    monkeypatch.setattr(azure, "create_ml_client", Mock(return_value=client))
    blob_factory = Mock(return_value=service)
    monkeypatch.setattr(azure, "BlobServiceClient", blob_factory)
    return SimpleNamespace(
        client=client,
        assets=assets,
        events=events,
        container=container,
        service=service,
        source=source,
        credential_factory=credential_factory,
        blob_factory=blob_factory,
    )


def feature_asset(bundle):
    return Data(
        name=azure.ASSET_NAME,
        version=bundle.version,
        type="uri_folder",
        path=azure._asset_path(bundle.digest),
        tags=azure._tags(bundle.digest),
    )


def add_feature(cloud, bundle):
    asset = feature_asset(bundle)
    cloud.assets[asset.name, asset.version] = asset
    return asset


def submit(bundle, config, project, **kwargs):
    return azure.submit(
        bundle.version,
        bundle.digest,
        "54",
        config,
        approve_costs=True,
        project_root=project,
        **kwargs,
    )


def test_publish_seven_objects_marker_last_and_resume(bundle, cloud, config, project):
    result = azure.publish(bundle.root, config, project_root=project)
    assert result["objects_verified"] == 7
    assert result["uploaded_objects"] == 7
    assert result["asset_version"] == bundle.version
    assert len(bundle.version) == 28
    uploads = [event for event in cloud.events if event[0] == "upload"]
    assert uploads[-1][1].endswith("/_SUCCESS.json")
    assert cloud.events.index(("get_asset", azure.SOURCE["asset_name"])) < cloud.events.index(
        uploads[0]
    )
    registration = cloud.events.index(("register", azure.ASSET_NAME))
    assert cloud.events[registration - 1][0] == "inventory"
    assert all(name.startswith(result["blob_prefix"]) for name in cloud.container.objects)
    cloud.service.get_container_client.assert_called_once_with(publishing.CURATED_CONTAINER)
    cloud.client.datastores.create_or_update.assert_not_called()
    cloud.client.jobs.create_or_update.assert_not_called()
    cloud.client.data.create_or_update.assert_called_once()
    receipt = project / ".azure" / "ml-ready-publication.json"
    assert json.loads(receipt.read_bytes()) == result
    assert config.subscription_id not in receipt.read_text()
    assert SECRET not in receipt.read_text()
    resumed = azure.publish(bundle.root, config, project_root=project)
    assert resumed["uploaded_objects"] == 0
    assert resumed["reused_objects"] == 7
    assert resumed["registered_new_version"] is False
    cloud.client.data.create_or_update.assert_called_once()


@pytest.mark.parametrize(
    "field",
    [
        "manifest_sha256",
        "source_asset_name",
        "source_asset_version",
        "source_manifest_sha256",
        "path",
        "type",
        "version",
    ],
)
def test_conflicting_feature_registration_stops_before_writes(
    field, bundle, cloud, config, project
):
    asset = add_feature(cloud, bundle)
    if field in asset.tags:
        asset.tags[field] = "incorrect"
    else:
        setattr(asset, field, "incorrect")
    with pytest.raises(DataError, match="conflicting"):
        azure.publish(bundle.root, config, project_root=project)
    cloud.blob_factory.assert_not_called()
    cloud.client.data.create_or_update.assert_not_called()


@pytest.mark.parametrize("field", ["manifest_sha256", "path", "version", "type"])
def test_incorrect_registered_source_stops_before_writes(field, bundle, cloud, config, project):
    if field == "manifest_sha256":
        cloud.source.tags[field] = "0" * 64
    else:
        setattr(cloud.source, field, "incorrect")
    with pytest.raises(DataError):
        azure.publish(bundle.root, config, project_root=project)
    cloud.blob_factory.assert_not_called()
    cloud.client.data.create_or_update.assert_not_called()


def test_local_verification_precedes_authentication(bundle, cloud, config, project):
    azure.verify_features.side_effect = DataError("Invalid features.")
    with pytest.raises(DataError, match="Invalid features"):
        azure.publish(bundle.root, config, project_root=project)
    cloud.credential_factory.assert_not_called()


@pytest.mark.parametrize("mutation", ["extra", "marker", "source", "traversal"])
def test_local_prepare_fails_closed(mutation, bundle):
    if mutation == "extra":
        (bundle.root / "credentials.env").write_text(SECRET)
    elif mutation == "marker":
        (bundle.root / "_SUCCESS.json").write_bytes(b"{}")
    else:
        if mutation == "source":
            bundle.manifest["source"]["manifest_sha256"] = "0" * 64
        else:
            bundle.manifest["files"][0]["path"] = "../credentials.env"
        (bundle.root / "manifest.json").write_bytes(canonical_json(bundle.manifest))
    with pytest.raises(DataError):
        azure._prepare(bundle.root)


def test_remote_corruption_is_not_overwritten(bundle, cloud, config, project):
    azure.publish(bundle.root, config, project_root=project)
    name = next(name for name in cloud.container.objects if name.endswith("train.parquet"))
    payload = cloud.container.objects[name]
    cloud.container.objects[name] = b"!" + payload[1:]
    cloud.client.data.create_or_update.reset_mock()
    with pytest.raises(DataError, match="checksum"):
        azure.publish(bundle.root, config, project_root=project)
    assert cloud.container.objects[name] == b"!" + payload[1:]
    cloud.client.data.create_or_update.assert_not_called()


def test_unexpected_remote_object_blocks_marker_and_registration(bundle, cloud, config, project):
    prefix = f"ml-ready/cmapss/sha256-{bundle.digest}/"
    cloud.container.objects[prefix + "extra.env"] = b"secret"
    with pytest.raises(DataError, match="unexpected objects"):
        azure.publish(bundle.root, config, project_root=project)
    assert prefix + "_SUCCESS.json" not in cloud.container.objects
    cloud.client.data.create_or_update.assert_not_called()


def test_changed_datastore_never_repointed(bundle, cloud, config, project):
    cloud.client.datastores.get.return_value.container_name = "some-other-container"
    with pytest.raises(DataError, match="Existing datastore"):
        azure.publish(bundle.root, config, project_root=project)
    cloud.blob_factory.assert_not_called()
    cloud.client.datastores.create_or_update.assert_not_called()


def test_private_container_required(bundle, cloud, config, project):
    cloud.container.public_access = "blob"
    with pytest.raises(DataError, match="private"):
        azure.publish(bundle.root, config, project_root=project)
    assert not cloud.container.objects


def test_job_contract_cpu_identity_timeout_input_output(bundle, cloud, config, project):
    add_feature(cloud, bundle)
    result = submit(bundle, config, project)
    job = cloud.client.jobs.create_or_update.call_args.args[0]
    assert azure._JOB_NAME.fullmatch(job.name)
    assert job.compute == "cpu-dev"
    assert job.experiment_name == "epm-baseline-rul"
    assert job.resources.instance_count == 1
    assert job.limits.timeout == 3600
    assert job.identity.type == "managed_identity"
    assert job.identity.client_id is None
    assert job.identity.object_id is None
    assert job.identity.resource_id is None
    assert set(job.inputs) == {"ml_ready"}
    assert job.inputs["ml_ready"].type == "uri_folder"
    assert job.inputs["ml_ready"].mode == "download"
    assert job.inputs["ml_ready"].path == f"azureml:{azure.ASSET_NAME}:{bundle.version}"
    assert job.outputs["baseline"].type == "uri_folder"
    assert job.outputs["baseline"].mode == "upload"
    assert job.outputs["baseline"].path == (
        f"azureml://datastores/workspaceblobstore/paths/baseline/{job.name}/"
    )
    assert job.environment == "azureml://registries/azureml/environments/sklearn-1.5/versions/54"
    assert result["env_ref"] == job.environment
    assert job.command == (
        "bash config/run-baseline.sh --data ${{inputs.ml_ready}} "
        "--config config/baseline.json --output ${{outputs.baseline}}"
    )
    assert job.environment_variables == {
        "PYTHONPATH": "./src",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONHASHSEED": "42",
    }
    assert job.tags["manifest_sha256"] == bundle.digest
    assert job.tags["source_manifest_sha256"] == azure.SOURCE["manifest_sha256"]
    assert Path(job.code) == project / ".azure" / "job-code" / result["codehash"]
    assert job.tags["code_sha256"] == result["codehash"]
    assert json.loads((project / ".azure" / "baseline-job.json").read_bytes()) == result
    assert config.subscription_id not in json.dumps(result)
    cloud.client.data.create_or_update.assert_not_called()
    cloud.blob_factory.assert_not_called()
    cloud.client.environments.get.assert_not_called()
    cloud.client.environments.create_or_update.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("size", "Standard_NC6"),
        ("min_instances", 1),
        ("max_instances", 2),
        ("max_instances", True),
        ("enable_node_public_ip", True),
        ("ssh_public_access_enabled", True),
        ("idle_time_before_scale_down", 600),
        ("provisioning_state", "Failed"),
        ("identity", SimpleNamespace(type="UserAssigned", principal_id="other")),
    ],
)
def test_compute_policy_blocks_submission(field, value, bundle, cloud, config, project):
    add_feature(cloud, bundle)
    setattr(cloud.client.compute.get.return_value, field, value)
    with pytest.raises(DataError, match="Compute"):
        submit(bundle, config, project)
    cloud.client.jobs.create_or_update.assert_not_called()
    assert not (project / ".azure").exists()


@pytest.mark.parametrize(
    "reference",
    [
        "azureml:sklearn-1.5:54",
        "azureml://registries/another/environments/sklearn-1.5/versions/54",
        "azureml://registries/azureml/environments/sklearn-1.5/versions/55",
    ],
)
def test_exact_public_registry_reference_required(
    reference, bundle, cloud, config, project, monkeypatch
):
    monkeypatch.setattr(azure, "ENVIRONMENT_REF", reference)
    with pytest.raises(DataError, match="public curated"):
        submit(bundle, config, project)
    cloud.credential_factory.assert_not_called()
    cloud.client.environments.get.assert_not_called()
    cloud.client.jobs.create_or_update.assert_not_called()


def test_fullhash_checked_before_submit(bundle, cloud, config, project):
    asset = add_feature(cloud, bundle)
    asset.tags["manifest_sha256"] = "f" * 64
    with pytest.raises(DataError, match="conflicting"):
        submit(bundle, config, project)
    cloud.client.jobs.create_or_update.assert_not_called()


@pytest.mark.parametrize(
    "approve,version,digest,environment",
    [
        (False, "valid", "valid", "54"),
        (True, "wrong", "valid", "54"),
        (True, "valid", "short", "54"),
        (True, "valid", "valid", "latest"),
        (True, "valid", "valid", "54;echo SECRET"),
        (True, "valid", "valid", "@latest"),
        (True, "valid", "valid", "1"),
        (True, "valid", "valid", "55"),
        (True, "valid", "valid", 54),
    ],
)
def test_approval_and_selectors_checked_before_auth(
    approve, version, digest, environment, bundle, cloud, config, project
):
    with pytest.raises(DataError):
        azure.submit(
            bundle.version if version == "valid" else version,
            bundle.digest if digest == "valid" else digest,
            environment,
            config,
            approve_costs=approve,
            project_root=project,
        )
    cloud.credential_factory.assert_not_called()


def test_uncertain_submission_has_recovery_receipt_and_never_retries(
    bundle, cloud, config, project
):
    add_feature(cloud, bundle)
    error = HttpResponseError(SECRET)
    error.status_code = 503
    cloud.client.jobs.create_or_update.side_effect = error
    with pytest.raises(DataError, match="HTTP 503") as caught:
        submit(bundle, config, project)
    assert SECRET not in str(caught.value)
    cloud.client.jobs.create_or_update.assert_called_once()
    receipt = json.loads((project / ".azure" / "baseline-job.json").read_bytes())
    assert receipt["status"] == "SubmissionUnknown"
    assert azure._JOB_NAME.fullmatch(receipt["job_name"])


def test_real_sdk_provisioning_enum_is_accepted(cloud, config):
    from enum import Enum

    class State(str, Enum):  # noqa: UP042 - reproduce the Azure SDK's legacy string-enum behavior.
        SUCCEEDED = "Succeeded"

    compute = cloud.client.compute.get.return_value
    compute.provisioning_state = State.SUCCEEDED
    azure._check_compute(compute, config)


def test_stage_rejects_windows_line_endings_in_linux_bootstrap(project):
    script = project / "scripts" / "run-baseline.sh"
    script.write_bytes(b"#!/bin/bash\r\nset -euo pipefail\r\n")
    with pytest.raises(DataError, match="LF"):
        azure._stage_code(project)


def test_stage_exact_allowlist_hash_and_no_secrets(project):
    for name in [
        ".azure/identity.json",
        ".env",
        "data/raw/private.txt",
        "keys/access.key",
        "src/epm_platform/data/publishing.py",
        "src/epm_platform/client.py",
        "src/epm_platform/baseline/azure.py",
        "src/epm_platform/features/unreviewed.py",
        "src/epm_platform/features/__pycache__/cache.pyc",
        "environments/baseline/Dockerfile",
        "environments/baseline/private.txt",
        "scripts/private.sh",
        "config/runtime-requirements.txt",
        "config/run-baseline.sh",
    ]:
        path = project.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SECRET)
    stage, digest = azure._stage_code(project)
    assert stage != project
    assert {
        path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
    } == set(azure._CODE_PATHS)
    assert not any(
        SECRET.encode() in path.read_bytes() for path in stage.rglob("*") if path.is_file()
    )
    assert azure._stage_code(project) == (stage, digest)
    (project / "config" / "baseline.json").write_bytes(b"{}")
    other_stage, other_digest = azure._stage_code(project)
    assert other_digest != digest
    assert other_stage != stage


@pytest.mark.parametrize(
    "destination,source",
    [
        ("config/runtime-requirements.txt", "environments/baseline/requirements.txt"),
        ("config/run-baseline.sh", "scripts/run-baseline.sh"),
    ],
)
def test_explicit_bootstrap_copies_are_hashed(destination, source, project):
    stage, digest = azure._stage_code(project)
    original = project.joinpath(*source.split("/"))
    copied = stage.joinpath(*destination.split("/"))
    assert copied.read_bytes() == original.read_bytes()
    assert not stage.joinpath(*source.split("/")).exists()
    original.write_bytes(original.read_bytes() + b"\n# changed bootstrap input\n")
    other_stage, other_digest = azure._stage_code(project)
    assert other_digest != digest
    assert other_stage.joinpath(*destination.split("/")).read_bytes() == original.read_bytes()


@pytest.mark.parametrize(
    "source",
    [
        "environments/baseline/requirements.txt",
        "scripts/run-baseline.sh",
    ],
)
def test_missing_bootstrap_blocks_submission(source, bundle, cloud, config, project):
    add_feature(cloud, bundle)
    project.joinpath(*source.split("/")).unlink()
    with pytest.raises(DataError, match="allowlisted code file is missing"):
        submit(bundle, config, project)
    cloud.client.jobs.create_or_update.assert_not_called()


@pytest.mark.parametrize("mutation", ["extra", "changed", "missing"])
def test_tampered_staging_fails_closed(mutation, project):
    stage, _ = azure._stage_code(project)
    if mutation == "extra":
        (stage / "private.env").write_text(SECRET)
    elif mutation == "changed":
        (stage / "config" / "baseline.json").write_bytes(b"{}")
    else:
        (stage / "config" / "baseline.json").unlink()
    with pytest.raises(DataError, match="staging"):
        azure._stage_code(project)


def test_stage_rejects_oversized_file(project, monkeypatch):
    monkeypatch.setattr(azure, "_MAX_CODE_FILE", 8)
    with pytest.raises(DataError, match="too large"):
        azure._stage_code(project)


def test_stage_rejects_link_before_read(project, monkeypatch):
    selected = project / "config" / "baseline.json"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == selected or original(self))
    with pytest.raises(DataError, match="Links"):
        azure._stage_code(project)


def test_staged_modules_import_without_cloud_sdk(project):
    stage, _ = azure._stage_code(project)
    env = dict(os.environ, PYTHONPATH=str(stage / "src"), PYTHONDONTWRITEBYTECODE="1")
    script = (
        "import sys; from epm_platform.features.pipeline import verify_features; "
        "from epm_platform.baseline import training; "
        "assert not any(name == 'azure' or name.startswith('azure.') for name in sys.modules); "
        "assert 'job-code' in training.__file__"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=stage,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_status_readonly_and_safe_failed_code(cloud, config):
    cloud.client.jobs.get.return_value = SimpleNamespace(name=JOB, status="Failed", error=SECRET)
    assert azure.status(JOB, config) == {
        "job_name": JOB,
        "status": "Failed",
        "error_code": "job_failed",
    }
    cloud.client.jobs.create_or_update.assert_not_called()
    cloud.client.jobs.download.assert_not_called()
    cloud.client.data.get.assert_not_called()


@pytest.mark.parametrize("job_status", ["Running", "Failed", "Canceled", SECRET])
def test_download_requires_completed(job_status, cloud, config, workspace):
    cloud.client.jobs.get.return_value.status = job_status
    with pytest.raises(DataError, match="Only Completed"):
        azure.download(JOB, workspace / "output", config)
    cloud.client.jobs.download.assert_not_called()


class OutputBlob(Blob):
    def get_blob_properties(self, **kwargs):
        if kwargs.get("etag") and self.container.fail_condition:
            error = ResourceModifiedError(SECRET)
            error.status_code = 412
            raise error
        return super().get_blob_properties(**kwargs)

    def download_blob(self, **kwargs):
        super().download_blob(**kwargs)
        self.container.events.append(("download", self.name))
        payload = self.container.objects[self.name]
        if self.container.stream_mode == "short":
            payload = payload[:-1]
        elif self.container.stream_mode == "long":
            payload += b"!"
        return SimpleNamespace(chunks=lambda: [payload[:17], payload[17:]])


class OutputContainer(Container):
    def __init__(self, events):
        super().__init__(events)
        self.fail_condition = False
        self.stream_mode = None
        self.change_after_read = False
        self.list_count = 0

    def get_blob_client(self, name):
        assert name.startswith(f"baseline/{JOB}/")
        return OutputBlob(self, name)

    def list_blobs(self, *, name_starts_with, **kwargs):
        assert name_starts_with == f"baseline/{JOB}/"
        self.list_count += 1
        etag = "revision-2" if self.change_after_read and self.list_count > 1 else "revision-1"
        return [
            SimpleNamespace(name=name, size=len(payload), etag=etag)
            for name, payload in self.objects.items()
            if name.startswith(name_starts_with)
        ]


@pytest.fixture
def output_cloud(cloud):
    container = OutputContainer(cloud.events)
    content = {
        name: (name + "\n").encode()
        for name in (
            "evaluation.md",
            "feature-importance.json",
            "metrics.json",
            "model.json",
            "predictions_test.parquet",
            "predictions_validation.parquet",
            "run-metadata.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "xgboost-rul-baseline",
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(content.items())
        ],
        "provenance": {"ml_ready_manifest_sha256": "a" * 64},
        "completion_marker": "artifact-manifest.json",
    }
    content["artifact-manifest.json"] = json.dumps(manifest, indent=2).encode()
    container.objects = {f"baseline/{JOB}/" + name: payload for name, payload in content.items()}
    cloud.service.get_container_client.return_value = container
    cloud.client.jobs.get.return_value.outputs = {
        "baseline": SimpleNamespace(
            type="uri_folder",
            path=(f"azureml://datastores/workspaceblobstore/paths/baseline/{JOB}/"),
        ),
    }
    cloud.client.datastores.get.return_value = AzureBlobDatastore(
        name="workspaceblobstore",
        account_name="epmstorage000",
        container_name="azureml-blobstore-fixture",
        protocol="https",
        endpoint="core.windows.net",
    )
    cloud.client.datastores.get_default.return_value = copy.deepcopy(
        cloud.client.datastores.get.return_value
    )
    cloud.client.jobs.download.return_value = None
    return SimpleNamespace(cloud=cloud, container=container, content=content, manifest=manifest)


def test_download_direct_verifies_all_hashes_despite_sdk_noop(output_cloud, config, workspace):
    cloud = output_cloud.cloud
    destination = workspace / "output"
    destination.mkdir()
    result = azure.download(JOB, destination, config)
    assert result == {
        "job_name": JOB,
        "status": "Completed",
        "output_name": "baseline",
        "downloaded": True,
        "objects_verified": 8,
        "artifact_manifest_sha256": hashlib.sha256(
            output_cloud.content["artifact-manifest.json"]
        ).hexdigest(),
    }
    assert {path.name for path in destination.iterdir()} == set(output_cloud.content)
    for name, payload in output_cloud.content.items():
        assert (destination / name).read_bytes() == payload
    for entry in output_cloud.manifest["files"]:
        assert fingerprint(destination / entry["path"]) == {
            key: entry[key] for key in ("sha256", "size_bytes")
        }
    cloud.client.jobs.download.assert_not_called()
    cloud.client.datastores.get.assert_called_once_with("workspaceblobstore", include_secrets=False)
    cloud.client.datastores.get_default.assert_called_once_with(include_secrets=False)
    cloud.service.get_container_client.assert_called_once_with("azureml-blobstore-fixture")
    cloud.client.jobs.create_or_update.assert_not_called()
    cloud.client.data.get.assert_not_called()
    cloud.service.close.assert_called_once()
    with pytest.raises(DataError, match="new or empty"):
        azure.download(JOB, destination, config)


@pytest.mark.parametrize("mutation", ["empty", "missing", "extra"])
def test_download_never_succeeds_on_empty_or_wrong_inventory(
    mutation, output_cloud, config, workspace
):
    if mutation == "empty":
        output_cloud.container.objects.clear()
    elif mutation == "missing":
        del output_cloud.container.objects[f"baseline/{JOB}/model.json"]
    else:
        output_cloud.container.objects[f"baseline/{JOB}/private.env"] = b"secret"
    destination = workspace / "output"
    destination.mkdir()
    with pytest.raises(DataError, match="inventory"):
        azure.download(JOB, destination, config)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "path",
    [
        f"azureml://datastores/workspaceblobstore/paths/baseline/{JOB}",
        "azureml://datastores/workspaceblobstore/paths/baseline/epm-baseline-ffffffffffff/",
        f"azureml://datastores/another/paths/baseline/{JOB}/",
        f"azureml://datastores/workspaceblobstore/paths/baseline/{JOB}/../",
        "https://example.invalid/output?sig=SECRET",
    ],
)
def test_download_rejects_malformed_or_wrong_job_output_path(path, output_cloud, config, workspace):
    output_cloud.cloud.client.jobs.get.return_value.outputs["baseline"].path = path
    with pytest.raises(DataError, match="exact approved job path"):
        azure.download(JOB, workspace / "output", config)
    output_cloud.cloud.blob_factory.assert_not_called()


def test_download_requires_uri_folder(output_cloud, config, workspace):
    output_cloud.cloud.client.jobs.get.return_value.outputs["baseline"].type = "uri_file"
    with pytest.raises(DataError, match="exact approved job path"):
        azure.download(JOB, workspace / "output", config)
    output_cloud.cloud.blob_factory.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_name", "otherstorage"),
        ("container_name", "other-container"),
        ("name", "anotherstore"),
        ("protocol", "http"),
        ("endpoint", "example.invalid"),
    ],
)
def test_download_requires_existing_workspace_storage(
    field, value, output_cloud, config, workspace
):
    setattr(output_cloud.cloud.client.datastores.get.return_value, field, value)
    with pytest.raises(DataError, match="workspace"):
        azure.download(JOB, workspace / "output", config)
    output_cloud.cloud.blob_factory.assert_not_called()


@pytest.mark.parametrize("mutation", ["hash", "size", "path", "duplicate", "schema", "json"])
def test_download_corrupt_manifest_never_finalizes(mutation, output_cloud, config, workspace):
    manifest = output_cloud.manifest
    if mutation == "hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        manifest["files"][0]["size_bytes"] += 1
    elif mutation == "path":
        manifest["files"][0]["path"] = "../private.env"
    elif mutation == "duplicate":
        manifest["files"][0] = manifest["files"][1]
    elif mutation == "schema":
        manifest["schema_version"] = 2
    payload = canonical_json(manifest) if mutation != "json" else b"{bad json}"
    output_cloud.container.objects[f"baseline/{JOB}/artifact-manifest.json"] = payload
    destination = workspace / "output"
    with pytest.raises(DataError):
        azure.download(JOB, destination, config)
    assert list(destination.iterdir()) == []


def test_download_corrupt_artifact_never_finalizes(output_cloud, config, workspace):
    key = f"baseline/{JOB}/model.json"
    original = output_cloud.container.objects[key]
    output_cloud.container.objects[key] = b"!" + original[1:]
    destination = workspace / "output"
    with pytest.raises(DataError, match="checksum"):
        azure.download(JOB, destination, config)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("mode", ["short", "long", "etag", "final_etag"])
def test_download_incomplete_or_changed_stream_never_finalizes(
    mode, output_cloud, config, workspace
):
    if mode == "etag":
        output_cloud.container.fail_condition = True
    elif mode == "final_etag":
        output_cloud.container.change_after_read = True
    else:
        output_cloud.container.stream_mode = mode
    destination = workspace / "output"
    with pytest.raises(DataError) as caught:
        azure.download(JOB, destination, config)
    assert SECRET not in str(caught.value)
    assert list(destination.iterdir()) == []


def test_download_finalization_is_exclusive_and_cleans_owned_files(
    output_cloud, config, workspace, monkeypatch
):
    destination = workspace / "output"
    original = Path.open
    seen = []

    def exclusive(path, mode="r", *args, **kwargs):
        if path.parent == destination and "x" in mode:
            seen.append(path.name)
            if path.name == "artifact-manifest.json":
                with original(path, "wb") as stream:
                    stream.write(b"concurrent writer")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", exclusive)
    with pytest.raises(DataError):
        azure.download(JOB, destination, config)
    assert seen[-1] == "artifact-manifest.json"
    assert {path.name for path in destination.iterdir()} == {"artifact-manifest.json"}
    assert (destination / "artifact-manifest.json").read_bytes() == b"concurrent writer"


@pytest.mark.parametrize("operation", ["publish", "submit", "status", "download"])
def test_cli_dispatch(operation, monkeypatch, config, bundle, workspace, capsys):
    monkeypatch.setattr(azure, "load_config", lambda: config)
    handler = Mock(return_value={"operation": operation, "ok": True})
    monkeypatch.setattr(azure, operation, handler)
    options = {
        "publish": ["--data", str(bundle.root)],
        "submit": [
            "--asset-version",
            bundle.version,
            "--manifest-sha256",
            bundle.digest,
            "--environment-version",
            "54",
            "--approve-costs",
        ],
        "status": ["--job-name", JOB],
        "download": ["--job-name", JOB, "--destination", str(workspace / "download")],
    }
    assert azure.main([operation, *options[operation]]) == 0
    assert json.loads(capsys.readouterr().out) == {"operation": operation, "ok": True}
    handler.assert_called_once()
    if operation == "submit":
        assert handler.call_args.kwargs["approve_costs"] is True


def test_cli_defaults_to_approved_public_version(monkeypatch, capsys, config, bundle):
    monkeypatch.setattr(azure, "load_config", lambda: config)
    handler = Mock(return_value={"status": "Queued"})
    monkeypatch.setattr(azure, "submit", handler)
    assert (
        azure.main(
            [
                "submit",
                "--asset-version",
                bundle.version,
                "--manifest-sha256",
                bundle.digest,
                "--approve-costs",
            ]
        )
        == 0
    )
    assert handler.call_args.args[2] == "54"
    assert json.loads(capsys.readouterr().out) == {"status": "Queued"}


def test_cli_sanitizes_sdk_logs_exceptions_and_invalid_configuration(monkeypatch, capsys, config):
    def failing_status(*args):
        print(SECRET)
        print(SECRET, file=sys.stderr)
        raise HttpResponseError(SECRET)

    monkeypatch.setattr(azure, "load_config", lambda: config)
    monkeypatch.setattr(azure, "status", failing_status)
    assert azure.main(["status", "--job-name", JOB]) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["code"] == "baseline_operation_failed"


def test_unrecognized_job_name_never_reaches_cloud(cloud, config):
    with pytest.raises(DataError, match="generated"):
        azure.status("another-job?sig=SECRET", config)
    cloud.credential_factory.assert_not_called()


def test_full_cloud_asset_path_is_checked_exactly(bundle, cloud, config):
    asset = add_feature(cloud, bundle)
    asset.path = (
        f"azureml://subscriptions/{config.subscription_id}/resourcegroups/{config.resource_group}"
        f"/workspaces/{config.workspace_name}/datastores/{publishing.DATASTORE_NAME}"
        f"/paths/ml-ready/cmapss/sha256-{bundle.digest}/"
    )
    azure._check_asset(asset, bundle.version, bundle.digest, config)
    changed = copy.deepcopy(asset)
    changed.path = asset.path.replace(config.workspace_name, "another-workspace")
    with pytest.raises(DataError, match="conflicting"):
        azure._check_asset(changed, bundle.version, bundle.digest, config)
