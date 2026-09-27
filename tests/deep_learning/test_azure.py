"""Offline contract tests: no cloud calls, installs or model loading."""

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from azure.ai.ml.entities import AzureBlobDatastore, Data, ManagedIdentityConfiguration
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError

from epm_platform.data import publishing
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint
from epm_platform.deep_learning import azure

ROOT = Path(__file__).absolute().parents[2]
JOB = "epm-pytorch-012345abcdef"
SECRET = "https://private.invalid/?sig=credential-sensitive"


@pytest.fixture
def workspace():
    path = ROOT / (".pytorch-azure-test-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def project(workspace):
    root = workspace / "project"
    for source in azure._CODE_PATHS.values():
        path = root.joinpath(*source.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if source == "config/pytorch.json":
            shutil.copyfile(ROOT / "config" / "pytorch.json", path)
        elif source == "config/baseline-reference.json":
            path.write_bytes(canonical_json({"ml_ready_manifest_sha256": azure.MANIFEST_SHA256}))
        elif source.endswith(".sh"):
            path.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
        else:
            path.write_bytes(b"# reviewed fixture\n")
    return root


class Blob:
    def __init__(self, container, name):
        self.container, self.name = container, name

    def get_blob_properties(self, **kwargs):
        if "etag" in kwargs:
            assert kwargs["etag"] == '"revision-1"'
            assert kwargs["match_condition"] == MatchConditions.IfNotModified
        if self.container.changed and "etag" in kwargs:
            raise ResourceModifiedError(SECRET)
        return SimpleNamespace(size=len(self.container.objects[self.name]), etag='"revision-1"')

    def download_blob(self, **kwargs):
        assert kwargs["etag"] == '"revision-1"'
        assert kwargs["match_condition"] == MatchConditions.IfNotModified
        assert kwargs["max_concurrency"] == 1
        assert kwargs["logging_enable"] is False
        self.container.downloads.append(self.name)
        data = self.container.objects[self.name]
        if self.container.stream_delta:
            data = data[:-1] if self.container.stream_delta < 0 else data + b"x"
        return SimpleNamespace(chunks=lambda: [data])


class Container:
    def __init__(self):
        self.objects, self.downloads = {}, []
        self.public_access, self.changed, self.stream_delta = None, False, 0
        self.list_count, self.revision_on_final = 0, False

    def get_container_properties(self, **kwargs):
        return SimpleNamespace(public_access=self.public_access)

    def get_blob_client(self, name):
        assert name.startswith(f"pytorch/{JOB}/")
        return Blob(self, name)

    def list_blobs(self, *, name_starts_with, **kwargs):
        self.list_count += 1
        etag = "revision-2" if self.revision_on_final and self.list_count > 1 else "revision-1"
        return [
            SimpleNamespace(name=name, size=len(data), etag=etag)
            for name, data in self.objects.items()
            if name.startswith(name_starts_with)
        ]


@pytest.fixture
def cloud(config, fake_credential, monkeypatch):
    client = Mock()
    source = Data(
        name=azure.safe.SOURCE["asset_name"],
        version=azure.safe.SOURCE["asset_version"],
        type="uri_folder",
        path=publishing._asset_path("sha256-" + azure.safe.SOURCE["manifest_sha256"]),
        tags={"manifest_sha256": azure.safe.SOURCE["manifest_sha256"]},
    )
    feature = Data(
        name=azure.ASSET_NAME,
        version=azure.ASSET_VERSION,
        type="uri_folder",
        path=azure.safe._asset_path(azure.MANIFEST_SHA256),
        tags=azure.safe._tags(azure.MANIFEST_SHA256),
    )
    assets = {source.name: source, feature.name: feature}
    client.data.get.side_effect = lambda *, name, version: assets[name]
    client.workspaces.get.return_value = SimpleNamespace(
        storage_account=(
            f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}"
            "/providers/Microsoft.Storage/storageAccounts/epmstorage000"
        )
    )
    output_store = AzureBlobDatastore(
        name="workspaceblobstore",
        account_name="epmstorage000",
        container_name="azureml-output",
        protocol="https",
        endpoint="core.windows.net",
    )
    curated_store = AzureBlobDatastore(
        name=publishing.DATASTORE_NAME,
        account_name="epmstorage000",
        container_name=publishing.CURATED_CONTAINER,
        protocol="https",
        endpoint="core.windows.net",
    )
    client.datastores.get.side_effect = lambda name, **kwargs: (
        output_store if name == "workspaceblobstore" else curated_store
    )
    client.datastores.get_default.return_value = output_store
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
    client.jobs.create_or_update.side_effect = lambda job: SimpleNamespace(
        name=job.name, status="Queued"
    )
    job = SimpleNamespace(
        name=JOB,
        status="Completed",
        outputs={
            "pytorch": SimpleNamespace(
                type="uri_folder",
                path=f"azureml://datastores/workspaceblobstore/paths/pytorch/{JOB}/",
            )
        },
    )
    client.jobs.get.return_value = job

    @contextmanager
    def local_client(config):
        yield client, fake_credential

    monkeypatch.setattr(azure.safe, "_client", local_client)
    container = Container()
    service = Mock()
    service.get_container_client.return_value = container
    factory = Mock(return_value=service)
    monkeypatch.setattr(azure, "BlobServiceClient", factory)
    return SimpleNamespace(
        client=client,
        feature=feature,
        source=source,
        container=container,
        service=service,
        factory=factory,
        output_store=output_store,
        job=job,
    )


def submit(project, config, **kwargs):
    options = {
        "asset_version": azure.ASSET_VERSION,
        "manifest_sha256": azure.MANIFEST_SHA256,
        "environment_version": azure.ENVIRONMENT_VERSION,
        "config": config,
        "approve_costs": True,
        "project_root": project,
    }
    return azure.submit(**(options | kwargs))


def output_bundle(cloud):
    prefix = f"pytorch/{JOB}/"
    for name in azure._CONTENT:
        cloud.container.objects[prefix + name] = (name + "\n").encode()
    files = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name in sorted(azure._CONTENT)
        for data in [cloud.container.objects[prefix + name]]
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": "pytorch-mlp-rul",
        "completion_marker": "artifact-manifest.json",
        "files": files,
        "model_sha256": hashlib.sha256(cloud.container.objects[prefix + "model.pt"]).hexdigest(),
        "provenance": {
            "ml_ready_manifest_sha256": azure.MANIFEST_SHA256,
            "source": azure.safe.SOURCE.copy(),
        },
    }
    cloud.container.objects[prefix + "artifact-manifest.json"] = canonical_json(manifest)
    return manifest


def test_submit_bounded_cpu_identity_pinned_inputs_and_no_registry(project, cloud, config):
    receipt_path = project / ".azure" / "pytorch-job.json"

    def create(job):
        assert json.loads(receipt_path.read_bytes())["status"] == "SubmissionPending"
        assert job.name == json.loads(receipt_path.read_bytes())["job_name"]
        return SimpleNamespace(name=job.name, status="Queued")

    cloud.client.jobs.create_or_update.side_effect = create
    result = submit(project, config)
    job = cloud.client.jobs.create_or_update.call_args.args[0]
    assert azure._JOB_NAME.fullmatch(job.name)
    assert job.experiment_name == "epm-pytorch-rul"
    assert job.compute == "cpu-dev"
    assert job.resources.instance_count == 1
    assert isinstance(job.identity, ManagedIdentityConfiguration)
    assert job.limits.timeout == 3600
    assert job.environment == azure.ENVIRONMENT_REF
    assert job.inputs["ml_ready"].path == f"azureml:{azure.ASSET_NAME}:{azure.ASSET_VERSION}"
    assert job.inputs["ml_ready"].mode == "download"
    assert set(job.outputs) == {"pytorch"}
    assert job.outputs["pytorch"].mode == "upload"
    assert job.outputs["pytorch"].path.endswith(f"/pytorch/{job.name}/")
    assert job.command == (
        "bash config/run-pytorch.sh --data ${{inputs.ml_ready}} --config config/pytorch.json "
        "--baseline-reference config/baseline-reference.json --output ${{outputs.pytorch}}"
    )
    assert job.environment_variables == {
        "PYTHONPATH": "./src",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONHASHSEED": "42",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
        "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
        "MLFLOW_ENABLE_TELEMETRY": "false",
    }
    assert json.loads(receipt_path.read_bytes()) == result
    assert SECRET not in receipt_path.read_text()
    for operation in (
        cloud.client.compute.create_or_update,
        cloud.client.data.create_or_update,
        cloud.client.environments.get,
        cloud.client.environments.create_or_update,
        cloud.client.models.create_or_update,
        cloud.client.jobs.cancel,
        cloud.client.datastores.create_or_update,
    ):
        operation.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"approve_costs": False},
        {"asset_version": "latest"},
        {"manifest_sha256": "0" * 64},
        {"environment_version": "latest"},
        {"approve_costs": 1},
    ],
)
def test_submit_rejects_unapproved_selectors_before_cloud(change, project, cloud, config):
    with pytest.raises(DataError):
        submit(project, config, **change)
    cloud.client.jobs.create_or_update.assert_not_called()
    cloud.client.data.get.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_instances", 2),
        ("min_instances", 1),
        ("size", "Standard_NC6"),
        ("enable_node_public_ip", True),
        ("ssh_public_access_enabled", True),
        ("identity", SimpleNamespace(type="UserAssigned", principal_id="other")),
    ],
)
def test_submit_rejects_compute_drift(field, value, project, cloud, config):
    setattr(cloud.client.compute.get.return_value, field, value)
    with pytest.raises(DataError):
        submit(project, config)
    cloud.client.jobs.create_or_update.assert_not_called()


def test_submit_rejects_source_asset_drift(project, cloud, config):
    cloud.feature.tags["source_manifest_sha256"] = "0" * 64
    with pytest.raises(DataError):
        submit(project, config)
    cloud.client.jobs.create_or_update.assert_not_called()


def test_uncertain_submit_preserves_name_never_retries(project, cloud, config):
    cloud.client.jobs.create_or_update.side_effect = RuntimeError(SECRET)
    with pytest.raises(DataError) as error:
        submit(project, config)
    assert SECRET not in str(error.value)
    receipt = json.loads((project / ".azure" / "pytorch-job.json").read_bytes())
    assert receipt["status"] == "SubmissionUnknown"
    assert azure._JOB_NAME.fullmatch(receipt["job_name"])
    cloud.client.jobs.create_or_update.assert_called_once()


def test_staging_exact_allowlist_hash_reuse_and_secret_exclusion(project):
    for path in (project / ".env", project / "data" / "secret", project / ".azure" / "secret"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SECRET)
    stage, digest = azure._stage_code(project)
    assert stage == project / ".azure" / "pytorch-code" / digest
    assert azure._stage_code(project) == (stage, digest)
    assert {
        path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
    } == set(azure._CODE_PATHS)
    assert "src/epm_platform/baseline/training.py" in azure._CODE_PATHS
    assert "src/epm_platform/deep_learning/azure.py" not in azure._CODE_PATHS
    assert all(
        SECRET.encode() not in path.read_bytes() for path in stage.rglob("*") if path.is_file()
    )
    (stage / ".env").write_text(SECRET)
    with pytest.raises(DataError):
        azure._stage_code(project)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda"),
        ("threads", 3),
        ("threads", True),
        ("max_epochs", 101),
        ("max_epochs", 0),
        ("patience", 13),
        ("seed", 43),
    ],
)
def test_staging_enforces_cpu_budget(field, value, project):
    path = project / "config" / "pytorch.json"
    recipe = json.loads(path.read_bytes())
    recipe[field] = value
    path.write_bytes(canonical_json(recipe))
    with pytest.raises(DataError, match="budget"):
        azure._stage_code(project)


def test_staging_requires_lf_and_detects_mutation(project):
    script = project / "scripts" / "run-pytorch.sh"
    script.write_bytes(b"#!/bin/bash\r\nexit 0\r\n")
    with pytest.raises(DataError, match="LF"):
        azure._stage_code(project)
    script.write_bytes(b"#!/bin/bash\nexit 0\n")
    stage, _ = azure._stage_code(project)
    (stage / "config" / "run-pytorch.sh").write_bytes(b"changed\n")
    with pytest.raises(DataError, match="changed"):
        azure._stage_code(project)


def test_download_direct_blob_exact_eleven_and_manifest_last(workspace, cloud, config, monkeypatch):
    output_bundle(cloud)
    destination = workspace / "output"
    events = []
    original = Path.open

    def record(path, mode="r", *args, **kwargs):
        if mode == "xb" and path.parent == destination:
            events.append(path.name)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record)
    result = azure.download(JOB, destination, config)
    assert result["objects_verified"] == 11
    assert result["output_name"] == "pytorch"
    assert {path.name for path in destination.iterdir()} == azure._CONTENT | {azure._MARKER}
    assert events[-1] == azure._MARKER
    assert fingerprint(destination / azure._MARKER)["sha256"] == result["artifact_manifest_sha256"]
    assert cloud.container.downloads[0].endswith("/artifact-manifest.json")
    cloud.client.jobs.download.assert_not_called()
    cloud.client.jobs.cancel.assert_not_called()
    cloud.service.close.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "traversal",
        "checksum",
        "manifest_size",
        "duplicate",
        "source",
        "model_hash",
        "etag",
        "final_etag",
        "short",
        "long",
        "public",
        "store",
        "container",
    ],
)
def test_download_integrity_failures_publish_nothing(case, workspace, cloud, config):
    manifest = output_bundle(cloud)
    prefix = f"pytorch/{JOB}/"
    objects = cloud.container.objects
    if case == "missing":
        del objects[prefix + "model.pt"]
    elif case == "extra":
        objects[prefix + "secret.txt"] = b"unexpected"
    elif case == "traversal":
        manifest["files"][0]["path"] = "../secret.txt"
    elif case == "checksum":
        objects[prefix + "model.pt"] = b"X" * len(objects[prefix + "model.pt"])
    elif case == "manifest_size":
        manifest["files"][0]["size_bytes"] += 1
    elif case == "duplicate":
        manifest["files"][1] = manifest["files"][0]
    elif case == "source":
        manifest["provenance"]["ml_ready_manifest_sha256"] = "0" * 64
    elif case == "model_hash":
        manifest["model_sha256"] = "0" * 64
    elif case == "etag":
        cloud.container.changed = True
    elif case == "final_etag":
        cloud.container.revision_on_final = True
    elif case in {"short", "long"}:
        cloud.container.stream_delta = -1 if case == "short" else 1
    elif case == "public":
        cloud.container.public_access = "blob"
    elif case == "store":
        cloud.output_store.account_name = "otheraccount"
    elif case == "container":
        cloud.output_store.container_name = "../bad"
    objects[prefix + azure._MARKER] = canonical_json(manifest)
    destination = workspace / "download"
    with pytest.raises(DataError) as error:
        azure.download(JOB, destination, config)
    assert SECRET not in str(error.value)
    assert not destination.exists() or not any(destination.iterdir())


@pytest.mark.parametrize(
    "uri",
    [
        f"azureml://datastores/workspaceblobstore/paths/baseline/{JOB}/",
        f"azureml://datastores/workspaceblobstore/paths/pytorch/{JOB}/../other/",
        f"azureml://datastores/workspaceblobstore/paths/pytorch/{JOB}/?sig=secret",
        "https://arbitrary.invalid/output",
        "azureml://datastores/other/paths/pytorch/",
    ],
)
def test_download_rejects_unapproved_output_uri(uri, workspace, cloud, config):
    cloud.job.outputs["pytorch"].path = uri
    with pytest.raises(DataError, match="path"):
        azure.download(JOB, workspace / "output", config)
    cloud.factory.assert_not_called()


def test_download_requires_completion_empty_destination_and_safe_name(workspace, cloud, config):
    cloud.job.status = "Running"
    with pytest.raises(DataError, match="Completed"):
        azure.download(JOB, workspace / "output", config)
    with pytest.raises(DataError, match="job name"):
        azure.status("epm-baseline-012345abcdef", config)
    destination = workspace / "existing"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    with pytest.raises(DataError, match="empty"):
        azure.download(JOB, destination, config)
    assert (destination / "keep").read_text() == "keep"
    cloud.factory.assert_not_called()


def test_status_and_cli_never_expose_service_errors(cloud, config, monkeypatch, capsys):
    cloud.job.status, cloud.job.error = "Failed", SECRET
    assert azure.status(JOB, config) == {
        "job_name": JOB,
        "status": "Failed",
        "error_code": "job_failed",
    }
    monkeypatch.setattr(azure, "load_config", lambda: config)
    cloud.client.jobs.get.side_effect = RuntimeError(SECRET)
    assert azure.main(["status", "--job-name", JOB]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert json.loads(captured.out)["code"] == "pytorch_operation_failed"
