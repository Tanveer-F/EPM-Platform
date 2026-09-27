"""Offline publication tests: real deterministic bundles and in-memory Azure clients."""

import base64
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from azure.ai.ml.entities import (
    AccountKeyConfiguration,
    AzureBlobDatastore,
    Data,
    NoneCredentialConfiguration,
    SasTokenConfiguration,
)
from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)

from epm_platform.config import AzureConfig
from epm_platform.data import publishing
from epm_platform.data.curation import curate
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint
from epm_platform.data.spec import SUBSETS, SourceSpec

_SECRET = "SENSITIVE-BODY-https://example.invalid/?sig=secret-token"


def _http(error_type, status):
    error = error_type(_SECRET)
    error.status_code = status
    return error


@dataclass
class _Object:
    payload: bytes
    metadata: dict
    revision: int = 1

    @property
    def etag(self):
        return f'"revision-{self.revision}"'


class _Blob:
    def __init__(self, storage, container, name):
        self.storage, self.container, self.name = storage, container, name

    def _object(self):
        try:
            return self.storage.objects[self.container][self.name]
        except KeyError:
            raise _http(ResourceNotFoundError, 404) from None

    def _condition(self, kwargs, item):
        if "etag" in kwargs:
            assert kwargs["match_condition"] == MatchConditions.IfNotModified
            if kwargs["etag"] != item.etag:
                raise _http(ResourceModifiedError, 412)

    def get_blob_properties(self, **kwargs):
        self.storage.check("properties")
        item = self._object()
        self._condition(kwargs, item)
        self.storage.events.append(("properties", self.container, self.name))
        return SimpleNamespace(size=len(item.payload), etag=item.etag, metadata=item.metadata)

    def upload_blob(self, stream, **kwargs):
        self.storage.check("upload")
        assert kwargs["overwrite"] is False
        assert kwargs["blob_type"] == "BlockBlob"
        assert kwargs["max_concurrency"] == 1
        assert kwargs["logging_enable"] is False
        assert stream.mode == "rb"
        objects = self.storage.objects[self.container]
        self.storage.events.append(("upload", self.container, self.name))
        payload = b"".join(iter(lambda: stream.read(4096), b""))
        assert len(payload) == kwargs["length"]
        assert hashlib.sha256(payload).hexdigest() == kwargs["metadata"]["sha256"]
        if self.name in self.storage.races:
            winner = self.storage.races.pop(self.name)
            objects[self.name] = _Object(payload if winner is None else winner, kwargs["metadata"])
        if self.name in objects:
            raise _http(ResourceExistsError, 409)
        if self.storage.corrupt_upload:
            payload = b"!" + payload[1:]
        objects[self.name] = _Object(payload, kwargs["metadata"])

    def download_blob(self, **kwargs):
        self.storage.check("download")
        item = self._object()
        assert kwargs["etag"] == item.etag
        self._condition(kwargs, item)
        assert kwargs["match_condition"] == MatchConditions.IfNotModified
        assert kwargs["max_concurrency"] == 1
        assert kwargs["logging_enable"] is False
        payload = item.payload
        self.storage.events.append(("download", self.container, self.name))

        def chunks():
            self.storage.check("chunks")
            if self.storage.stream_mode == "short":
                yield payload[:-1]
            elif self.storage.stream_mode == "long":
                yield payload + b"!"
            elif self.storage.stream_mode == "concurrent":
                item.revision += 1
                self._condition(kwargs, item)
            elif self.storage.stream_mode == "changed_after_read":
                yield payload
                item.revision += 1
            else:
                for start in range(0, len(payload), 97):
                    self._condition(kwargs, item)
                    yield payload[start : start + 97]

        return SimpleNamespace(chunks=chunks)


class _Container:
    def __init__(self, storage, name):
        self.storage, self.name = storage, name

    def get_container_properties(self, **kwargs):
        self.storage.check("container")
        if self.name not in self.storage.objects:
            raise _http(ResourceNotFoundError, 404)
        self.storage.events.append(("container", self.name))
        return SimpleNamespace(public_access=self.storage.public_access[self.name])

    def get_blob_client(self, name):
        assert name.startswith("cmapss/")
        return _Blob(self.storage, self.name, name)

    def list_blobs(self, *, name_starts_with, **kwargs):
        self.storage.events.append(("list", self.name, name_starts_with))
        if self.storage.before_list is not None:
            self.storage.before_list(self.name)
        for name, item in sorted(self.storage.objects[self.name].items()):
            self.storage.check("list")
            if name.startswith(name_starts_with):
                etag = item.etag.strip('"') if self.storage.unquoted_list_etags else item.etag
                yield SimpleNamespace(name=name, size=len(item.payload), etag=etag)


class _Storage:
    def __init__(self, events):
        self.events = events
        self.objects = {publishing.RAW_CONTAINER: {}, publishing.CURATED_CONTAINER: {}}
        self.public_access = dict.fromkeys(self.objects)
        self.failures = {}
        self.races = {}
        self.corrupt_upload = False
        self.stream_mode = None
        self.before_list = None
        self.unquoted_list_etags = False
        self.close = MagicMock()

    def check(self, operation):
        if operation in self.failures:
            raise self.failures[operation]

    def get_container_client(self, name):
        assert name in (publishing.RAW_CONTAINER, publishing.CURATED_CONTAINER)
        return _Container(self, name)


@pytest.fixture
def bundle(data_workspace, observation_row):
    raw = data_workspace / "raw"
    (raw / "files").mkdir(parents=True)
    subsets = {}
    for index, subset in enumerate(SUBSETS):
        for split, cycles in (("train", (1, 2)), ("test", (1,))):
            base = (10.0 if split == "train" else 1000.0) + index
            rows = [observation_row(1, cycle, base=base) for cycle in cycles]
            (raw / "files" / f"{split}_{subset}.txt").write_bytes(
                ("\r\n".join(rows) + "\r\n").encode("ascii")
            )
        (raw / "files" / f"RUL_{subset}.txt").write_bytes(b"7\r\n")
        subsets[subset] = {
            "train_rows": 2,
            "train_units": 1,
            "test_rows": 1,
            "test_units": 1,
            "rul_rows": 1,
            "conditions": 1,
            "fault_modes": 1,
        }
    (raw / "files" / "readme.txt").write_bytes(b"unchanged Windows-1252 \x96\r\n")
    (raw / "files" / "Damage Propagation Modeling.pdf").write_bytes(b"%PDF-1.4\r\nfixture\x00")
    archive = raw / "CMAPSSData.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for path in sorted((raw / "files").iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2000, 1, 1, 0, 0, 0))
            output.writestr(info, path.read_bytes())
    archive_info = fingerprint(archive)
    spec = SourceSpec(
        archive_name=archive.name,
        archive_sha256=archive_info["sha256"],
        archive_size_bytes=archive_info["size_bytes"],
        catalog_url="https://example.invalid/catalog",
        download_url="https://example.invalid/archive",
        files={path.name: fingerprint(path) for path in sorted((raw / "files").iterdir())},
        subsets=subsets,
        source_notes=("Synthetic deterministic publication fixture.",),
        spec_sha256=hashlib.sha256(b"reviewed-synthetic-source").hexdigest(),
    )
    renamed = data_workspace / spec.archive_sha256
    raw.rename(renamed)
    raw = renamed
    manifest = {
        "schema_version": 1,
        "dataset": "nasa-cmapss",
        "archive_sha256": spec.archive_sha256,
        "source": {"catalog_url": spec.catalog_url, "download_url": spec.download_url},
        "files": {
            spec.archive_name: archive_info,
            **{f"files/{name}": value for name, value in spec.files.items()},
        },
    }
    (raw / "raw-manifest.json").write_bytes(canonical_json(manifest))
    curated = curate(raw, data_workspace / "curated", spec)
    config = AzureConfig(subscription_id="11111111-2222-3333-4444-555555555555")
    return SimpleNamespace(
        raw=raw,
        curated=curated.root,
        spec=spec,
        config=config,
        digest=curated.manifest_sha256,
        curated_version=curated.version,
        asset_version="d-"
        + base64.b32encode(bytes.fromhex(curated.manifest_sha256)[:16])
        .decode("ascii")
        .rstrip("=")
        .lower(),
        args=(raw, curated.root, spec, config),
    )


@pytest.fixture
def cloud(bundle, monkeypatch):
    events = []
    state = SimpleNamespace(events=events, datastore=None, asset=None, normalize=False)
    state.storage = _Storage(events)
    state.credential = SimpleNamespace(close=MagicMock())
    state.account = "existingepmstorage"
    config = bundle.config
    state.storage_id = (
        f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}"
        f"/providers/Microsoft.Storage/storageAccounts/{state.account}"
    )

    def get_datastore(name, *, include_secrets):
        assert name == publishing.DATASTORE_NAME
        assert include_secrets is False
        events.append(("get_datastore",))
        if state.datastore is None:
            raise _http(ResourceNotFoundError, 404)
        return state.datastore

    def create_datastore(datastore):
        assert state.datastore is None
        assert isinstance(datastore, AzureBlobDatastore)
        assert isinstance(datastore.credentials, NoneCredentialConfiguration)
        events.append(("create_datastore",))
        state.datastore = datastore
        return datastore

    def get_data(*, name, version):
        assert name == publishing.ASSET_NAME
        assert version == bundle.asset_version
        events.append(("get_asset",))
        if state.asset is None:
            raise _http(ResourceNotFoundError, 404)
        return state.asset

    def create_data(asset):
        assert state.asset is None
        assert isinstance(asset, Data)
        assert asset.version == bundle.asset_version
        assert asset.path == (
            f"azureml://datastores/{publishing.DATASTORE_NAME}"
            f"/paths/cmapss/{bundle.curated_version}/"
        )
        events.append(("create_asset",))
        state.asset = asset
        return asset

    state.ml = SimpleNamespace(
        workspaces=SimpleNamespace(
            get=MagicMock(return_value=SimpleNamespace(storage_account=state.storage_id))
        ),
        datastores=SimpleNamespace(
            get=MagicMock(side_effect=get_datastore),
            create_or_update=MagicMock(side_effect=create_datastore),
        ),
        data=SimpleNamespace(
            get=MagicMock(side_effect=get_data),
            create_or_update=MagicMock(side_effect=create_data),
        ),
    )
    state.create_credential = MagicMock(return_value=state.credential)
    state.create_ml_client = MagicMock(return_value=state.ml)
    state.create_blob_service = MagicMock(return_value=state.storage)
    monkeypatch.setattr(publishing, "create_credential", state.create_credential)
    monkeypatch.setattr(publishing, "create_ml_client", state.create_ml_client)
    monkeypatch.setattr(publishing, "BlobServiceClient", state.create_blob_service)
    return state


def _seed(bundle, cloud):
    for root, container in (
        (bundle.raw, publishing.RAW_CONTAINER),
        (bundle.curated, publishing.CURATED_CONTAINER),
    ):
        for path in root.rglob("*"):
            if path.is_file():
                name = f"cmapss/{root.name}/{path.relative_to(root).as_posix()}"
                cloud.storage.objects[container][name] = _Object(
                    path.read_bytes(), fingerprint(path)
                )
    cloud.datastore = AzureBlobDatastore(
        name=publishing.DATASTORE_NAME,
        account_name=cloud.account,
        container_name=publishing.CURATED_CONTAINER,
        endpoint="core.windows.net",
        credentials=None,
    )
    cloud.asset = Data(
        name=publishing.ASSET_NAME,
        version=bundle.asset_version,
        type="uri_folder",
        path=(
            f"azureml://datastores/{publishing.DATASTORE_NAME}"
            f"/paths/cmapss/{bundle.curated_version}/"
        ),
        tags={
            "source_sha256": bundle.spec.archive_sha256,
            "manifest_sha256": bundle.digest,
            "source_spec_sha256": bundle.spec.spec_sha256,
            "phase": "2",
            "dataset": "nasa-cmapss",
            "recipe_version": "1",
            "subsets": ",".join(SUBSETS),
        },
    )


def _assert_no_writes(cloud):
    assert not any(event[0] == "upload" for event in cloud.events)
    cloud.ml.datastores.create_or_update.assert_not_called()
    cloud.ml.data.create_or_update.assert_not_called()


@pytest.mark.parametrize(
    "digest", ["00" * 32, "ff" * 32, "fb" * 32, "00" * 31 + "01", bytes(range(32)).hex()]
)
def test_registry_alias_is_lowercase_128_bit_selector_under_30_characters(digest):
    version = publishing._asset_version(digest)
    assert version.startswith("d-")
    assert len(version) == 28
    assert re.fullmatch(r"[a-z0-9-]{1,30}", version)
    decoded = base64.b32decode(version.removeprefix("d-").upper() + "=" * 6)
    assert decoded.hex() == digest[:32]


@pytest.mark.parametrize("field", ["tag", "path"])
def test_alias_collision_never_repoints_or_reuses_different_full_digest(bundle, cloud, field):
    _seed(bundle, cloud)
    replacement = "0" if bundle.digest[-1] != "0" else "1"
    other_digest = bundle.digest[:-1] + replacement
    assert other_digest != bundle.digest
    assert publishing._asset_version(other_digest) == bundle.asset_version
    if field == "tag":
        cloud.asset.tags["manifest_sha256"] = other_digest
    else:
        cloud.asset.path = cloud.asset.path.replace(bundle.digest, other_digest)
    with pytest.raises(DataError):
        publishing.publish(*bundle.args)
    _assert_no_writes(cloud)


def test_first_publish_preserves_31_objects_and_registers_last(bundle, cloud):
    receipt = publishing.publish(*bundle.args)
    assert receipt == {
        "schema_version": 1,
        "verified_at_utc": receipt["verified_at_utc"],
        "asset_name": publishing.ASSET_NAME,
        "asset_version": bundle.asset_version,
        "curated_version": bundle.curated_version,
        "asset_type": "uri_folder",
        "datastore_name": publishing.DATASTORE_NAME,
        "raw_archive_sha256": bundle.spec.archive_sha256,
        "curated_manifest_sha256": bundle.digest,
        "raw_objects_verified": 16,
        "curated_objects_verified": 15,
        "uploaded_objects": 31,
        "reused_objects": 0,
        "registered_new_version": True,
        "workspace_name": bundle.config.workspace_name,
        "raw_container": publishing.RAW_CONTAINER,
        "curated_container": publishing.CURATED_CONTAINER,
        "jobs_submitted": False,
    }
    assert receipt["verified_at_utc"].endswith("Z")
    assert bundle.curated.name == bundle.curated_version == "sha256-" + bundle.digest
    assert cloud.asset.version == bundle.asset_version
    assert cloud.asset.path.endswith(f"/cmapss/{bundle.curated_version}/")
    assert cloud.asset.tags["manifest_sha256"] == bundle.digest
    assert all(
        call.kwargs["version"] == bundle.asset_version for call in cloud.ml.data.get.call_args_list
    )
    for root, container, marker, count in (
        (bundle.raw, publishing.RAW_CONTAINER, "raw-manifest.json", 16),
        (bundle.curated, publishing.CURATED_CONTAINER, "_SUCCESS.json", 15),
    ):
        uploads = [
            event for event in cloud.events if event[0] == "upload" and event[1] == container
        ]
        assert len(uploads) == count
        assert uploads[-1][2] == f"cmapss/{root.name}/{marker}"
        marker_index = cloud.events.index(uploads[-1])
        for event in uploads[:-1]:
            assert cloud.events.index(("download", container, event[2])) < marker_index
        for name, item in cloud.storage.objects[container].items():
            relative = name.removeprefix(f"cmapss/{root.name}/")
            assert item.payload == root.joinpath(*relative.split("/")).read_bytes()
    register_index = cloud.events.index(("create_asset",))
    assert max(i for i, event in enumerate(cloud.events) if event[0] == "download") < register_index
    assert sum(event[0] == "download" for event in cloud.events) == 31
    assert any(event[0] == "get_asset" for event in cloud.events[register_index + 1 :])
    kwargs = cloud.create_blob_service.call_args.kwargs
    assert kwargs["account_url"] == "https://existingepmstorage.blob.core.windows.net"
    assert kwargs["credential"] is cloud.credential
    cloud.create_credential.assert_called_once_with(bundle.config)
    cloud.create_ml_client.assert_called_once_with(bundle.config, cloud.credential)
    cloud.ml.workspaces.get.assert_called_once_with(name=bundle.config.workspace_name)
    cloud.credential.close.assert_called_once()
    cloud.storage.close.assert_called_once()
    serialized = json.dumps(receipt)
    assert bundle.config.subscription_id not in serialized
    assert cloud.account not in serialized
    assert _SECRET not in serialized
    assert "azureml://" not in serialized


@pytest.mark.parametrize("already_published", [False, True])
def test_publish_reuses_identical_bytes_and_metadata(bundle, cloud, already_published):
    if already_published:
        publishing.publish(*bundle.args)
        cloud.events.clear()
        cloud.ml.data.create_or_update.reset_mock()
        cloud.ml.datastores.create_or_update.reset_mock()
    else:
        _seed(bundle, cloud)
    receipt = publishing.publish(*bundle.args)
    assert receipt["uploaded_objects"] == 0
    assert receipt["reused_objects"] == 31
    assert receipt["registered_new_version"] is False
    assert receipt["asset_version"] == bundle.asset_version
    assert receipt["curated_version"] == bundle.curated_version
    assert all(
        call.kwargs["version"] == bundle.asset_version for call in cloud.ml.data.get.call_args_list
    )
    _assert_no_writes(cloud)
    assert sum(event[0] == "download" for event in cloud.events) == 31


@pytest.mark.parametrize("matching", [True, False])
def test_conditional_create_race_checks_winners_bytes(bundle, cloud, matching):
    name = f"cmapss/{bundle.spec.archive_sha256}/CMAPSSData.zip"
    payload = (bundle.raw / "CMAPSSData.zip").read_bytes()
    cloud.storage.races[name] = None if matching else b"!" + payload[1:]
    if matching:
        receipt = publishing.publish(*bundle.args)
        assert receipt["uploaded_objects"] == 30
        assert receipt["reused_objects"] == 1
    else:
        with pytest.raises(DataError, match="checksum"):
            publishing.publish(*bundle.args)
        cloud.ml.data.create_or_update.assert_not_called()
        assert len(cloud.storage.objects[publishing.RAW_CONTAINER]) == 1
    assert cloud.storage.objects[publishing.RAW_CONTAINER][name].payload == (
        payload if matching else b"!" + payload[1:]
    )


@pytest.mark.parametrize(
    "container,relative",
    [
        (publishing.RAW_CONTAINER, "CMAPSSData.zip"),
        (publishing.RAW_CONTAINER, "raw-manifest.json"),
        (publishing.CURATED_CONTAINER, "FD004/test.parquet"),
        (publishing.CURATED_CONTAINER, "manifest.json"),
        (publishing.CURATED_CONTAINER, "_SUCCESS.json"),
    ],
)
@pytest.mark.parametrize("corruption", ["size", "checksum"])
def test_existing_remote_corruption_is_never_overwritten(
    bundle, cloud, container, relative, corruption
):
    _seed(bundle, cloud)
    root = bundle.raw if container == publishing.RAW_CONTAINER else bundle.curated
    item = cloud.storage.objects[container][f"cmapss/{root.name}/{relative}"]
    original_metadata = dict(item.metadata)
    item.payload = item.payload + b"!" if corruption == "size" else b"!" + item.payload[1:]
    item.revision += 1
    with pytest.raises(DataError, match=corruption):
        publishing.publish(*bundle.args)
    assert item.metadata == original_metadata  # Caller-set metadata is not integrity evidence.
    _assert_no_writes(cloud)


@pytest.mark.parametrize("container", [publishing.RAW_CONTAINER, publishing.CURATED_CONTAINER])
def test_unexpected_prefix_objects_block_registration_and_marker(bundle, cloud, container):
    root = bundle.raw if container == publishing.RAW_CONTAINER else bundle.curated
    extra = f"cmapss/{root.name}/unexpected.txt"
    cloud.storage.objects[container][extra] = _Object(b"extra", {})
    with pytest.raises(DataError, match="unexpected objects"):
        publishing.publish(*bundle.args)
    cloud.ml.data.create_or_update.assert_not_called()
    marker = "raw-manifest.json" if container == publishing.RAW_CONTAINER else "_SUCCESS.json"
    assert f"cmapss/{root.name}/{marker}" not in cloud.storage.objects[container]
    assert cloud.storage.objects[container][extra].payload == b"extra"


def test_other_versions_outside_exact_prefix_are_untouched(bundle, cloud):
    _seed(bundle, cloud)
    outside = f"cmapss/{bundle.curated_version}-other/not-part-of-this-version"
    cloud.storage.objects[publishing.CURATED_CONTAINER][outside] = _Object(b"other", {})
    receipt = publishing.verify_publication(*bundle.args)
    assert receipt["curated_objects_verified"] == 15
    assert cloud.storage.objects[publishing.CURATED_CONTAINER][outside].payload == b"other"
    _assert_no_writes(cloud)


@pytest.mark.parametrize("kind", ["raw", "curated", "source_archive", "source_spec", "extra_local"])
def test_local_validation_aborts_before_any_azure_access(bundle, cloud, kind):
    args = bundle.args
    if kind == "raw":
        (bundle.raw / "files" / "readme.txt").write_bytes(b"changed")
    elif kind == "curated":
        (bundle.curated / "FD001" / "train.parquet").write_bytes(b"changed")
    elif kind == "source_archive":
        args = (*args[:2], replace(bundle.spec, archive_sha256="a" * 64), bundle.config)
    elif kind == "source_spec":
        args = (*args[:2], replace(bundle.spec, spec_sha256="b" * 64), bundle.config)
    else:
        (bundle.raw / "unrelated.txt").write_bytes(b"never upload")
    with pytest.raises(DataError):
        publishing.publish(*args)
    cloud.create_credential.assert_not_called()
    cloud.create_ml_client.assert_not_called()
    cloud.create_blob_service.assert_not_called()
    _assert_no_writes(cloud)


@pytest.mark.parametrize("field", ["source_archive_sha256", "source_spec_sha256"])
def test_publication_itself_checks_source_ties_even_with_mocked_verifiers(
    bundle, cloud, monkeypatch, field
):
    manifest = json.loads((bundle.curated / "manifest.json").read_bytes())
    manifest[field] = "a" * 64
    monkeypatch.setattr(publishing, "verify_curated", lambda *_: manifest)
    with pytest.raises(DataError, match="pinned source and recipe"):
        publishing.publish(*bundle.args)
    cloud.create_credential.assert_not_called()


@pytest.mark.parametrize(
    "unsafe", ["../outside", "FD001/../../outside", "C:\\outside", "FD001\\test.parquet"]
)
def test_mocked_manifest_cannot_escape_the_local_bundle(bundle, cloud, monkeypatch, unsafe):
    manifest = json.loads((bundle.curated / "manifest.json").read_bytes())
    manifest["files"][0]["path"] = unsafe
    monkeypatch.setattr(publishing, "verify_curated", lambda *_: manifest)
    with pytest.raises(DataError, match="unsafe or incomplete inventory"):
        publishing.publish(*bundle.args)
    cloud.create_credential.assert_not_called()


def test_inventory_still_blocks_unrelated_files_with_mocked_verifiers(bundle, cloud, monkeypatch):
    manifest = json.loads((bundle.curated / "manifest.json").read_bytes())
    monkeypatch.setattr(publishing, "verify_curated", lambda *_: manifest)
    (bundle.curated / "private-unrelated.txt").write_bytes(b"not an upload candidate")
    with pytest.raises(DataError, match="unexpected filesystem entry"):
        publishing.publish(*bundle.args)
    cloud.create_credential.assert_not_called()


@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
def test_inventory_rejects_links_without_needing_os_link_privileges(
    bundle, cloud, monkeypatch, link_kind
):
    manifest = json.loads((bundle.curated / "manifest.json").read_bytes())
    monkeypatch.setattr(publishing, "verify_curated", lambda *_: manifest)
    path_type = type(bundle.curated)
    method = "is_symlink" if link_kind == "symlink" else "is_junction"
    original = getattr(path_type, method)
    target = bundle.curated / "FD001" / "train.parquet"
    monkeypatch.setattr(path_type, method, lambda path: path == target or original(path))
    with pytest.raises(DataError, match="links or junctions"):
        publishing.publish(*bundle.args)
    cloud.create_credential.assert_not_called()


@pytest.mark.parametrize(
    "bad_id",
    [
        "/subscriptions/other/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/account123",
        "/subscriptions/{sub}/resourceGroups/other/providers/Microsoft.Storage/storageAccounts/account123",
        "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/BadAccount",
        "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/aa",
        "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/good?sig=bad",
        "/subscriptions/{sub}/resourceGroups/{rg}/providers/Other/storageAccounts/account123",
        "https://account123.blob.core.windows.net",
        None,
    ],
)
def test_workspace_storage_must_be_in_configured_scope(bundle, cloud, bad_id):
    if bad_id is not None:
        bad_id = bad_id.format(sub=bundle.config.subscription_id, rg=bundle.config.resource_group)
    cloud.ml.workspaces.get.return_value.storage_account = bad_id
    with pytest.raises(DataError, match="configured subscription/RG"):
        publishing.publish(*bundle.args)
    cloud.create_blob_service.assert_not_called()
    _assert_no_writes(cloud)


@pytest.mark.parametrize("container", [publishing.RAW_CONTAINER, publishing.CURATED_CONTAINER])
@pytest.mark.parametrize("problem", ["missing", "blob", "container"])
def test_both_containers_must_exist_and_be_private(bundle, cloud, container, problem):
    if problem == "missing":
        del cloud.storage.objects[container]
    else:
        cloud.storage.public_access[container] = problem
    with pytest.raises(DataError, match="provisioning") as caught:
        publishing.publish(*bundle.args)
    assert _SECRET not in str(caught.value)
    _assert_no_writes(cloud)


@pytest.mark.parametrize(
    "problem",
    [
        "account",
        "container",
        "protocol",
        "endpoint",
        "key",
        "sas",
        "type",
        "scope",
    ],
)
def test_existing_datastore_mismatch_never_repoints_or_uploads(bundle, cloud, problem):
    _seed(bundle, cloud)
    if problem == "account":
        cloud.datastore.account_name = "otheraccount"
    elif problem == "container":
        cloud.datastore.container_name = publishing.RAW_CONTAINER
    elif problem == "protocol":
        cloud.datastore.protocol = "http"
    elif problem == "endpoint":
        cloud.datastore.endpoint = "example.invalid"
    elif problem == "key":
        cloud.datastore.credentials = AccountKeyConfiguration(account_key="secret-key")
    elif problem == "sas":
        cloud.datastore.credentials = SasTokenConfiguration(sas_token="secret-token")
    elif problem == "scope":
        cloud.datastore = AzureBlobDatastore(
            name=publishing.DATASTORE_NAME,
            account_name=cloud.account,
            container_name=publishing.CURATED_CONTAINER,
            endpoint="core.windows.net",
            credentials=None,
            id="/subscriptions/other/datastores/epm_cmapss_curated",
        )
    else:
        cloud.datastore = SimpleNamespace(account_name=cloud.account)
    with pytest.raises(DataError, match="will not be repointed"):
        publishing.publish(*bundle.args)
    _assert_no_writes(cloud)


@pytest.mark.parametrize(
    "problem",
    [
        "type",
        "path",
        "query",
        "fragment",
        "other_workspace",
        "other_subscription",
        "other_container",
        "other_version",
        "registry_version_in_path",
        "curated_version_in_registry",
        "double_slash",
        "source_sha256",
        "manifest_sha256",
        "source_spec_sha256",
        "phase",
    ],
)
def test_existing_asset_mismatch_never_updates_or_uploads(bundle, cloud, problem):
    _seed(bundle, cloud)
    if problem == "type":
        cloud.asset.type = "uri_file"
    elif problem == "path":
        cloud.asset.path = str(bundle.curated)
    elif problem == "query":
        cloud.asset.path += "?sig=secret-token"
    elif problem == "fragment":
        cloud.asset.path += "#unexpected"
    elif problem == "other_container":
        cloud.asset.path = cloud.asset.path.replace(publishing.DATASTORE_NAME, "other_datastore")
    elif problem == "other_version":
        cloud.asset.path = cloud.asset.path.replace(bundle.curated_version, "other_version")
    elif problem == "registry_version_in_path":
        cloud.asset.path = cloud.asset.path.replace(bundle.curated_version, bundle.asset_version)
    elif problem == "curated_version_in_registry":
        cloud.asset.version = bundle.curated_version
    elif problem == "double_slash":
        cloud.asset.path += "/"
    elif problem.startswith("other_"):
        sub = "other" if problem == "other_subscription" else bundle.config.subscription_id
        workspace = "other" if problem == "other_workspace" else bundle.config.workspace_name
        cloud.asset.path = (
            f"azureml://subscriptions/{sub}/resourcegroups/{bundle.config.resource_group}"
            f"/workspaces/{workspace}/datastores/{publishing.DATASTORE_NAME}"
            f"/paths/cmapss/{bundle.curated_version}/"
        )
    else:
        cloud.asset.tags[problem] = "incorrect"
    with pytest.raises(DataError, match="conflicting type, remote path or provenance tags"):
        publishing.publish(*bundle.args)
    _assert_no_writes(cloud)


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("trailing_slash", [False, True])
def test_exact_sdk_normalized_asset_uri_is_accepted(bundle, cloud, expanded, trailing_slash):
    _seed(bundle, cloud)
    if expanded:
        cloud.asset.path = (
            f"azureml://subscriptions/{bundle.config.subscription_id}"
            f"/resourcegroups/{bundle.config.resource_group}/workspaces/{bundle.config.workspace_name}"
            f"/datastores/{publishing.DATASTORE_NAME}/paths/cmapss/{bundle.curated_version}/"
        )
    if not trailing_slash:
        cloud.asset.path = cloud.asset.path.removesuffix("/")
    receipt = publishing.verify_publication(*bundle.args)
    assert receipt["reused_objects"] == 31
    assert receipt["asset_version"] == bundle.asset_version
    assert receipt["curated_version"] == bundle.curated_version
    assert bundle.asset_version not in cloud.asset.path
    _assert_no_writes(cloud)


def test_new_registration_accepts_normalized_path_with_separate_versions(bundle, cloud):
    original_create = cloud.ml.data.create_or_update.side_effect

    def normalized_create(asset):
        original_create(asset)
        asset.path = (
            f"azureml://subscriptions/{bundle.config.subscription_id}"
            f"/resourcegroups/{bundle.config.resource_group}/workspaces/{bundle.config.workspace_name}"
            f"/datastores/{publishing.DATASTORE_NAME}/paths/cmapss/{bundle.curated_version}/"
        )
        return asset

    cloud.ml.data.create_or_update.side_effect = normalized_create
    receipt = publishing.publish(*bundle.args)
    assert receipt["registered_new_version"] is True
    assert receipt["asset_version"] == bundle.asset_version
    assert receipt["curated_version"] == bundle.curated_version
    assert cloud.asset.version == bundle.asset_version
    assert cloud.asset.path.endswith(f"/{bundle.curated_version}/")
    assert bundle.asset_version not in cloud.asset.path


@pytest.mark.parametrize("read_only", [False, True])
def test_asset_lookup_http400_remains_a_failure_not_a_missing_version(bundle, cloud, read_only):
    if read_only:
        _seed(bundle, cloud)
    cloud.ml.data.get.side_effect = _http(HttpResponseError, 400)
    operation = publishing.verify_publication if read_only else publishing.publish
    with pytest.raises(DataError, match=r"Data asset lookup failed \(HTTP 400\)") as caught:
        operation(*bundle.args)
    assert _SECRET not in str(caught.value)
    cloud.ml.data.get.assert_called_once_with(
        name=publishing.ASSET_NAME, version=bundle.asset_version
    )
    _assert_no_writes(cloud)


def test_registration_failure_retains_complete_bytes_for_resume(bundle, cloud):
    cloud.ml.data.create_or_update.side_effect = _http(HttpResponseError, 503)
    with pytest.raises(DataError, match="Data asset registration failed.*503") as caught:
        publishing.publish(*bundle.args)
    assert _SECRET not in str(caught.value)
    assert len(cloud.storage.objects[publishing.RAW_CONTAINER]) == 16
    assert len(cloud.storage.objects[publishing.CURATED_CONTAINER]) == 15
    assert cloud.asset is None
    cloud.ml.data.create_or_update.side_effect = lambda asset: setattr(cloud, "asset", asset)
    cloud.events.clear()
    receipt = publishing.publish(*bundle.args)
    assert receipt["uploaded_objects"] == 0
    assert receipt["reused_objects"] == 31
    assert receipt["registered_new_version"] is True


def test_retrieved_registration_must_match_exact_reference(bundle, cloud):
    def register_wrong(asset):
        asset.path = "azureml://datastores/wrong/paths/wrong/"
        cloud.asset = asset
        return asset

    cloud.ml.data.create_or_update.side_effect = register_wrong
    with pytest.raises(DataError, match="conflicting"):
        publishing.publish(*bundle.args)
    assert cloud.asset is not None


@pytest.mark.parametrize(
    "stage",
    [
        "workspace",
        "container",
        "properties",
        "download",
        "chunks",
        "list",
        "upload",
        "datastore_get",
        "datastore_create",
        "asset_get",
        "asset_create",
    ],
)
def test_unauthorized_errors_are_sanitized_and_actionable(bundle, cloud, capsys, stage):
    error = _http(HttpResponseError, 403)
    operations = {
        "workspace": cloud.ml.workspaces.get,
        "datastore_get": cloud.ml.datastores.get,
        "datastore_create": cloud.ml.datastores.create_or_update,
        "asset_get": cloud.ml.data.get,
        "asset_create": cloud.ml.data.create_or_update,
    }
    if stage in operations:
        operations[stage].side_effect = error
    else:
        cloud.storage.failures[stage] = error
    with pytest.raises(DataError, match="HTTP 403") as caught:
        publishing.publish(*bundle.args)
    text = str(caught.value)
    assert "publishing identity" in text
    assert "container-scoped Storage Blob Data Contributor" in text
    assert _SECRET not in text
    assert bundle.config.subscription_id not in text
    assert caught.value.__suppress_context__ is True
    captured = capsys.readouterr()
    assert _SECRET not in captured.out + captured.err


@pytest.mark.parametrize("mode", ["short", "long", "concurrent", "changed_after_read"])
def test_streams_are_fully_hashed_and_etag_conditioned(bundle, cloud, mode):
    cloud.storage.stream_mode = mode
    with pytest.raises(DataError):
        publishing.publish(*bundle.args)
    cloud.ml.data.create_or_update.assert_not_called()
    assert len(cloud.storage.objects[publishing.RAW_CONTAINER]) == 1


def test_uploaded_bytes_are_downloaded_and_verified(bundle, cloud):
    cloud.storage.corrupt_upload = True
    with pytest.raises(DataError, match="checksum"):
        publishing.publish(*bundle.args)
    cloud.ml.data.create_or_update.assert_not_called()
    assert len(cloud.storage.objects[publishing.RAW_CONTAINER]) == 1


def test_final_inventory_rejects_changes_after_checksums(bundle, cloud):
    def mutate_raw_during_curated_listing(container):
        if container == publishing.CURATED_CONTAINER:
            item = next(iter(cloud.storage.objects[publishing.RAW_CONTAINER].values()))
            item.payload = b"!" + item.payload[1:]
            item.revision += 1

    cloud.storage.before_list = mutate_raw_during_curated_listing
    with pytest.raises(DataError, match="changed after checksum verification"):
        publishing.publish(*bundle.args)
    cloud.ml.data.create_or_update.assert_not_called()


@pytest.mark.parametrize("unquoted_etags", [False, True])
def test_verify_publication_is_strictly_read_only(bundle, cloud, unquoted_etags):
    _seed(bundle, cloud)
    cloud.storage.unquoted_list_etags = unquoted_etags
    receipt = publishing.verify_publication(*bundle.args)
    assert receipt["uploaded_objects"] == 0
    assert receipt["reused_objects"] == 31
    assert receipt["registered_new_version"] is False
    assert sum(event[0] == "download" for event in cloud.events) == 31
    _assert_no_writes(cloud)


@pytest.mark.parametrize("missing", ["datastore", "asset", "raw_blob", "curated_blob", "success"])
def test_verify_missing_resources_never_repairs_them(bundle, cloud, missing):
    _seed(bundle, cloud)
    if missing in ("datastore", "asset"):
        setattr(cloud, missing, None)
    else:
        if missing == "raw_blob":
            container, root, relative = publishing.RAW_CONTAINER, bundle.raw, "CMAPSSData.zip"
        elif missing == "curated_blob":
            container, root, relative = (
                publishing.CURATED_CONTAINER,
                bundle.curated,
                "FD001/test.parquet",
            )
        else:
            container, root, relative = (
                publishing.CURATED_CONTAINER,
                bundle.curated,
                "_SUCCESS.json",
            )
        del cloud.storage.objects[container][f"cmapss/{root.name}/{relative}"]
    with pytest.raises(DataError):
        publishing.verify_publication(*bundle.args)
    _assert_no_writes(cloud)


def test_verify_detects_external_mutation_after_successful_publish(bundle, cloud):
    publishing.publish(*bundle.args)
    cloud.events.clear()
    cloud.ml.datastores.create_or_update.reset_mock()
    cloud.ml.data.create_or_update.reset_mock()
    name = f"cmapss/{bundle.curated_version}/FD003/test_rul.parquet"
    item = cloud.storage.objects[publishing.CURATED_CONTAINER][name]
    item.payload = b"!" + item.payload[1:]
    item.revision += 1
    with pytest.raises(DataError, match="checksum"):
        publishing.verify_publication(*bundle.args)
    _assert_no_writes(cloud)
