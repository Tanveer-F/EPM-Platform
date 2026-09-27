"""Keyless, write-once publication of the two pinned C-MAPSS bundles.

Checksum verification and conditional blob creation are application safeguards,
not a storage immutability policy. Verification detects subsequent external edits.
"""

import base64
import hashlib
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from azure.ai.ml.entities import AzureBlobDatastore, Data, NoneCredentialConfiguration
from azure.core import MatchConditions
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from epm_platform.client import create_credential, create_ml_client
from epm_platform.config import AzureConfig
from epm_platform.data.curation import verify_curated
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint
from epm_platform.data.source import verify_raw
from epm_platform.data.spec import (
    EXPECTED_MEMBERS,
    LABEL_COLUMNS,
    OBSERVATION_COLUMNS,
    SUBSETS,
    SourceSpec,
)

RAW_CONTAINER = "epm-cmapss-raw"
CURATED_CONTAINER = "epm-cmapss-curated"
DATASTORE_NAME = "epm_cmapss_curated"
ASSET_NAME = "epm-cmapss-curated"
_SHA256 = re.compile(r"[a-f0-9]{64}")
_STORAGE_ID = re.compile(
    r"/subscriptions/([^/]+)/resourceGroups/([^/]+)/providers/"
    r"Microsoft\.Storage/storageAccounts/([^/]+)/?",
    re.IGNORECASE,
)
_CURATED_CONTENT = frozenset(
    {f"{subset}/{split}.parquet" for subset in SUBSETS for split in ("train", "test", "test_rul")}
    | {"data-quality.json"}
)
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class _Bundle:
    root: Path
    prefix: str
    files: dict[str, dict]
    marker: str


@contextmanager
def _operation(name: str):
    try:
        yield
    except DataError:
        raise
    except Exception as error:
        # Never interpolate exception text, response bodies, URLs or server error codes.
        status = getattr(error, "status_code", None)
        code = f"HTTP {status}" if type(status) is int and 100 <= status <= 599 else "no status"
        if status in (401, 403):
            guidance = (
                "Authenticate as the configured publishing identity; confirm Azure ML access, "
                "the approved container-scoped Storage Blob Data Contributor grants on both "
                "containers, and private-network access. No key fallback is permitted."
            )
        elif status == 404:
            guidance = "Required resource is missing; apply the approved Phase 2 provisioning."
        elif status in (409, 412):
            guidance = "Concurrent change or conflict detected; inspect it before retrying."
        else:
            guidance = "Check configured identity, service availability and private-network access."
        raise DataError(f"{name} failed ({code}). {guidance}") from None


def _check_inventory(root: Path, expected: set[str]) -> None:
    """Only exact, regular local bundle entries can become upload candidates."""
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink() or ancestor.is_junction():
            raise DataError("Publication bundles must not use links or junctions.")
    if not root.is_dir():
        raise DataError("Publication bundle directory is missing.")
    directories = {name.split("/")[0] for name in expected if "/" in name}
    found = set()
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink() or path.is_junction():
                raise DataError("Publication bundles must not contain links or junctions.")
            relative = path.relative_to(root).as_posix()
            if path.is_dir() and relative in directories:
                pending.append(path)
            elif path.is_file() and relative in expected:
                found.add(relative)
            else:
                raise DataError("Publication bundle contains an unexpected filesystem entry.")
    if found != expected:
        raise DataError("Publication bundle inventory is incomplete.")


def _check_fingerprints(files: dict[str, dict]) -> None:
    for info in files.values():
        if (
            not isinstance(info, dict)
            or set(info) != {"sha256", "size_bytes"}
            or not isinstance(info["sha256"], str)
            or not _SHA256.fullmatch(info["sha256"])
            or type(info["size_bytes"]) is not int
            or info["size_bytes"] <= 0
        ):
            raise DataError("Publication manifest contains an invalid fingerprint.")


def _prepare(raw_root: Path, curated_root: Path, spec: SourceSpec) -> tuple[_Bundle, _Bundle, str]:
    raw_manifest = verify_raw(raw_root, spec)
    curated_manifest = verify_curated(curated_root, spec)
    if (
        spec.archive_name != "CMAPSSData.zip"
        or not _SHA256.fullmatch(spec.archive_sha256)
        or not _SHA256.fullmatch(spec.spec_sha256)
        or set(spec.files) != EXPECTED_MEMBERS
        or raw_root.name != spec.archive_sha256
        or raw_manifest.get("archive_sha256") != spec.archive_sha256
    ):
        raise DataError("Publication source identity differs from the pinned C-MAPSS source.")
    raw_files = {
        spec.archive_name: {"sha256": spec.archive_sha256, "size_bytes": spec.archive_size_bytes},
        **{f"files/{name}": dict(info) for name, info in spec.files.items()},
    }
    if raw_manifest.get("files") != raw_files:
        raise DataError("Publication raw manifest differs from the pinned source fingerprints.")
    header = {
        "schema_version": 1,
        "dataset": "nasa-cmapss",
        "source_archive_sha256": spec.archive_sha256,
        "source_spec_sha256": spec.spec_sha256,
        "recipe_version": "1",
        "subsets": list(SUBSETS),
        "observation_columns": list(OBSERVATION_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
    }
    if any(curated_manifest.get(key) != value for key, value in header.items()) or not isinstance(
        curated_manifest.get("writer"), dict
    ):
        raise DataError("Publication curated manifest differs from the pinned source and recipe.")
    entries = curated_manifest.get("files")
    if (
        not isinstance(entries, list)
        or len(entries) != len(_CURATED_CONTENT)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in _CURATED_CONTENT
            for entry in entries
        )
        or {entry["path"] for entry in entries} != _CURATED_CONTENT
    ):
        raise DataError("Publication curated manifest has an unsafe or incomplete inventory.")
    curated_files = {
        entry["path"]: {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
        for entry in entries
    }
    _check_fingerprints(raw_files)
    _check_fingerprints(curated_files)
    _check_inventory(raw_root, set(raw_files) | {"raw-manifest.json"})
    _check_inventory(curated_root, set(curated_files) | {"manifest.json", "_SUCCESS.json"})
    for root, name, manifest in (
        (raw_root, "raw-manifest.json", raw_manifest),
        (curated_root, "manifest.json", curated_manifest),
    ):
        if (root / name).read_bytes() != canonical_json(manifest):
            raise DataError("Publication manifest bytes changed after local verification.")
    manifest_info = fingerprint(curated_root / "manifest.json")
    digest = manifest_info["sha256"]
    version = f"sha256-{digest}"
    if curated_root.name != version or (
        curated_root / "_SUCCESS.json"
    ).read_bytes() != canonical_json({"manifest_sha256": digest, "version": version}):
        raise DataError("Publication curated version or completion marker is invalid.")
    raw_files["raw-manifest.json"] = fingerprint(raw_root / "raw-manifest.json")
    curated_files["manifest.json"] = manifest_info
    curated_files["_SUCCESS.json"] = fingerprint(curated_root / "_SUCCESS.json")
    return (
        _Bundle(raw_root, f"cmapss/{spec.archive_sha256}/", raw_files, "raw-manifest.json"),
        _Bundle(curated_root, f"cmapss/{version}/", curated_files, "_SUCCESS.json"),
        digest,
    )


def _storage_account(workspace, config: AzureConfig) -> str:
    resource_id = getattr(workspace, "storage_account", None)
    match = _STORAGE_ID.fullmatch(resource_id) if isinstance(resource_id, str) else None
    if (
        match is None
        or match[1].casefold() != config.subscription_id.casefold()
        or match[2].casefold() != config.resource_group.casefold()
        or not re.fullmatch(r"[a-z0-9]{3,24}", match[3])
    ):
        raise DataError(
            "Workspace storage must be a valid account in the configured subscription/RG."
        )
    return match[3]


def _get_datastore(client):
    with _operation("Datastore lookup"):
        try:
            return client.datastores.get(DATASTORE_NAME, include_secrets=False)
        except ResourceNotFoundError:
            return None


def _check_datastore(datastore, account: str, config: AzureConfig) -> None:
    expected_id = (
        f"/subscriptions/{config.subscription_id}/resourceGroups/{config.resource_group}/providers/"
        f"Microsoft.MachineLearningServices/workspaces/{config.workspace_name}/datastores/"
        f"{DATASTORE_NAME}"
    )
    if (
        not isinstance(datastore, AzureBlobDatastore)
        or datastore.name != DATASTORE_NAME
        or datastore.account_name != account
        or datastore.container_name != CURATED_CONTAINER
        or datastore.protocol != "https"
        or datastore.endpoint != "core.windows.net"
        or not (
            datastore.credentials is None
            or isinstance(datastore.credentials, NoneCredentialConfiguration)
        )
        or (
            datastore.id is not None
            and datastore.id.removesuffix("/").casefold() != expected_id.casefold()
        )
    ):
        raise DataError(
            "Existing datastore must match the workspace, storage account and curated container, "
            "use HTTPS, and have no stored credentials. It will not be repointed or updated."
        )


def _asset_version(manifest_sha256: str) -> str:
    """Create a 28-character selector; full digest and path checks prevent collision reuse."""
    prefix = bytes.fromhex(manifest_sha256)[:16]
    return "d-" + base64.b32encode(prefix).decode("ascii").rstrip("=").lower()


def _asset_path(curated_version: str) -> str:
    return f"azureml://datastores/{DATASTORE_NAME}/paths/cmapss/{curated_version}/"


def _tags(spec: SourceSpec, digest: str) -> dict[str, str]:
    return {
        "source_sha256": spec.archive_sha256,
        "manifest_sha256": digest,
        "source_spec_sha256": spec.spec_sha256,
        "phase": "2",
        "dataset": "nasa-cmapss",
        "recipe_version": "1",
        "subsets": ",".join(SUBSETS),
    }


def _get_asset(client, asset_version: str):
    with _operation("Data asset lookup"):
        try:
            return client.data.get(name=ASSET_NAME, version=asset_version)
        except ResourceNotFoundError:
            return None


def _check_asset(
    asset, asset_version: str, curated_version: str, tags: dict[str, str], config: AzureConfig
) -> None:
    full_path = (
        f"azureml://subscriptions/{config.subscription_id}/resourcegroups/{config.resource_group}"
        f"/workspaces/{config.workspace_name}/datastores/{DATASTORE_NAME}"
        f"/paths/cmapss/{curated_version}/"
    )
    if (
        not isinstance(asset, Data)
        or asset.name != ASSET_NAME
        or asset.version != asset_version
        or asset.type != "uri_folder"
        or not isinstance(asset.path, str)
        or asset.path.removesuffix("/")
        not in {_asset_path(curated_version).removesuffix("/"), full_path.removesuffix("/")}
        or not isinstance(asset.tags, dict)
        or any(asset.tags.get(key) != value for key, value in tags.items())
    ):
        raise DataError(
            "Existing data asset version has a conflicting type, remote path or provenance tags. "
            "It will not be repointed or updated."
        )


def _metadata(
    client, account, config, asset_version, curated_version, tags, *, allow_missing: bool
) -> None:
    datastore = _get_datastore(client)
    if datastore is None:
        if not allow_missing:
            raise DataError(
                "Required datastore is missing (HTTP 404); publish it before verifying."
            )
    else:
        _check_datastore(datastore, account, config)
    asset = _get_asset(client, asset_version)
    if asset is None:
        if not allow_missing:
            raise DataError(
                "Required data asset is missing (HTTP 404); publish it before verifying."
            )
    else:
        _check_asset(asset, asset_version, curated_version, tags, config)


def _verify_blob(blob, expected: dict, properties=None) -> str:
    if properties is None:
        properties = blob.get_blob_properties(logging_enable=False)
    if properties.size != expected["size_bytes"]:
        raise DataError(
            "Remote blob size differs from the verified bundle; no overwrite is allowed."
        )
    if not isinstance(properties.etag, str) or not properties.etag:
        raise DataError("Remote blob has no ETag; a conditional integrity check is required.")
    condition = {"etag": properties.etag, "match_condition": MatchConditions.IfNotModified}
    download = blob.download_blob(**condition, max_concurrency=1, logging_enable=False)
    digest = hashlib.sha256()
    size = 0
    for chunk in download.chunks():
        size += len(chunk)
        if size > expected["size_bytes"]:
            raise DataError("Remote blob stream exceeds the verified bundle size.")
        digest.update(chunk)
    if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise DataError(
            "Remote blob checksum differs from the verified bundle; no overwrite is allowed."
        )
    blob.get_blob_properties(**condition, logging_enable=False)
    return properties.etag


def _remote_inventory(
    container, bundle: _Bundle, etags: dict[str, str], *, marker_optional: bool = False
) -> None:
    expected = {bundle.prefix + name for name in bundle.files}
    actual = set()
    for blob in container.list_blobs(name_starts_with=bundle.prefix, logging_enable=False):
        if blob.name not in expected or blob.name in actual:
            raise DataError(
                "Remote version prefix contains unexpected objects; nothing was deleted."
            )
        # List XML and HTTP property responses may differ in their ETag quoting.
        if blob.name in etags and (
            not isinstance(blob.etag, str)
            or blob.etag.removeprefix('"').removesuffix('"')
            != etags[blob.name].removeprefix('"').removesuffix('"')
        ):
            raise DataError(
                "Remote blob changed after checksum verification; registration is blocked."
            )
        actual.add(blob.name)
    missing = expected - actual
    if missing and not (marker_optional and missing == {bundle.prefix + bundle.marker}):
        raise DataError(
            "Remote version prefix inventory is incomplete; registration is not allowed."
        )


def _transfer(container, bundle: _Bundle, *, allow_write: bool) -> tuple[int, int, dict[str, str]]:
    uploaded = reused = 0
    etags = {}
    ordered = [name for name in sorted(bundle.files) if name != bundle.marker] + [bundle.marker]
    with _operation("Blob publication" if allow_write else "Read-only blob verification"):
        for name in ordered:
            if name == bundle.marker:
                _remote_inventory(container, bundle, etags, marker_optional=allow_write)
            expected = bundle.files[name]
            blob = container.get_blob_client(bundle.prefix + name)
            try:
                properties = blob.get_blob_properties(logging_enable=False)
            except ResourceNotFoundError:
                if not allow_write:
                    raise
                properties = None
            created = False
            if properties is None:
                local = bundle.root.joinpath(*name.split("/"))
                for path in (local, *local.parents):
                    if path.is_symlink() or path.is_junction():
                        raise DataError(
                            "Publication source became a link after local verification."
                        )
                # Recheck just before opening: no changed local bytes become intentional uploads.
                if fingerprint(local) != expected:
                    raise DataError("Publication source changed after local verification.")
                try:
                    with local.open("rb") as stream:
                        blob.upload_blob(
                            stream,
                            blob_type="BlockBlob",
                            length=expected["size_bytes"],
                            overwrite=False,
                            metadata={"sha256": expected["sha256"]},
                            max_concurrency=1,
                            logging_enable=False,
                        )
                    created = True
                except ResourceExistsError:
                    # Another publisher won the conditional create; its bytes must still match.
                    pass
            etags[bundle.prefix + name] = _verify_blob(blob, expected, properties)
            uploaded += int(created)
            reused += int(not created)
        _remote_inventory(container, bundle, etags)
    return uploaded, reused, etags


def _register(client, account, config, asset_version, curated_version, tags) -> bool:
    datastore = _get_datastore(client)
    if datastore is None:
        with _operation("Credential-free datastore registration"):
            try:
                client.datastores.create_or_update(
                    AzureBlobDatastore(
                        name=DATASTORE_NAME,
                        account_name=account,
                        container_name=CURATED_CONTAINER,
                        endpoint="core.windows.net",
                        protocol="https",
                        credentials=None,
                        description="Private C-MAPSS curated bundles; identity-based access.",
                    )
                )
            except ResourceExistsError:
                pass
        datastore = _get_datastore(client)
    if datastore is None:
        raise DataError("Datastore registration was not visible; no data asset was registered.")
    _check_datastore(datastore, account, config)
    asset = _get_asset(client, asset_version)
    created = False
    if asset is None:
        with _operation("Data asset registration"):
            try:
                client.data.create_or_update(
                    Data(
                        name=ASSET_NAME,
                        version=asset_version,
                        type="uri_folder",
                        path=_asset_path(curated_version),
                        tags=tags,
                        description=(
                            "NASA C-MAPSS FD001-FD004: original train/test splits preserved as "
                            "typed Parquet with separate supplied test RUL labels."
                        ),
                    )
                )
                created = True
            except ResourceExistsError:
                pass
        asset = _get_asset(client, asset_version)
    if asset is None:
        raise DataError("Registered data asset was not visible; retry read-only verification.")
    _check_asset(asset, asset_version, curated_version, tags, config)
    return created


def _publication(raw_root, curated_root, spec, config, *, allow_write: bool) -> dict:
    with _operation("Local bundle verification"):
        raw, curated, digest = _prepare(Path(raw_root), Path(curated_root), spec)
    curated_version = curated.root.name
    asset_version = _asset_version(digest)
    tags = _tags(spec, digest)
    with _operation("Publication"), ExitStack() as stack:
        with _operation("Explicit Azure authentication"):
            credential = create_credential(config)
            stack.callback(credential.close)
            client = create_ml_client(config, credential)
        with _operation("Workspace storage lookup"):
            account = _storage_account(client.workspaces.get(name=config.workspace_name), config)
        with _operation("Blob service initialization"):
            service = BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net",
                credential=credential,
                max_single_get_size=_CHUNK,
                max_chunk_get_size=_CHUNK,
                max_single_put_size=4 * _CHUNK,
                max_block_size=4 * _CHUNK,
                logging_enable=False,
            )
            stack.callback(service.close)
        containers = []
        with _operation("Private container lookup"):
            for name in (RAW_CONTAINER, CURATED_CONTAINER):
                container = service.get_container_client(name)
                properties = container.get_container_properties(logging_enable=False)
                if properties.public_access is not None:
                    raise DataError(
                        "Both approved containers must already be private. Apply the approved "
                        "Phase 2 provisioning; the publisher never creates or changes containers."
                    )
                containers.append(container)
        _metadata(
            client, account, config, asset_version, curated_version, tags, allow_missing=allow_write
        )
        raw_uploaded, raw_reused, raw_etags = _transfer(containers[0], raw, allow_write=allow_write)
        curated_uploaded, curated_reused, curated_etags = _transfer(
            containers[1], curated, allow_write=allow_write
        )
        # Re-list both prefixes and ETags before registration, including the completed raw prefix.
        with _operation("Final remote inventory verification"):
            _remote_inventory(containers[0], raw, raw_etags)
            _remote_inventory(containers[1], curated, curated_etags)
        registered = (
            _register(client, account, config, asset_version, curated_version, tags)
            if allow_write
            else False
        )
        _metadata(
            client, account, config, asset_version, curated_version, tags, allow_missing=False
        )
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "asset_name": ASSET_NAME,
        "asset_version": asset_version,
        "curated_version": curated_version,
        "asset_type": "uri_folder",
        "datastore_name": DATASTORE_NAME,
        "raw_archive_sha256": spec.archive_sha256,
        "curated_manifest_sha256": digest,
        "raw_objects_verified": len(raw.files),
        "curated_objects_verified": len(curated.files),
        "uploaded_objects": raw_uploaded + curated_uploaded,
        "reused_objects": raw_reused + curated_reused,
        "registered_new_version": registered,
        "workspace_name": config.workspace_name,
        "raw_container": RAW_CONTAINER,
        "curated_container": CURATED_CONTAINER,
        "jobs_submitted": False,
    }


def publish(raw_root: Path, curated_root: Path, spec: SourceSpec, config: AzureConfig) -> dict:
    """Verify local bundles, conditionally upload, then register/reuse one remote URI asset.

    Containers and scoped identity grants must already exist. Failures retain
    incomplete prefixes for a checksum-verified resume; nothing is overwritten.
    Receipts distinguish the lossless compact registry asset_version from the
    full-hex curated_version used by bundle folders, markers and storage paths.
    """
    return _publication(raw_root, curated_root, spec, config, allow_write=True)


def verify_publication(
    raw_root: Path, curated_root: Path, spec: SourceSpec, config: AzureConfig
) -> dict:
    """Read-only local/remote byte, inventory, datastore and asset-reference verification."""
    return _publication(raw_root, curated_root, spec, config, allow_write=False)
