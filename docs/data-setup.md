# Phase 2 — C-MAPSS data setup and decisions

## Scope and approved decisions

Use all **FD001–FD004** subsets from NASA C-MAPSS, separately. Preserve the complete original ZIP and all 14 members, including README/PDF. Curate typed Parquet, not a new ML split. Publish one versioned Azure ML `uri_folder` bundle so observation tables, supplied test labels, manifests and quality evidence cannot accidentally refer to different revisions.

The user approved two private containers in the **existing** workspace storage account, container-scoped Blob Data Contributor for the current publishing user, and one credential-free curated datastore. No storage account, compute, service, Fabric component, MLflow experiment or job is created. Processing runs locally.

## Authoritative source and provenance

- NASA catalog: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Catalog API: <https://data.nasa.gov/api/3/action/package_show?id=cmapss-jet-engine-simulated-data>
- Official archive URL: <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- NASA repository/context: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Citation: A. Saxena, K. Goebel, D. Simon and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” PHM08, 2008.

The official GET endpoint redirects to a NASA-controlled signed S3 download. This is NASA's distribution mechanism, **not an AWS deployment for EPM**. HEAD returned 403 because the issued link is for GET; do not interpret a HEAD failure alone as unavailable data. A normal GET downloaded the 12,425,978-byte ZIP and passed CRC inspection. No third-party mirror was used.

The observed SHA-256 is pinned in `config\cmapss-source.json`, along with every member's hash/size and the verified row/unit counts. This fingerprint is locally recorded integrity evidence, not a separately published NASA digital signature. A changed upstream archive is rejected until explicitly reviewed; hashes are never silently refreshed.

Signed redirect query parameters are transient and must not enter configuration, receipts, logs or manifests. Only the stable public catalog/download URLs are recorded. NASA's catalog says **License not specified**; PCoE requests acknowledgement of NASA and contributors. Public availability is not a claim of an unrestricted license; retain attribution and review terms before external redistribution.

## Data interpretation and justified transformations

Each observation file has **26 numeric columns**: `unit_id`, `cycle`, three operating settings, and `sensor_01` through `sensor_21`. The first two become int32; the other 24 remain float64. No float32 downcast, rounding, scaling, clipping, denoising, imputation or sensor removal is applied. Whitespace is a delimiter, including repeated/trailing whitespace—not evidence of extra missing sensor columns.

The separate supplied RUL vector is aligned by line order to test units 1..N, after verifying contiguous IDs and matching cardinality. Its curated table contains `unit_id` and `rul` as int32. RUL values are preserved, not recomputed or joined into observation features. **Training RUL/failure targets are not derived in Phase 2.**

Engine identity is `(subset, split, unit_id)`. Numeric IDs restart in each file; overlap across train/test or subsets is expected and is not itself leakage. Validate schema, counts, finite values, integer domains, duplicate records/conflicting keys, contiguous unit blocks, cycle start/order/continuity, label alignment and exact copied test trajectories/training prefixes. Statistical summaries are quality evidence only. Negative operating settings, simulation noise and zero-variance channels are retained.

Two source-documentation discrepancies are explicit exceptions:

1. FD004 **files** contain 249 training and 248 test engines; NASA's README/catalog reverse those counts. Preserve filenames and records; do not move rows to match the prose.
2. The last sensor label in that prose says sensor 26, but 26 total columns minus ID/cycle/three settings means **21 sensors**, confirmed in all eight observation files.

README bytes are Windows-1252; numeric files are ASCII. Original files are not transcoded. See the generated [quality report](data-quality-report.md).

## Layout and integrity

```text
data\raw\cmapss\<archive-sha256>\
  CMAPSSData.zip               Original download, byte-for-byte
  files\                      All 14 original members, byte-for-byte
  raw-manifest.json            Stable source URLs and member fingerprints

data\curated\cmapss\sha256-<manifest-sha256>\
  FD001\train.parquet
  FD001\test.parquet
  FD001\test_rul.parquet       Supplied labels only
  FD002\...                   Same three-file layout
  FD003\...
  FD004\...
  data-quality.json            Full deterministic rule/statistical evidence
  manifest.json               Recipe/schema/writer/source/content fingerprints
  _SUCCESS.json               Completion marker bound to the manifest
```

The asset root is a **bundle**, not one homogeneous Parquet table. Select an explicit file such as `FD001\train.parquet`; do not read every Parquet file into one frame or mix supplied RUL rows with observations.

Data directories are ignored by Git. Builds use isolated staging directories and finalize only validated content. An existing version is verified and reused, never overwritten. Corruption, unexpected files, invalid paths or links fail closed. Raw original bytes and train/test order are preserved.

Parquet uses pinned PyArrow 25.0.1, ZSTD level 3 and fixed writer settings. The manifest includes the source specification hash, raw archive hash, recipe version, writer version and file hashes. No timestamps, local absolute paths or Azure identifiers belong in these immutable bundles. Operational timestamps live only in ignored `.azure` receipts.

### Azure version identifiers

The local/blob version remains `sha256-<64 hexadecimal digits>`. Live Azure validation exposed two limits: lookup allowed at most 50 characters, while the registration backend enforced **30**. The owner explicitly approved a shorter deterministic registry alias:

```text
d- + lowercase Base32(first 16 bytes of the SHA-256 manifest digest), without '=' padding
```

This is **28 characters** and serves only as a 128-bit selector. **All integrity checks, tags, manifests and storage paths retain the complete SHA-256.** An existing alias with any different full digest, path or source tag fails closed; it is never overwritten or repointed. Explicit collision tests cover this case.

`--version` / PowerShell `-Version` select the full-hex **curated folder version**. The publication receipt separately returns the shorter Azure `asset_version`; use that value in `azureml:<name>:<asset_version>` references. Never use a mutable `latest` alias for reproducibility.

## Reproduce locally

Run from the project root with the Phase 1 tooling and locked Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .

powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-DataPipeline.ps1 -Action Acquire
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-DataPipeline.ps1 -Action Curate
```

Acquire downloads with validated HTTPS redirects restricted to the reviewed publisher hosts. If a verified raw bundle already exists, it only verifies/reuses it. Direct Python transfers timed out on this host during initial setup; the successful official GET download was imported through the checksum-locked offline path. No TLS verification was disabled or mirror substituted.

If needed, download the **same official URL** with the system curl and import it. Use a fresh temporary path; do not overwrite an existing original archive:

```powershell
$archive = Join-Path $env:TEMP ('epm-cmapss-' + [guid]::NewGuid().ToString() + '.zip')
curl.exe --fail --location --proto '=https' --proto-redir '=https' --max-time 180 --output $archive 'https://data.nasa.gov/docs/legacy/CMAPSSData.zip'
if ($LASTEXITCODE -ne 0) { throw 'Official download failed.' }
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-DataPipeline.ps1 -Action Acquire -ArchivePath $archive
if ($LASTEXITCODE -eq 0) { Remove-Item -LiteralPath $archive }
```

Do not print curl's effective signed URL. The imported archive must match the pinned size/hash and every member; an arbitrary local ZIP is not accepted. `Curate` prints the full-hex curated version and writes `.azure\data-preparation.json` plus `.azure\data-quality-summary.md`. Failures return nonzero; no failed curated version is finalized or registered.

## Prepare scoped Azure access

Select the existing Phase 1 subscription through `az account set`; no real subscription/principal ID is hardcoded in source. The initialization script resolves the workspace-associated storage and the current signed-in user, verifies keyless/private-blob settings, and targets that existing account only.

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-DataInfrastructure.ps1
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Initialize-DataStorage.ps1 -Action WhatIf
# Only after approving these exact container-level grants:
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Initialize-DataStorage.ps1 -Action Deploy -ApproveAccessChanges
```

The template creates `epm-cmapss-raw` and `epm-cmapss-curated`, with no public access, and grants only Blob Data Contributor at those containers. A pre-existing subscription Owner role is not a Blob data-plane permission and is not created by this project. Publisher object IDs are non-secret runtime parameters with no source default; temporary parameter files are removed. No role is granted to the compute identity in this phase.

## Publish and verify

Verified registered reference:

```text
azureml:epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm
```

The full curated folder version remains `sha256-ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640`. Both names are recorded in publication receipts.

```powershell
$prepared = Get-Content .\.azure\data-preparation.json -Raw | ConvertFrom-Json
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-DataPipeline.ps1 -Action Publish -Version $prepared.curated_version -ApproveAzureWrites
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-DataPipeline.ps1 -Action Verify -Version $prepared.curated_version
```

Publishing uses explicit Entra credentials from Phase 1—no SAS, connection strings, account keys or credential fallback. It verifies local bytes before mutation, requires the existing private containers, and creates/reuses the credential-free `epm_cmapss_curated` datastore. Raw and curated prefixes are distinct and content-addressed.

Every upload is conditional/no-overwrite. Existing blobs are streamed and SHA-256 checked, not trusted based on metadata alone; ETag conditions detect concurrent changes. Remote prefix inventories must exactly match. The raw manifest and curated completion marker are written last. Only then is `epm-cmapss-curated` registered as a `uri_folder` data asset referencing the **remote** curated datastore path. A mismatched existing datastore/asset/version aborts rather than repointing it. No SDK local-path auto-upload into the default datastore is used.

A successful rerun reuses matching objects and the same asset version. `Verify` is strictly read-only and must never upload, create a datastore or register a version. Sanitized receipts are `.azure\data-publication.json` and `.azure\data-verification.json`; no full subscription/principal IDs, secrets or signed URLs are recorded there.

**Immutability limit:** a versioned data asset is a reference, not a WORM policy for underlying blobs. Application no-overwrite rules plus checksum verification protect reproducibility and detect external mutation; a separately privileged actor can still change/delete storage. No legal hold or locked retention policy was added. Failed partial uploads are not registered and are never automatically deleted; retries may resume only matching bytes. Stop and investigate corruption rather than enabling keys or overwriting data.

## Tests, costs and phase boundary

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --disable-warnings -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pip check
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-DataInfrastructure.ps1
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Security.ps1
```

The Windows real-symlink test may skip without OS privilege; simulated link/junction rejection is tested. Do not change machine privilege/policy merely to remove that skip. Dependency locks are validated for Windows/Python 3.12, not a future Linux training image.

Costs are usage-based Blob storage, writes, verification reads and possible egress, in the existing account. No CPU cluster is started; foundation storage/telemetry costs still apply. No new service is added. Historical versions consume storage until separately reviewed cleanup; no retention/retraining workflow is implemented.

**Phase 2 checkpoint boundary:** when this data workflow was completed, Phase 3 had not yet been authorized; no model, feature windowing, training targets/scalers, PyTorch, experiment, model registry workflow, endpoint, monitoring, retraining or CI/CD was included in Phase 2. Later phase outcomes are summarized in the [final lifecycle status](architecture.md).

Reference: [Azure ML v2 data assets and supported types](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2).
