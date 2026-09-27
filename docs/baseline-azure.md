# Azure ML v2 baseline: publication and one approved CPU job

The `epm_platform.baseline.azure` module provides four explicit operations:
`publish`, `submit`, `status`, and `download`. It does not provision infrastructure,
register environments, change RBAC, publish Phase 2 data again, call MLflow, register
models, or create endpoints. The experiment name `epm-baseline-rul` is only Azure ML
job grouping. All examples below use the existing project virtual environment.

## Prerequisites and boundaries

- Use the existing process-environment `AzureConfig` settings. Authentication uses
  exactly the configured Azure CLI or managed-identity credential; no dotenv loading,
  keys, SAS tokens, or fallback credential chains are used.
- The curated registration `epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm` must exist,
  with manifest tag
  `ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640` and its corresponding
  full-hash curated datastore path. It is checked before publication writes and
  submission. No raw or curated objects are reuploaded.
- The existing credential-free `epm_cmapss_curated` datastore must still reference
  the private `epm-cmapss-curated` container in the workspace storage account.
  Mismatches or missing resources stop the operation; nothing is repointed or created.
- Use the approved Microsoft public curated environment reference
  `azureml://registries/azureml/environments/sklearn-1.5/versions/54` directly.
  The local, read-only guard permits only that registry/name/version. It does not
  perform a workspace environment lookup or register/build an environment.
  `--environment-version` defaults to `54`; every other value, including `latest`,
  is rejected before authentication.
- The existing `AllowInternetOutbound` managed network would require Premium ACR
  for the proposed workspace-registry attachment. The user-approved alternative
  avoids that attachment and its approximately $50/month Premium tier: consume the
  existing curated image without creating any workspace image/environment variants.
  The curated image uses Ubuntu 20.04/Python 3.10; the separately maintained
  `scripts\run-baseline.sh` bootstrap creates an isolated Python 3.12.10/pip 25.2
  runtime inside the job and installs only the pinned NumPy, SciPy, PyArrow and
  `xgboost-cpu` requirements. Existing curated-image packages are not the training
  runtime. Outbound runtime downloads are part of the approved job execution.
- Existing `cpu-dev` must be a successfully provisioned dedicated
  `Standard_D2s_v3` AmlCompute cluster: minimum zero, maximum one, 300-second idle
  scale-down, no public node IP or public SSH, and a system-assigned identity.
  ARM-level Linux/local-auth validation and scoped identity grants remain the
  infrastructure procedure's responsibility. This module never repairs compute.
- The compute identity needs the separately approved input/output access. Human
  publication and submission permissions are not substitutes for compute access.

## Reproduce local features and scoped access

Install the project development lock (includes CPU-only baseline dependencies):

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Generate features without any Azure operation:

```powershell
@'
from pathlib import Path
from epm_platform.features.pipeline import build_features
result = build_features(
    Path('data') / 'curated' / 'cmapss' / 'sha256-ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640',
    Path('data') / 'ml-ready' / 'cmapss',
    Path('config') / 'features.json',
    Path('config') / 'cmapss-source.json',
)
print(result.version)
'@ | .\.venv\Scripts\python.exe -
```

Use the Phase 1 environment loader before direct Python Azure commands, or use the PowerShell wrappers:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
. .\scripts\Initialize-Environment.ps1
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Initialize-Training.ps1 -Action WhatIf
# Only after approving the five container-scoped runtime permissions:
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Initialize-Training.ps1 -Action Deploy -ApproveCosts
```

The final training template has **no registry resource or registry attachment**. It uses only existing workspace/compute/storage references and scoped data permissions. Do not restore the abandoned Basic-registry proposal. No Premium ACR is required for the selected public curated-image path.

## Publish the verified feature bundle

From the repository root, after generating the local features:

```powershell
.\.venv\Scripts\python.exe -m epm_platform.baseline.azure publish --data "data\ml-ready\cmapss\sha256-2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
```

`verify_features` checks the local integrity and feature semantics. The publisher
then checks the approved source, 35 predictor columns, target `rul`, manifest bytes,
and exact local inventory. Local folder names may differ from the original hash
folder, because the verified manifest digest determines the remote identity.

Exactly **seven** objects are published: `train.parquet`, `validation.parquet`,
`test.parquet`, `splits.json`, `feature-summary.json`, `manifest.json`, and
`_SUCCESS.json`. Their prefix is
`ml-ready/cmapss/sha256-<full-manifest-sha256>/` in the existing curated container.

The publisher reuses the Phase 2 `_Bundle`, `_transfer`, `_remote_inventory`,
`_check_datastore`, `_storage_account`, and compact `_asset_version` contracts.
Blob writes are conditional and never overwrite. Existing bytes must pass size,
SHA-256, and ETag checks; unexpected remote objects stop registration. The completion
marker is uploaded last. Repeating publication reuses verified objects and the
matching data registration; it does not create another version.

The single data asset name is `epm-cmapss-ml-ready`, with a `uri_folder` path and a
28-character version alias derived from the manifest digest. The alias is only a
selector: the **full digest, exact path, and source provenance tags** must all match.
A conflicting existing registration is never intentionally updated. Azure data
registration uses SDK `create_or_update` only after a missing-version lookup; it
is not a storage immutability policy or a compare-and-swap registry guarantee.

Successful publication writes `.azure\ml-ready-publication.json`, containing the
asset selector, full digest, source selectors, prefix, and verified/uploaded/reused
object counts. Receipts exclude subscription IDs, full cloud resource IDs, credentials,
and service response text.

## Submit only after explicit cost approval

Read the asset selector and full digest from the publication receipt. The public
curated environment is pinned to version **54**; do not substitute `latest` or a
workspace environment. The explicit flag below may be omitted because `54` is the
default. Both the bootstrap script and runtime lock file must exist locally before
submission; this module copies but does not generate or execute them locally.

```powershell
$publication = Get-Content .azure\ml-ready-publication.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m epm_platform.baseline.azure submit --asset-version $publication.asset_version --manifest-sha256 $publication.manifest_sha256 --environment-version 54 --approve-costs
```

Without `--approve-costs`, submission fails before creating credentials or contacting
Azure. The named feature registration is checked against its full manifest digest,
source and exact URI before the job is created. Submission does not repeat the
publisher's full remote byte download: the training entry point independently
verifies its downloaded feature bundle before fitting.

The submitted command is fixed; no user text is interpolated into shell commands:

```text
bash config/run-baseline.sh --data ${{inputs.ml_ready}} --config config/baseline.json --output ${{outputs.baseline}}
```

Job contract:

| Setting | Value |
| --- | --- |
| Name | `epm-baseline-<12 random hexadecimal characters>` |
| Experiment | `epm-baseline-rul` |
| Compute / instance count | Existing `cpu-dev` / 1 |
| Identity | Managed identity, using the compute system-assigned identity |
| Execution limit | `CommandJobLimits(timeout=3600)` |
| Input | Named `ml_ready`: exact `azureml:epm-cmapss-ml-ready:<version>`, `uri_folder`, `download` |
| Output | `uri_folder`, `upload`, named `baseline` |
| Output path | `azureml://datastores/workspaceblobstore/paths/baseline/<job-name>/` |
| Environment | `azureml://registries/azureml/environments/sklearn-1.5/versions/54` |
| Process environment | `PYTHONPATH=./src`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `PYTHONHASHSEED=42` |
| Tags | Full feature/source hashes, source asset selectors, code bundle hash |

The bootstrap uses the isolated runtime to invoke the unchanged training CLI:
`python -m epm_platform.baseline.training --data ... --config ... --output ...`.
Runtime creation and dependency downloads/installations execute inside the same
3600-second command-job limit, reducing the time available for training. There is
no custom image build or workspace environment registration. The 60-minute limit
is not a monetary spending cap; service provisioning and idle-scale-down can incur
additional time/cost.

### Restricted code staging

Only these reviewed Python modules, configurations and two explicit bootstrap
inputs are copied:

- `src\epm_platform\__init__.py`
- `src\epm_platform\features\__init__.py`, `pipeline.py`
- `src\epm_platform\baseline\__init__.py`, `training.py`
- `src\epm_platform\data\__init__.py`, `curation.py`, `errors.py`, `manifest.py`,
  `source.py`, `spec.py`, `validation.py`
- `config\baseline.json`, `features.json`, `cmapss-source.json`
- `environments\baseline\requirements.txt` → `config\runtime-requirements.txt`
- `scripts\run-baseline.sh` → `config\run-baseline.sh`

Only those source paths supply the renamed bootstrap files; similarly named files
already under `config` are ignored. The bootstrap and requirements bytes and their
staged destination names are included in the code hash. Missing either file blocks
submission, and changes to either produce a different staging hash.

The staged path is `.azure\job-code\<sha256-of-paths-sizes-and-content-hashes>`.
**The repository root is never supplied as job code.** Source snapshots are bounded
at 1 MiB per file and 4 MiB total. Links/junctions, missing files, and any unexpected
or altered entries in an existing staging directory are rejected. The allowlist
excludes cloud clients, publishers, the Azure submission module, `.azure` contents,
all other environment files and scripts, raw/curated/feature datasets, credentials,
keys, caches, notebooks, and arbitrary additional Python modules. The staged
training runtime does not import Azure SDKs. The `.sh` and `.txt` additions are
exact source-to-destination mappings, not extension-wide allowlists. Changes to
bootstrap inputs must retain the approved runtime pins and policy.

### Uncertain submission recovery

Before the one `jobs.create_or_update` call, `.azure\baseline-job.json` records the
unique job name, `SubmissionPending`, asset/environment references, code hash and
manifest digest. The receipt is replaced atomically with the returned safe status.
A failed or ambiguous submit response leaves `SubmissionUnknown` and the same job
name. There is **no automatic resubmission**. Use `status` on that recorded name and
inspect the workspace before deciding whether a new submission is justified.
Every explicit new submit invocation creates a new unique name and can incur cost.

## Status and completed output download

```powershell
$job = Get-Content .azure\baseline-job.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m epm_platform.baseline.azure status --job-name $job.job_name
.\.venv\Scripts\python.exe -m epm_platform.baseline.azure download --job-name $job.job_name --destination "artifacts\baseline\$($job.job_name)"
```

`status` is read-only and returns only the verified job name, a recognized lifecycle
status (otherwise `Unknown`), and a fixed safe code for failed/canceled jobs. Raw
service error objects, stack traces, URLs and response bodies are never returned.
CLI Azure failures report only bounded HTTP status information, not service text;
SDK logging/progress output is suppressed.

`download` requires the exact job to be `Completed` and a new or empty local directory.
The Azure ML named-output downloader can return without downloading explicit datastore
outputs, so retrieval uses the Blob SDK directly. The job's `baseline` output must be
`uri_folder` at exactly
`azureml://datastores/workspaceblobstore/paths/baseline/<job-name>/`. The named and
default workspace datastores must agree on the private container and workspace
storage account; no credentials are retrieved or datastore metadata changed.

The exact prefix must contain eight files: `artifact-manifest.json`, `evaluation.md`,
`feature-importance.json`, `metrics.json`, `model.json`, `predictions_test.parquet`,
`predictions_validation.parquet`, and `run-metadata.json`. Downloads use bounded
streaming and ETag conditions into exclusive staging files. The manifest schema,
exact seven content entries, sizes and SHA-256 checksums are verified, followed by
a final remote inventory/ETag check. Limits are 1 MiB for the manifest, 64 MiB per
artifact, and 256 MiB total. The verified files are exclusively copied directly into
the destination with the manifest last; failed downloads remove owned partial files.
Existing destination files are never overwritten, and empty or incomplete downloads
never report success. The result includes `objects_verified: 8` and the artifact
manifest SHA-256. Logs and other job outputs are not fetched.

Downloaded outputs still require the operator's artifact/metric inspection. Neither
status nor download submits a job or changes the stored submission receipt.

## Offline verification

No tests require credentials, live cloud resources, or package installation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\baseline\test_azure.py -q --disable-warnings
.\.venv\Scripts\python.exe -m ruff check src\epm_platform\baseline\azure.py tests\baseline\test_azure.py
```

Tests exercise real SDK job entities with mocked clients, marker-last conditional
uploads and byte verification, metadata/source conflicts, constrained code staging,
CPU/identity/timeout/input/output policy, explicit approval, safe CLI failures,
ambiguous-submit receipts, and completed-only output retrieval. Disposable test
artifacts are created within the project and removed after the tests.
