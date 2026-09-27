# EPM Platform

**Enterprise Predictive Maintenance & Failure Prediction Platform** — a modular Python/Azure Machine Learning project for future industrial equipment failure-risk prediction.

**GitHub repository:** <https://github.com/Tanveer-F/EPM-Platform> (`main`)

## Implemented phases

- **Phase 1 — complete:** Azure foundation, authentication, configuration and verification. See [Phase 1 evidence](docs/phase-1-validation.md).
- **Phase 2 — complete:** authoritative NASA C-MAPSS raw preservation, validated Parquet curation, remote checksum verification and versioned Azure ML publication. Registered asset: `epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm`. Repeat publication changed no objects or asset version. See [data setup and decisions](docs/data-setup.md), the [data-quality report](docs/data-quality-report.md), and [Phase 2 evidence](docs/phase-2-validation.md).

- **Phases 3 + 4 — complete:** 35 causal features, engine-disjoint validation, registered ML-ready data and one classical XGBoost RUL baseline. See [features](docs/features.md), [training design](docs/baseline-design.md), [remote execution](docs/baseline-azure.md), and [actual baseline results](docs/baseline-results.md).

- **Phases 5 + 6 — complete:** compact CPU PyTorch MLP trained remotely with basic MLflow parameters, metrics and checkpoint/artifact tracking. Same data/splits/evaluation as XGBoost; lower test MAE but worse RMSE and asymmetric risk score. See [PyTorch results](docs/pytorch-results.md), [runtime setup](docs/pytorch-runtime.md), and [model design](docs/pytorch-design.md).

- **Phases 7 + 8 — complete:** selected and registered the XGBoost baseline as Azure ML model `epm-cmapss-rul-xgboost:1`, preserving Azure ML job lineage and verifying all downloaded artifacts. See [model selection and registry evidence](docs/model-registry.md).

- **Phases 9 + 10 — local fallback complete; Azure endpoint blocked:** implemented model-verified XGBoost serving, a loopback API, input validation and aggregate drift/latency monitoring. The Azure Activity Log shows an asynchronous `SubscriptionNotRegistered` failure but reports the required provider as `[N/A]`; no speculative provider registration or retry was made. See [deployment and monitoring status](docs/deployment-monitoring.md).

- **Phases 11 + 12 — complete:** implemented manual drift/performance triggers, explicit-cost-approved Azure ML retraining, artifact/lineage checks, a strict model-promotion gate, and GitHub CI. No schedule or CI Azure credential is configured. See [retraining and CI/CD](docs/retraining-cicd.md).
- **Phase 13 — complete:** final lifecycle, security, cost and documentation review; see the [final architecture and project summary](docs/architecture.md).
- **Phase 14 — partially validated:** connected and pushed the project to the supplied GitHub repository; hosted CI passes. One explicitly validation-only Azure retraining job completed and was correctly rejected with unchanged incumbent metrics. Azure endpoint activation remains blocked until Support identifies the provider namespace omitted from the Activity Log; no endpoint is retained.

**Current boundary:** there is no retained production endpoint, scheduled Azure retraining, automatic cloud promotion/deployment, Fabric or GPU infrastructure. The verified inference path is local loopback; real Azure inference and endpoint monitoring remain unavailable pending the exact provider namespace.

## Architecture

See the [final end-to-end architecture diagram](docs/architecture.md).

This is an **authenticated public-endpoint DEVELOPMENT foundation**, explicitly selected because no corporate VNet/VPN is available. It is not a private production environment. Storage keys and anonymous blobs are disabled. Compute has no node public IP or public SSH. Managed networking governs compute traffic; it does not make workstation access private.

[Architecture decisions and production prerequisites](docs/architecture-decisions.md) document network access, permissions, dependencies and deliberate exclusions.

### Azure resources

| Resource | Development configuration |
|---|---|
| Resource group | `rg-epm-dev-eastus`; dedicated lifecycle and tagging boundary |
| Azure ML workspace | `mlw-epm-dev-eastus`; standard workspace, SDK/CLI v2, system-assigned identity |
| Storage account | Deterministic globally unique name; `StorageV2`, `Standard_LRS`, HTTPS/TLS 1.2, no shared keys/public blobs |
| Key Vault | Deterministic globally unique name; Standard, RBAC, soft delete, 90-day retention, purge protection |
| Workspace telemetry dependencies | `appi-epm-dev-eastus` → `log-epm-dev-eastus`; Entra auth, 30-day retention, 1 GB/day ingestion safeguard |
| Platform-default action group | Azure-created `Application Insights Smart Detection`, explicitly managed as **disabled with no recipients** |
| CPU cluster | `cpu-dev`; `Standard_D2s_v3`, Linux, dedicated, minimum **0**, maximum **1**, five-minute idle scale-down |
| Managed identity | Separate system-assigned identities attached to workspace and compute; no client secrets |

Application Insights and Log Analytics were explicitly approved after Azure ML rejected workspace creation without the dependency linkage. They are workspace dependencies only: no application instrumentation, alerts, drift/performance monitoring or retraining is configured. Telemetry ingestion/retention can incur charges; the ingestion safeguard is not a guaranteed budget cap.

Bicep does not explicitly create ACR, VPN, firewall, private endpoints, a GPU, or a compute instance. Azure ML may manage supporting networking in its own managed resource group; deployment verification inventories what the service actually creates.

### Cost and lifecycle

Compute scales to zero, but **the foundation is not guaranteed to cost zero**: storage, operations, retained blobs and service-managed networking can incur charges. The public East US Linux VM estimate for `Standard_D2s_v3` was **USD 0.096/hour while allocated** on 2026-09-26, excluding disks/networking/taxes and contract differences. Phase 14 ran one validation-only Azure ML job on this existing CPU compute; its exact charge is not available here. Zero-node provisioning does not guarantee later regional capacity.

Key Vault purge protection cannot be disabled after activation; deleted vault names remain reserved during retention. Managed networking cannot be disabled after enablement. No automated delete/purge script is provided. Review dependencies, retained data and Azure ML-managed resources before any separately approved cleanup.

## Project structure

```text
infra\                       Bicep modules, development parameters and IaC validation dependency
src\epm_platform\            Configuration, authentication and foundation verification
src\epm_platform\data\       Acquisition, validation, curation, reporting and keyless publication
src\epm_platform\features\   Causal features, retrospective labels and engine-disjoint split
src\epm_platform\baseline\   Cloud-agnostic XGBoost training; separate Azure job orchestration
src\epm_platform\deep_learning\  CPU PyTorch training; separate MLflow/Azure adapters
src\epm_platform\serving\  Shared Azure ML scoring, local HTTP API and payload-free drift summaries
src\epm_platform\retraining.py  Drift/performance triggers and acceptance-gated baseline orchestration
environments\baseline\      Pinned CPU-only XGBoost runtime requirements
environments\pytorch\       Isolated CPU PyTorch/MLflow Windows and Linux locks
scripts\                     Explicit setup, deployment and data-pipeline commands
config\                      Environment example and reviewed NASA source fingerprint contract
.github\workflows\           GitHub CI; no Azure credentials or paid-job steps
tests\                       Isolated foundation and data-pipeline unit tests
data\raw\cmapss\             Ignored, checksum-locked original ZIP and original members
data\curated\cmapss\         Ignored, hash-versioned Parquet bundles and quality reports
docs\                        Decisions, setup and validation evidence
docs\architecture.md         Final architecture diagram, lifecycle and security/cost boundaries
.azure\                      Ignored local generated templates, previews and reports
.venv\                       Ignored Python application environment
.tools\                      Ignored isolated infrastructure tooling
```

This workspace is folder-backed but now has a Git repository on `main` connected to <https://github.com/Tanveer-F/EPM-Platform>. The initial project and Phase 14 documentation/CI fixes are pushed. Review `git status` before future commits. Never add `.env.local`, `.azure`, datasets, credentials or generated ML artifacts.

## Prerequisites

- Windows PowerShell 5.1 or later; commands below use Windows paths.
- Git, Python **3.12**, Azure CLI, Bicep CLI and the **`ml` v2 extension**.
- An enabled Azure subscription and Entra sign-in authorized to deploy at subscription scope (the template creates its resource group). Workspace initialization also needs authorized identity/role setup. Day-to-day read-only verification does not need Owner.
- Resource providers: `Microsoft.MachineLearningServices`, `Microsoft.Storage`, `Microsoft.KeyVault`, `Microsoft.ManagedIdentity`, `Microsoft.Network`, `Microsoft.Compute`, `Microsoft.Insights`, `Microsoft.OperationalInsights`.

```powershell
# Install only missing tools. Keep Python 3.14 or other existing runtimes intact.
py install 3.12
az bicep install
az extension add --name ml
az login
# Select the intended subscription yourself; never put its real ID in source files.
az account set --subscription "<your subscription name>"
```

If a required provider is unregistered, an authorized administrator must register it explicitly, for example `az provider register --namespace Microsoft.Network --wait`. Provider registration is subscription configuration, not a reason to grant broad roles to every developer.

## Local environment and configuration

See [Setup and operations](docs/setup.md) for exact dependency installation and validation commands.

Configuration precedence: existing process environment, optional ignored `.env.local`, then `config\.env.example`. Only documented `KEY=value` entries are accepted; the loader does **not** execute configuration as code. Leave `AZURE_SUBSCRIPTION_ID` empty in files: initialization obtains the current Azure CLI subscription in memory. No secret, token or real subscription ID is stored in tracked source.

`EPM_AUTH_MODE=azure-cli` uses the signed-in user's Entra identity. The Python client also supports `managed-identity` with runtime-injected subscription configuration; it does not use credentials embedded in files. Managed-identity authentication must run on an Azure host with that identity and suitable RBAC, not on an ordinary laptop.

If local script execution is restricted, the approved invocation below uses **process-only `RemoteSigned`**, not a persistent policy change:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Prerequisites.ps1
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Infrastructure.ps1
```

## Preview, deploy and verify

Bicep is the source of truth. Edit `infra\main.dev.bicepparam` for approved naming/region changes and keep environment configuration aligned. This phase supports **development only**; do not treat a public-network flag as a production security profile.

```powershell
# Read-only ARM validation and preview; creates no Azure resources.
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Deploy-Infrastructure.ps1 -Action WhatIf

# Only after reviewing the preview, permissions, public-development exception and charges:
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Deploy-Infrastructure.ps1 -Action Deploy -ApproveCosts

# Read-only CLI, raw ARM, RBAC and Python SDK verification:
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Verify-Foundation.ps1
```

Deployment is incremental, refuses planned deletions and never changes global CLI defaults. Every deployment targets the selected subscription explicitly. Generated previews/outputs remain under ignored `.azure`.

For manual inspection, use Azure ML Studio (<https://ml.azure.com>) and select the workspace, or run the CLI commands in [Setup and operations](docs/setup.md). A successfully provisioned cluster with **zero current nodes is healthy when idle**, not an error.

## Phase 2 data workflow

NASA's original archive and all 14 original members are preserved unchanged. All four subsets retain separate train/test tables and supplied test-RUL labels. Only whitespace parsing, explicit names/types and RUL row-to-unit alignment are applied; no scaling, imputation, dropped sensors or derived training targets.

Two private dataset containers reuse the existing storage account; `infra\data-access.bicep` grants the publishing user Blob Data Contributor **only at those containers**. One credential-free datastore and one versioned `uri_folder` data asset reference the verified curated bundle. No compute is started for this workflow.

See [the reproducible commands, version semantics and access decisions](docs/data-setup.md), [quality findings](docs/data-quality-report.md), and [Phase 2 acceptance evidence](docs/phase-2-validation.md).

Data-asset versions pin a reference, not a WORM lock on blobs. This pipeline never overwrites existing versioned objects and verifies their bytes; privileged external mutation is detected, not made impossible. Production networking and future ML phases remain separate approvals.

Phase 14 is the final project phase. Azure endpoint activation remains blocked on Azure Support identifying the exact missing provider namespace; no further phase is planned.
