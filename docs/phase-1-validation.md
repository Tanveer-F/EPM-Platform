# Phase 1 validation evidence

## Status

**Phase 1 complete — approved DEVELOPMENT foundation, verified 2026-09-26.** This is not a private production network. Final combined CLI/ARM/SDK verification passed 20 checks at `2026-09-26T07:47:30Z`; subsequent reads confirmed zero allocated/target CPU nodes and an empty job history. At that Phase 1 completion checkpoint, Phase 2 had not started.

## Initial environment inspection — 2026-09-26

- Project folder initially empty and not a Git repository.
- Git: 2.53.0.windows.4.
- Azure CLI: 2.90.0.
- Bicep: 0.47.16.
- Missing Azure ML CLI extension installed: `ml` 2.45.0.
- Python 3.14.4 initially installed; approved side-by-side Python 3.12.10 installed. Existing Python retained.
- Azure CLI authentication succeeded; user selected the current subscription and `eastus`. No real identifiers recorded here.
- `Standard_D2s_v3` is in the Azure ML region size list. DSv3 dedicated quota: 6 vCPUs; usage: 0. No quota increase requested.
- Project resource group did not exist at inspection.
- No corporate VNet/VPN is available; public authenticated development network exception explicitly approved.
- Local script execution was restricted. User approved process-scoped `RemoteSigned`, without a persistent policy change.
- Required provider `Microsoft.Network` was not registered at initial preflight. It was registered with explicit deployment approval; final prerequisite checks passed for every required provider.
- Application dependencies are locked separately from the isolated infrastructure tool environment. No real subscription ID was found in source/configuration/documentation during the source-hygiene check.

## Acceptance checklist

| Check | Evidence/status |
|---|---|
| Bicep compilation | Final templates and development parameters compiled successfully, exit 0 |
| Infrastructure contract tests | 39 checks passed; includes phase boundary, security/cost contracts and all PowerShell syntax |
| Python unit tests/lint | 92 passed; Ruff and pip check clean. SDK-inaccessible fields explicitly delegated to ARM; regression test with actual SDK entity objects passes |
| IaC security scan | Checkov 3.3.19: 7 passed, 0 failed, 6 reviewed exceptions; final run without scanner errors, exit 0 |
| ARM validation and WhatIf | Succeeded before deployment; subsequent approved incremental deployments retained existing resources. Final preview has no new resources or resource deletions; service-managed differences are explained below |
| Deployment approval | User explicitly acknowledged costs and irreversible settings; authorized provider registration, deployment and a brief one-node/no-job availability test |
| Resource provisioning | Passed. Initial minimal attempt failed with Missing dependent resources; RG/storage/vault retained. Owner-approved Application Insights + Log Analytics correction and subsequent repeat deployment both succeeded |
| CLI and SDK workspace access | Live CLI/ARM/SDK verification passed after corrected deployment |
| CPU cluster availability | Passed: one idle CPU node verified without a job; zero allocated and zero target nodes confirmed after cleanup. The first test exposed a transient-state parser issue; cleanup succeeded and the corrected retry passed |
| Identity/keyless storage/RBAC verification | Passed; workspace storage roles: Storage Blob Data Contributor and Storage File Data Privileged Contributor; vault role: Key Vault Administrator. Compute identity has no explicit role assignments |
| Default Azure alerting artifact | Azure auto-created Application Insights Smart Detection; owner approved explicit disabled/recipient-free configuration. Live check passed; final WhatIf reports NoChange for it |
| Reproducible second deployment/preview | Repeat application succeeded and live configuration reverified. Final WhatIf: 5 NoChange, 4 Modify entries for Azure-managed fields/defaults; not claimed to be a zero-diff preview |
| Phase boundaries | Workspace job count is 0. No data ingestion, jobs, experiments, models, registry workflows, endpoints, monitoring workflows, retraining or CI/CD created |

## Repeat-deployment interpretation

The repeat deployment succeeded without replacing the workspace or dependencies. The final WhatIf still reports Modify for App Insights creation hints, an Azure ML-generated legacy Key Vault policy (ineffective while RBAC authorization is enabled), workspace defaults/read-only metadata, and compute network defaults. These entries are **not hidden or presented as a clean no-op preview**. Review the full local preview before future deployments. The intended security, authentication and cost settings were reverified after reapplication.

## Local evidence and limits

Ignored `.azure` contains the latest `verification.json`, `compute-availability.json`, `checkov.json`, `what-if.json`, and deployment outputs. The evidence contains no training artifacts. Never copy cloud identifiers, tokens or credentials from local diagnostic files into source control.

The CPU test proves that one node could allocate and reach idle state at test time, and that cleanup completed. It does not reserve future capacity or prove a future training/image-build/data-mount path. Local Entra authentication was exercised; managed identities and their roles were verified, but no application workload was run to request a managed-identity token inside a node.

**Cost state:** allocated=0, target=0, min=0, max=1, idle scale-down=300 seconds. Storage, retained telemetry and platform operations may still incur charges. No deletion/purge was performed. Production private networking and later ML phases remain explicitly out of scope.
