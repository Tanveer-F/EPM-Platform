# Phase 1 development security exceptions

Owner approval: 2026-09-26, authenticated public endpoints for development after confirming that no corporate VNet/VPN exists. These exceptions **must not be copied into a production security profile**.

Checkov runs locally with `--skip-download --skip-results-upload`: no code or scan results are uploaded. Offline scans do not supply vendor severity metadata, so no severity rating is invented. All failed checks from the initial scan were reviewed individually; only the following resource-local suppressions are present.

| Check | Resource | Disposition and compensating controls |
|---|---|---|
| `CKV_AZURE_35` | Storage | Default network rule is intentionally Allow for approved public development access. Shared keys and anonymous blobs are disabled; HTTPS/TLS 1.2 and Entra data authorization remain enforced. |
| `CKV_AZURE_109` | Key Vault | Firewall restriction deferred with the approved public development network. Vault RBAC, soft delete and purge protection remain enforced. |
| `CKV_AZURE_189` | Key Vault | Public network access is an approved development exception. Entra authorization remains required. Private endpoints/DNS/client connectivity are production prerequisites. |
| `CKV_AZURE_243` | Azure ML workspace | No private client endpoint in development. Entra workspace authorization and separate managed identities remain enabled. Compute node public IP, public SSH and local auth are disabled. |
| `CKV_AZURE_206` | Storage | Standard LRS is the approved development cost/resilience tradeoff; it is not a regional DR solution. Production redundancy/RTO/RPO require an explicit decision. |
| `CKV_AZURE_43` | Storage | Static analyzer cannot evaluate the deterministic `take`/`uniqueString` name. This is a scanner limitation, not a waived Azure naming rule. Bicep/ARM validation and deployment WhatIf validate the actual generated lowercase alphanumeric name. |

Initial Checkov 3.3.19 also failed internally when inspecting an expression-valued Key Vault public-access setting. That property is now explicitly `Enabled` in the development-only module; the module requires a true development acknowledgement. The final scan must have **no scanner errors**, not merely exit zero with a suppressed internal exception.

No global `--skip-check` or soft-fail flag is used. Encryption, no anonymous access, keyless storage, managed identity, retention protections, scale-to-zero and the absence of future-phase resources are separately covered by compiled-template contract tests and live verification.

## Phase 3/4 tooling coverage

The final runtime-only `training-access.bicep` contains five container-scoped role assignments and no registry, storage, compute or other service resource. Checkov 3.3.19 cannot parse its existing-resource identity references. `Test-Security.ps1` explicitly excludes that file from the failing parser and invokes `Test-TrainingInfrastructure.ps1`, which compiles it and checks exact resource counts/types, container scopes, built-in role IDs and read-only curated input access. This is a documented scanner coverage limitation, not a suppressed vulnerability result.

An initially approved Basic ACR was created but could not attach to the managed-network workspace (Premium required). The owner selected a pinned public curated environment instead and approved deletion of the verified-empty unattached registry. Final source does not recreate that registry; the provisional registry scan findings are not applicable to retained resources.

Re-review these exceptions when networking, storage redundancy or environment scope changes. Adding a service or loosening another control requires a new architecture decision, not another blanket suppression.
