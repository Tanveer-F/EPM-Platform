# Setup and operations

Run commands from `C:\EPM Platform`. Commands use Windows PowerShell. Use project-local virtual environments; do not install application or scanning packages into system Python or Azure CLI's Python.

## 1. Select the Azure context

```powershell
az login
az account list --query "[].{name:name,state:state,isDefault:isDefault}" --output table
az account set --subscription "<your subscription name>"
```

No real subscription/tenant/principal ID belongs in source files. The scripts resolve the selected subscription in memory and pass it explicitly on operations. They do not change global Azure CLI defaults or silently choose another subscription.

An administrator registers required resource providers if needed:

```powershell
az provider register --namespace Microsoft.Network --wait
```

The full provider list is in the README. Do not run broad role grants or register unrelated services. Quota checks are read-only; deployment never automatically requests an increase or changes regions/SKUs.

## 2. Install local application dependencies

Python 3.12 is the project baseline. Install the locked development dependencies and the local package:

```powershell
py -3.12 -m venv .\.venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest tests -q --disable-warnings -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache src\epm_platform tests
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` pins runtime dependencies; `requirements-dev.txt` pins development dependencies against that runtime lock. They were generated for Windows/Python 3.12. Azure ML SDK v2 is provided by `azure-ai-ml==1.35.0`; authentication uses `azure-identity==1.25.3`. Test/lint tools are pytest 9.1.1 and Ruff 0.16.9. The SDK emits upstream Marshmallow deprecation warnings; these are not evidence of Azure ML v1 usage.

Regenerate locks only when deliberately updating dependencies, then repeat tests and live verification:

```powershell
.\.venv\Scripts\python.exe -m piptools compile --resolver=backtracking --strip-extras --output-file=requirements.txt pyproject.toml
.\.venv\Scripts\python.exe -m piptools compile --resolver=backtracking --strip-extras --allow-unsafe --extra=dev --constraint=requirements.txt --output-file=requirements-dev.txt pyproject.toml
```

Do not assume a Windows-generated dependency lock has been validated for a future Linux training image. Pin and test that environment when its phase is authorized; no training image is built now.

## 3. Configure the environment

```powershell
# Optional: modify non-secret configuration for your approved target.
Copy-Item .\config\.env.example .\.env.local
```

Copy only if `.env.local` does not already exist; never overwrite someone else's local configuration. Keep subscription ID blank to derive it from the selected CLI account. Supported variables:

| Variable | Meaning |
|---|---|
| `AZURE_SUBSCRIPTION_ID` | Runtime target subscription UUID; resolve from CLI or inject into process environment |
| `AZURE_RESOURCE_GROUP` | Must match Bicep's generated group name |
| `AZURE_ML_WORKSPACE` | Must match Bicep's generated workspace name |
| `AZURE_ML_COMPUTE` | `cpu-dev` for this phase |
| `AZURE_LOCATION` | Must match development Bicep parameters |
| `EPM_AUTH_MODE` | `azure-cli` locally, `managed-identity` on an appropriately authorized Azure host |
| `AZURE_CLIENT_ID` | Optional user-assigned managed identity selector; not a secret; never set a client secret |

Use simple unquoted `KEY=value` lines, not shell commands or variable interpolation. Empty lines and full-line `#` comments are accepted. No dotenv package is needed. Existing process variables take precedence over `.env.local` and example defaults.

For interactive SDK/CLI use in an authorized PowerShell session:

```powershell
# Applies to this process only; do not change LocalMachine/CurrentUser policy.
Set-ExecutionPolicy -Scope Process RemoteSigned
. .\scripts\Initialize-Environment.ps1
.\.venv\Scripts\python.exe -m epm_platform.verify
```

Standalone scripts initialize configuration themselves. An Azure managed identity cannot be validated from an ordinary local laptop; local authentication validates the developer's Entra identity, while raw resource checks validate the attached identities/configuration.

## 4. Infrastructure validation

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Prerequisites.ps1
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Infrastructure.ps1
```

The first command checks tooling, authentication, provider registration and configuration alignment. The second compiles Bicep, checks the exact resource-type inventory and security/cost contracts, and parses all PowerShell scripts. It does not contact Azure control/data planes beyond the installed Bicep tooling.

Install the isolated scanner once:

```powershell
py -3.12 -m venv .\.tools\iac
.\.tools\iac\Scripts\python.exe -m pip install -r .\infra\requirements-tools.lock.txt
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-Security.ps1
```

`infra\requirements-tools.txt` declares the scanner version; `infra\requirements-tools.lock.txt` records the 98-package validated Windows/Python 3.12 tool environment. After deliberately updating the declaration, rebuild the isolated tool environment, re-run the scanner, and regenerate its lock with `python -m pip freeze --all` from that environment.

On this Windows installation Checkov's `.cmd` launcher selects the wrong interpreter. The script deliberately invokes the Checkov entry-point file with **its own virtual-environment Python**. This is not a reason to modify system file associations or install Checkov globally. See [reviewed exceptions](security-exceptions.md). Results stay under ignored `.azure`.

## 5. Preview and deploy

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Deploy-Infrastructure.ps1 -Action Validate
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Deploy-Infrastructure.ps1 -Action WhatIf
# After reviewing changes, costs, public-development access and irreversible vault/network choices:
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Deploy-Infrastructure.ps1 -Action Deploy -ApproveCosts
```

Subscription-scope deployment creates `rg-epm-dev-eastus` and a nested incremental resource-group deployment. Re-running uses the same names and declarations; it never uses Complete mode. The script refuses a preview containing deletions. It does not authorize unrelated changes or deletion/purge operations. Preview outputs redact the subscription prefix; the full preview is retained locally in ignored `.azure\what-if.json`.

RBAC propagation and Azure ML managed-network creation can take several minutes. Investigate service errors before retrying; do not fix a role/network problem by enabling storage keys or adding optional services without approval.

## 6. Verify live resources

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Verify-Foundation.ps1
```

This checks both the v2 SDK and CLI, raw stable ARM properties, workspace dependency role assignments and project resource inventory. SDK 1.35.0 does not expose workspace provisioning state or compute OS type; the ARM layer verifies those properties rather than manufacturing unsupported SDK attributes. It writes a sanitized local `.azure\verification.json`. A failed check returns a failing script exit status. Full cloud tokens, identifiers, datastore contents and secret values are not printed by the verifier.

Manual inspection after loading the environment:

```powershell
az ml workspace show --subscription $env:AZURE_SUBSCRIPTION_ID --resource-group $env:AZURE_RESOURCE_GROUP --name $env:AZURE_ML_WORKSPACE --query "{name:name,location:location}" --output json
az resource show --subscription $env:AZURE_SUBSCRIPTION_ID --resource-group $env:AZURE_RESOURCE_GROUP --name $env:AZURE_ML_WORKSPACE --resource-type Microsoft.MachineLearningServices/workspaces --api-version 2025-06-01 --query properties.provisioningState --output tsv
az ml compute show --subscription $env:AZURE_SUBSCRIPTION_ID --resource-group $env:AZURE_RESOURCE_GROUP --workspace-name $env:AZURE_ML_WORKSPACE --name $env:AZURE_ML_COMPUTE --query "{name:name,state:provisioning_state,size:size,min:min_instances,max:max_instances}" --output json
```

Azure ML Studio: <https://ml.azure.com> → selected subscription/workspace → **Compute → Compute clusters**. Idle zero nodes are expected with min=0. Do not click Create job, upload data, or create experiments as a Phase 1 test.

Successful zero-node verification proves the foundation control plane and access policy, not training runtime, data-mount compatibility or future regional allocation capacity. Phase 2 must explicitly authorize any data/workload activity.

## Local inference and monitoring fallback

The current registered model has a verified loopback inference API and local aggregate
monitoring. Follow [Phase 9 + 10 deployment and monitoring](deployment-monitoring.md)
for the pinned model, sample request, scoring contract, monitoring interpretation,
and endpoint cleanup instructions. The first Azure endpoint attempt failed and its
temporary endpoint was deleted; do not retry it automatically.

### Optional billable CPU availability test

Only with explicit cost approval, test actual node allocation without creating a job:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Test-ComputeAvailability.ps1 -ApproveCosts
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Verify-Foundation.ps1
```

The test requires a healthy zero-node baseline, temporarily sets min=1/max=1, waits for one idle node, and uses `finally` to restore min=0/max=1 with the five-minute idle timeout. It waits for actual zero allocated/target nodes, not merely successful submission of a scale request. Each wait has a configurable 5-30 minute timeout (20 by default). No data, job, experiment or model is created. The test does not reserve future capacity.

The sanitized report is `.azure\compute-availability.json`. If restoration cannot be confirmed, the script fails with an urgent warning; inspect this exact cluster immediately because compute charges may continue. Do not interrupt the process during cleanup. The brief allocation itself incurs normal CPU charges.

## Troubleshooting boundaries

- **AuthorizationFailed:** use an authorized deployment principal; distinguish control-plane permissions from data-plane roles and RBAC assignment rights. Do not grant Owner to the compute identity.
- **Expired sign-in:** repeat `az login`, explicitly select the target account and retry read-only validation.
- **ProviderNotRegistered:** have an administrator register the named required provider; no unrelated registrations.
- **Quota/capacity unavailable:** stop and ask before changing VM size/region or requesting quota.
- **Soft-deleted vault name conflict:** stop; inspect recovery options. Purge protection is intentional and must not be bypassed.
- **Missing optional dependency reported by Azure ML:** retain error evidence and ask before adding ACR or observability resources.
- **Storage data access failure:** do not enable account keys; approve exact identity/data-role requirements in the relevant later phase.
