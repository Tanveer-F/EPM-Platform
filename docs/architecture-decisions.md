# Phase 1 architecture decisions

These decisions apply to the **development foundation**, not to a production deployment approval. Approved by the project owner on 2026-09-26.

## 1. Bicep and standard Azure ML

Use Azure-native Bicep with a subscription-scope entry point (creates the dedicated resource group) and resource-group modules. Use a standard Azure ML workspace (`kind: Default`), not an AI Foundry hub/project. Stable ARM APIs are pinned: Azure ML `2025-06-01`, Storage `2025-01-01`, Key Vault `2024-11-01`, Resource Groups `2025-04-01`.

Python uses `azure.ai.ml.MLClient`; CLI uses the `ml` v2 extension. The `azure-ai-ml` **package version can be 1.x while implementing SDK v2**; this is distinct from deprecated `azureml-core`. No v1 SDK or legacy workspace mode is used.

Alternative: Terraform/azd would add tooling and abstractions without a Phase 1 requirement. Managed services are used rather than custom VMs/Kubernetes. Future phases add modules and Python packages only when requested.

## 2. Required dependencies only

Explicitly create StorageV2 and Key Vault and link them to the workspace. Their workspace association is effectively immutable: use deterministic names and do not casually replace either dependency.

The initial minimal deployment omitted ACR/Application Insights/Log Analytics following Microsoft's identity-based ARM example. ARM validation succeeded, but actual workspace creation failed with `Missing dependent resources in workspace json`. No workspace or compute was created by that attempt; the resource group/storage/vault were retained.

The owner then explicitly approved adding **workspace-based Application Insights plus its Log Analytics workspace** and retrying. They are linked workspace dependencies, not application instrumentation or a monitoring workflow. Both require Entra authentication, use the approved public development network, and stay in East US. Log Analytics uses PerGB2018, 30-day retention and a 1 GB/day ingestion safeguard; the safeguard is not a guaranteed spending cap. No connection strings or instrumentation keys are emitted into source/configuration.

ACR remains deferred because no image build or deployment exists. No diagnostics settings, alert rules, dashboards, drift/performance monitoring or retraining are configured.

Azure asynchronously created its default `Application Insights Smart Detection` action group with two role-based recipients. The owner explicitly approved managing that **existing platform-default action group as disabled and recipient-free** in Bicep. This suppresses notifications rather than implementing a monitoring workflow. Live verification requires it to remain disabled and rejects unexpected alert-rule resource types. If another dependency is rejected or unexpected service appears, stop and ask rather than silently expanding scope.

Storage has hierarchical namespace disabled, as required for default workspace storage. Seven-day blob/container soft delete provides basic protection without ingestion pipelines or backup services. Standard LRS is an intentional development cost/reliability tradeoff, not disaster recovery.

## 3. Public authenticated development access

The owner initially preferred an existing private corporate network, but confirmed that no VNet/VPN is available. The owner explicitly approved public authenticated service endpoints for development instead of funding a new VPN topology.

- Workspace, storage and Key Vault service endpoints are public. Public does **not** mean anonymous: Entra/RBAC protects access, storage shared keys and public blobs are disabled, and HTTPS/TLS 1.2 is enforced.
- Azure ML manages compute networking with `AllowInternetOutbound`. This allows CPU nodes without public node IPs; public SSH and local compute authentication are disabled.
- There is no custom firewall/FQDN egress rule, VPN, public inference endpoint or customer VNet.
- Managed networking cannot be disabled after it is enabled. Managed-network generation `V1` is a networking property, **not Azure ML SDK v1**.
- Managed network resources and any platform-created private endpoints must be inventoried after deployment. Do not equate "not in Bicep" with "no platform-managed resources or charges."

**Before production:** approve private client connectivity, private endpoints and private DNS for workspace/Blob/Files/Key Vault, disable public service access, choose an egress policy, and review regional availability, resilience and cost. Merely toggling `publicNetworkAccess` without connectivity/DNS would break access; this repository intentionally does not offer that incomplete configuration.

## 4. Identities and least privilege

Local development uses `AzureCliCredential` after interactive Entra `az login`. A deterministic credential mode avoids silently selecting unrelated environment/service-principal credentials. Deployed Python can explicitly select `ManagedIdentityCredential`; runtime configuration supplies identifiers, not secrets.

Workspace and compute each have their own system-assigned managed identity. Azure ML manages the workspace identity's service dependency grants. Inspect and record actual role names after provisioning; do not assume that resource Contributor grants Blob/Files/Key Vault data-plane access. Do not invent narrower grants that prevent workspace initialization, or grant subscription-wide Owner to the compute identity.

The empty compute cluster is not pre-granted dataset/container write access, Key Vault Administrator or workspace Contributor. Grant exact workload access when a future phase defines which identity reads/writes which resources. No data-access role or secret permission is added to the human merely for metadata validation.

Deployment authorization is separate from runtime access. Creating the RG needs subscription-scope resource-group/deployment permissions; creating resources and applicable role assignments needs appropriately scoped authorization. An existing account's Owner access is not a role assigned by this project. A production organization should use an approved deployment principal and scoped roles, with humans granted only appropriate workspace/Reader roles.

Key Vault uses RBAC, soft delete and 90-day purge protection. Bicep declares no legacy access-policy grants. Azure ML nevertheless adds a service-generated legacy access-policy entry; **with RBAC authorization enabled, that policy does not authorize access**. Live verification confirms the effective workspace identity role is Key Vault Administrator, scoped to this dedicated vault. Purge protection is irreversible. No secrets, keys, connection strings, tokens or real subscription/tenant/principal IDs belong in tracked files.

## 5. CPU cost boundary and verification semantics

`cpu-dev` is dedicated Linux `Standard_D2s_v3` (2 vCPUs), minimum 0, maximum 1, with a five-minute idle scale-down. East US DSv3 dedicated quota was checked before implementation (6 cores, 0 used). Quota is not a reservation or guarantee of future allocation. Dedicated avoids Spot eviction during initial development; GPU and compute instances are excluded.

Read-only validation checks provisioned resource state, identities, CLI/SDK authentication, policies and dependency grants. It does not submit a job, create an MLflow experiment, stage data or scale up compute. A zero-node cluster can be successfully provisioned without proving the runtime path for future training. Do not hide this distinction in completion claims.

## 6. Reproducibility and boundaries

Bicep and environment configuration are separated from Python source. Dependencies are pinned in a project-local environment; infrastructure scanning is isolated under `.tools` so scanner dependencies cannot constrain the application SDK. No notebook/demo structure or unnecessary model scaffolding is added.

Environment precedence is process > ignored `.env.local` > safe example defaults. IDs are resolved at runtime. PowerShell config parsing accepts a limited set of literal `KEY=value` entries, never `Invoke-Expression`.

No monitoring, CI/CD, retraining, registry, model deployment, data processing or PyTorch is implemented under the guise of foundation validation. Application logs and minimal command diagnostics are not a monitoring platform.

## 7. Phase 9 + 10 — endpoint cost boundary and local fallback

The approved first deployment attempt uses one short-lived Azure ML Managed Online
Endpoint with token authentication, a single small CPU replica, and the existing
workspace Application Insights component. A smoke script always deletes the endpoint
after validation and refuses automatic retry. The September 2026 attempt returned an
Azure ML SDK `HttpResponseError` before a cloud prediction could be verified; the
temporary endpoint was deleted and the inventory confirmed empty. Per the explicit
cost-control rule, no second Azure deployment or new registry/image-building resource
was provisioned.

The current validated path is local loopback inference with the same registered
XGBoost JSON, 35-feature causal online transformation, strict request checks and
aggregate range/latency/error telemetry. It adds no monitoring service or persisted
cloud resources. Input drift compares 20-or-more request batches to the frozen Phase 4
training feature percentiles; prediction-range checks use the retrospective
validation outputs. Those data/reference caveats are documented with the serving
contract in [deployment and monitoring](deployment-monitoring.md).

## 8. Final lifecycle status — Phases 11–13

The later phases add an **operator-triggered** retraining/promotion path and GitHub
validation workflow without adding Azure services. Drift and supplied labeled
performance evidence are evaluated locally; an Azure ML baseline job requires an
explicit cost-approval flag. Candidate registration is gated by verified lineage and
the documented RMSE/NASA score thresholds. There is no schedule, CI Azure identity,
automatic deployment, or retraining run. The reviewed limits—including repeat use of
the fixed C-MAPSS public test set—are in [the final architecture summary](architecture.md)
and [retraining policy](retraining-cicd.md).

Final read-only workspace validation confirms the registered ML-ready data and model,
zero active Azure ML jobs, no online endpoints, and no listed `cpu-dev` nodes. The
local test suite, separate PyTorch/MLflow suite, Ruff, dependency checks, and compiled
Bicep/infrastructure contracts were run without Azure training or endpoint operations.

## Sources

- [Disable shared-key storage authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-disable-local-auth-storage?view=azureml-api-2)
- [Workspace management and limitations](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2)
- [Managed identity service authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication?view=azureml-api-2)
- [Compute in managed networks](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network-compute?view=azureml-api-2)
- [Managed-network behavior and costs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network?view=azureml-api-2)
- [Workspace ARM API](https://learn.microsoft.com/en-us/azure/templates/microsoft.machinelearningservices/2025-06-01/workspaces)
- [Compute ARM API](https://learn.microsoft.com/en-us/azure/templates/microsoft.machinelearningservices/2025-06-01/workspaces/computes)
- [Azure naming constraints](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules)
- [Azure retail pricing API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices)
