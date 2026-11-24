# Miles Kubernetes Backend CI Runner Manifests

This directory contains the Kubernetes manifests for running Miles `--cluster-backend kubernetes` CI jobs using an in-cluster thin ephemeral GitHub Actions runner (`runner-launcher`) that dispatches per-job 0-GPU Driver `batch/v1 Job` pods under the `miles-workbench` ServiceAccount.

## Architecture

1. **`runner-launcher-rbac.yaml`**: Least-privilege `runner-launcher` SA/Role (`batch/v1 jobs` CRUD and `pods`/`pods/log`/`events` read only — no `secrets`, `services`, `leaderworkersets`, or `pods/exec`), plus `miles-ci-reaper` SA/Role for the orphan cleanup CronJob.
2. **`runner-deployment.yaml`**: 0-GPU non-root ephemeral GitHub Actions runner `Deployment` (no `/var/run/docker.sock`). The `mint-registration-token` `initContainer` exchanges a repo-scoped credential (from Secret `github-runner-app-creds`) for a single-use 1-hour runner registration token and exits before any workflow step runs.
3. **`admission-policy.yaml`**: Namespace-scoped `ValidatingAdmissionPolicy` enforcing non-privileged containers, `allowPrivilegeEscalation: false`, no `hostPID`/`hostNetwork`, capability allowlist `{SYS_PTRACE, SYS_NICE, IPC_LOCK}`, `hostPath` allowlist, and forbidding literal `NVIDIA_VISIBLE_DEVICES=all` (while allowing downward-API `fieldRef` for paired sub-node colocate inference pods).
4. **`network-policy.yaml`**: Namespace isolation allowing intra-namespace pod/CoreDNS communication, kube-system DNS, and HTTPS egress while blocking cloud metadata (`169.254.169.254/32`) and cross-namespace pod traffic.
5. **`reaper-cronjob.yaml`**: 15-minute `CronJob` running `python3 /root/miles/tests/ci/k8s_run_driver_job.py reap` to sweep leaked Helm releases, `command-job` objects, and expired GPFS staging/run directories whose Driver Job has terminated or disappeared.

## Cluster Prerequisites

Before applying these manifests to a namespace (default `miles-ci-k8s`):

1. Install LeaderWorkerSet (`leaderworkerset.x-k8s.io/v1`) CRD and Controller.
2. Create the target namespace and install the `miles-workbench` RBAC/ServiceAccount (`charts/miles-workbench`).
3. Create Secret `github-runner-app-creds` with keys `repository` (e.g. `radixark/miles`) and `token` (a repo-scoped GitHub App installation token minter or fine-grained PAT with `Self-hosted runners: write` only).
   - If using a manual registration token transitionally before GitHub App approval, replace the `Deployment` with a `StatefulSet` backed by persistent runner credentials and drop the `initContainer`.
4. Apply the manifests:
   ```bash
   kubectl apply -f tests/ci/k8s_runner/runner-launcher-rbac.yaml
   kubectl apply -f tests/ci/k8s_runner/admission-policy.yaml
   kubectl apply -f tests/ci/k8s_runner/network-policy.yaml
   kubectl apply -f tests/ci/k8s_runner/runner-deployment.yaml
   kubectl apply -f tests/ci/k8s_runner/reaper-cronjob.yaml
   ```
