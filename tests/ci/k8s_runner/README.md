# Kubernetes CI runner manifests

The cluster-side objects Miles owns for running `--cluster-backend kubernetes` CI: an in-cluster GitHub Actions runner and the RBAC it needs. The runner executes no test code. Each GitHub job it picks up submits one 0-GPU driver `Job`, and the driver installs the Helm releases the tests run in.

| File | What it defines |
|---|---|
| `runner-launcher-rbac.yaml` | `runner-launcher` (may manage `Job`s and read `Pod`s, logs, and `Event`s in `miles-ci-k8s`) and `miles-ci-reaper` (may delete leaked CI objects in `miles-ci-k8s`) |
| `runner-deployment.yaml` | the ephemeral runner `Deployment` |
| `reaper-cronjob.yaml` | a `CronJob` that removes Helm releases, command jobs, and staging directories whose driver `Job` is gone |

A CI job reaches the runner through `.github/workflows/_run-ci-k8s.yml`, which runs `tests/ci/k8s_run_driver_job.py` from the default branch.

## Who owns what

The platform team installs and maintains the namespaces, LeaderWorkerSet, the admission policy, and the `NetworkPolicy`. This directory holds no copy of those policies; the section below lists what Miles workloads need them to permit.

## Namespaces

| Namespace | Holds |
|---|---|
| `miles-ci-k8s-runner` | the runner `Deployment`, the `runner-launcher` ServiceAccount, and the `github-runner-app-creds` Secret |
| `miles-ci-k8s` | the driver `Job`s and everything they create, under the `miles-workbench` ServiceAccount (`charts/miles-workbench`) |

A driver `Job` can only run as a ServiceAccount of `miles-ci-k8s`, and nothing in `miles-ci-k8s` can read a Secret of `miles-ci-k8s-runner`, so code under test never reaches the runner credential.

## Runner registration

The runner registers with the organization into the runner group `miles-k8s-ci`. An organization owner creates that group with:

- repository access limited to this repository, with public repositories allowed;
- workflow access limited to `radixark/miles/.github/workflows/_run-ci-k8s.yml@refs/heads/main`.

GitHub then hands the runner only jobs defined in that workflow file on the default branch. That file also refuses `pull_request` events whose head is a fork, because a fork can call the pinned file from its own workflow.

`github-runner-app-creds` has one key, `token`: a credential that may create organization runner registration tokens (`Self-hosted runners: write` on the organization) and nothing else.

## What the workloads need the policies to permit

In `miles-ci-k8s`:

- capabilities `SYS_PTRACE` and `SYS_NICE` on GPU containers; every other container drops all capabilities; no container is privileged or allows privilege escalation;
- `hostIPC` on GPU pods (`charts/miles-run`), which is why the node must be dedicated to this namespace;
- `hostPath` mounts of `/storage/shared/miles-ci-k8s/ext`, `/storage/models` (read-only), `/data/miles_ci/jit`, and `/dev/shm/miles-<release>`;
- `NVIDIA_VISIBLE_DEVICES` set only through a `fieldRef`;
- egress to pods of the same namespace, cluster DNS, the cluster API server, public HTTPS, and the metric-history Postgres endpoint.

In `miles-ci-k8s-runner`: the `hostPath` mount `/storage/shared/miles-ci-k8s/ext`, and egress to cluster DNS, the cluster API server, and public HTTPS.

No workload requests an RDMA resource.

## Apply

After the platform prerequisites above exist and the Secret is created:

```bash
kubectl apply -f tests/ci/k8s_runner/runner-launcher-rbac.yaml
```

```bash
kubectl apply -f tests/ci/k8s_runner/runner-deployment.yaml
```

The reaper image is pinned by digest, so it reads the launcher from a ConfigMap. Recreate the ConfigMap whenever `tests/ci/k8s_run_driver_job.py` changes on the default branch:

```bash
kubectl create configmap miles-ci-reaper-script --namespace miles-ci-k8s --from-file=tests/ci/k8s_run_driver_job.py --dry-run=client -o yaml | kubectl apply -f -
```

```bash
kubectl apply -f tests/ci/k8s_runner/reaper-cronjob.yaml
```
