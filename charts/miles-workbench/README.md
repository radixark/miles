# miles-workbench

One long-lived CPU pod on the training image, with shared storage and the RBAC
to install `miles-run` releases. Launch scripts run here, not on a laptop: same
OS, same image, same paths as the training pods.

## Prerequisites

- An admin installed the [LWS](https://github.com/kubernetes-sigs/lws) CRDs and
  granted you rights over LeaderWorkerSets. LWS ships no aggregation labels, so a
  namespace `admin` role never includes them.
- Local `kubectl` and `helm`, and a kubeconfig for the namespace.

## Use

- Every subcommand takes `-n/--namespace` and `-r/--release`.
- `install` creates the namespace only if it is missing, checks that your identity
  may install the chart into an otherwise empty namespace, vendors the library
  chart, runs `helm upgrade --install`, then waits for the pod to be Ready.

```bash
./charts/miles-workbench/cli.py install -n <ns> -r <release> --image-tag <tag> -f my-cluster.yaml
```

Run only those checks, changing nothing in the cluster:

```bash
./charts/miles-workbench/cli.py install --dry-run -n <ns> -r <release>
```

Shell into the pod:

```bash
./charts/miles-workbench/cli.py exec -n <ns> -r <release>
```

Run any command in it instead:

```bash
./charts/miles-workbench/cli.py exec -n <ns> -r <release> -- python -c 'print(1)'
```

Collect pod logs, describes and events into one directory:

```bash
./charts/miles-workbench/cli.py collect-diagnosis -n <ns> -r <release> --output-dir . --run-dir <run state dir>
```

Remove the release, keeping the namespace:

```bash
./charts/miles-workbench/cli.py uninstall -n <ns> -r <release>
```

| Subcommand | Extra flags |
| --- | --- |
| `install` | `--dry-run`, `--image-tag`, `-f/--values` (repeatable), `--set` (repeatable), `--skip-doctor`, `--timeout` (seconds, default 600), `--no-rbac`, `--no-lws` |
| `exec` | trailing command, `bash` by default |
| `collect-diagnosis` | `--output-dir` (default the working directory), `--run-dir` |
| `uninstall` | none |

## Values

| Key | Default | Meaning |
| --- | --- | --- |
| `infra.image.repository` / `.tag` | `radixark/miles` / `dev` | Same image and tag as the training pods. |
| `infra.image.pullPolicy` / `.pullSecrets` | `Always` / `[]` | Standard pull settings. |
| `infra.sharedStorage.type` | `hostPath` | `hostPath`, `pvc`, or `none`. |
| `infra.sharedStorage.hostPath` | `/cluster-storage` | Host path to bind. |
| `infra.sharedStorage.pvcClaimName` | `""` | Pre-existing RWX claim. |
| `infra.sharedStorage.mountPath` | `/cluster-storage` | Keep identical to the training pods; no path translation. |
| `infra.paths.runsSubPath` | `miles_data` | Sub-path of the shared volume that `miles-run` writes run directories under. |
| `infra.paths.repos.miles` / `.megatron` / `.sglang` | `""` | Sub-path of a checkout to mount over `/root/miles`, `/root/Megatron-LM`, `/sgl-workspace/sglang` and put on `PYTHONPATH`. Empty keeps the image's own copy. |
| `infra.nodeLocalStorage.hostPath` / `.mountPath` | `""` / `/scratch` | Node-local scratch disk for `miles-run` pods; empty mounts none. |
| `infra.scheduling.nodeSelector` / `.tolerations` / `.affinity` | `{}` / `[]` / `{}` | Node-pool placement. |
| `infra.env` | `{}` | Extra environment variables. |
| `resources` | 2 CPU / 8Gi requests | It parses args, renders values, follows logs. |
| `rbac.create` | `true` | Create the ServiceAccount, Role and RoleBinding. |
| `rbac.leaderWorkerSets` | `true` | Include the LeaderWorkerSet rules `miles-run` needs. |
| `serviceAccount.name` | `""` | Defaults to the release fullname. |

## Shared values with `miles-run`

- The whole `infra` subtree sits under the same paths in every Miles chart, so one
  per-cluster file drives all of them.
- `charts/shared-infra.schema.json` is the contract; helm cannot `$ref` across
  files, so each chart inlines it and a test pins them equal.
- `charts/miles-common` renders those sections plus naming and labels.

## Permission chain

| Step | Who | What |
| --- | --- | --- |
| 1, once | Admin | Install LWS CRDs and controller; grant users LeaderWorkerSet rights. |
| 2, per namespace | You | `helm install`. Needs every rule in the Role, or both `escalate` and `bind` on roles — either cluster-wide or restricted to this release's Role by name. |
| 3, daily | The pod | Installs and uninstalls `miles-run` releases as its ServiceAccount. |

- The Role lists exactly the object kinds `miles-run` is made of — ConfigMaps,
  Secrets, Services, ServiceAccounts, Deployments, StatefulSets, Jobs,
  LeaderWorkerSets, and the namespaced Roles and RoleBindings its colocate
  pairing controller needs — plus `pods`, `pods/exec`, `pods/log` and read-only
  `events` and `persistentvolumeclaims`; nothing cluster-scoped and no `scale`.
  Adding a new object kind to the chart means adding it here too.
- It grants no `escalate` and no `bind`: Kubernetes admits a namespaced Role or
  RoleBinding write only when the writer already holds every rule being granted,
  so this Role must stay a superset of the pairing controller's Role or the
  install breaks.
- The real boundary is the namespace, not the Role: anything that may create
  workloads may name another ServiceAccount and read its token. Keep privileged
  accounts out of the namespace, and use an admission policy if you need a hard
  boundary.
