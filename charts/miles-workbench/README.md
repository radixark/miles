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

```bash
# check whether your identity may install the chart and use the workbench
./charts/miles-workbench/cli.py doctor -n <ns> -r <release>

# doctor with matching options, vendor the library chart, then helm upgrade --install
./charts/miles-workbench/cli.py install -n <ns> -r <release> --image-tag <tag> -f my-cluster.yaml

# shell into the pod, or run any command in it
./charts/miles-workbench/cli.py exec -n <ns> -r <release>
```

## Values

| Key | Default | Meaning |
| --- | --- | --- |
| `image.repository` / `.tag` | `radixark/miles` / `dev` | Same image and tag as the training pods. |
| `image.pullPolicy` / `.pullSecrets` | `Always` / `[]` | Standard pull settings. |
| `sharedStorage.type` | `hostPath` | `hostPath`, `pvc`, or `none`. |
| `sharedStorage.hostPath` | `/cluster-storage` | Host path to bind. |
| `sharedStorage.pvcClaimName` | `""` | Pre-existing RWX claim. |
| `sharedStorage.mountPath` | `/cluster-storage` | Keep identical to the training pods; no path translation. |
| `scheduling.nodeSelector` / `.tolerations` / `.affinity` | `{}` / `[]` / `{}` | Node-pool placement. |
| `env` | `{}` | Extra environment variables. |
| `resources` | 2 CPU / 8Gi requests | It parses args, renders values, follows logs. |
| `rbac.create` | `true` | Create the ServiceAccount, Role and RoleBinding. |
| `rbac.leaderWorkerSets` | `true` | Include the LeaderWorkerSet rules `miles-run` needs. |
| `serviceAccount.name` | `""` | Defaults to the release fullname. |

## Shared values with `miles-run`

- `image`, `sharedStorage`, `scheduling`, `env` sit under the same paths in every
  Miles chart, so one per-cluster file drives all of them.
- `charts/shared-infra.schema.json` is the contract; helm cannot `$ref` across
  files, so each chart inlines it and a test pins them equal.
- `charts/miles-common` renders those sections plus naming and labels.

## Permission chain

| Step | Who | What |
| --- | --- | --- |
| 1, once | Admin | Install LWS CRDs and controller; grant users LeaderWorkerSet rights. |
| 2, per namespace | You | `helm install`. Needs every rule in the Role, or both `escalate` and `bind` on roles — either cluster-wide or restricted to this release's Role by name. |
| 3, daily | The pod | Installs and uninstalls `miles-run` releases as its ServiceAccount. |

- The Role carries what `miles-run` is made of — ConfigMaps, Secrets, Services,
  ServiceAccounts, StatefulSets, Jobs, LeaderWorkerSets — plus `pods`,
  `pods/log`, `pods/exec`, read-only `events`. Nothing over RBAC objects, nothing
  cluster-scoped, no `scale`. It is fixed: another object type gets added here.
- The boundary is the namespace, not the Role: anything that may create
  workloads may name another ServiceAccount and read its token. Keep privileged
  accounts out of the namespace; use an admission policy for a hard boundary.
