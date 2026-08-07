# miles-common

Helm library chart: the naming, labelling and shared-infra rendering every Miles
chart would otherwise copy. Ships no templates of its own and cannot be installed.

| Helper | Renders |
| --- | --- |
| `miles-common.fullname` | Release-scoped object name, truncated to 52 so StatefulSet-derived names and labels stay inside 63. |
| `miles-common.componentName` | `<fullname>-<component>`, for charts rendering several workloads. |
| `miles-common.labels` / `.selectorLabels` | The `app.kubernetes.io/*` set; takes `(dict "context" . "component" "<name>")`. |
| `miles-common.image` | Quoted `repository:tag` from `infra.image`. |
| `miles-common.imagePullSecrets` | The `imagePullSecrets` block, or nothing. |
| `miles-common.scheduling` | `nodeSelector` / `tolerations` / `affinity`. |
| `miles-common.env` | The container `env` list: `infra.env` plus the derived `PYTHONPATH`. |
| `miles-common.envBase` | That same map, unrendered, for charts merging per-pod variables into it. |
| `miles-common.envBlock` | An `env` list from any map; takes the map as its context. |
| `miles-common.sharedStorageVolume` / `.sharedStorageVolumeMount` | The shared-storage volume and mount, or nothing when `infra.sharedStorage.type` is `none`. |
| `miles-common.overriddenRepos` | The `infra.paths.repos` entries that are set, as `path` / `subPath` pairs. |
| `miles-common.codeVolumeMounts` | A `subPath` mount per overridden repo, over its in-image path. |
| `miles-common.codePythonPath` | Those in-image paths joined with `:`, for `PYTHONPATH`. |

- Every helper reads the `infra` subtree pinned by `charts/shared-infra.schema.json`,
  so any chart accepting those values renders them identically.
- Consume it with:

  ```yaml
  dependencies:
    - name: miles-common
      version: 0.1.0
      repository: file://../miles-common
  ```

- Then `helm dependency build <chart>` before lint, template or install.
