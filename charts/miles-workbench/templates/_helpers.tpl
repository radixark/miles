{{- define "miles-workbench.fullname" -}}
{{- .Values.objectName }}
{{- end }}

{{- define "miles-workbench.labels" -}}
{{- include "miles-common.labels" (dict "context" . "component" "workbench") }}
{{- end }}

{{- define "miles-workbench.selectorLabels" -}}
{{- include "miles-common.selectorLabels" (dict "context" . "component" "workbench") }}
{{- end }}

{{- define "miles-workbench.serviceAccountName" -}}
{{- default (include "miles-workbench.fullname" .) .Values.serviceAccount.name }}
{{- end }}

{{- define "miles-workbench.assertServiceAccountNameIsFree" -}}
{{- $uninstaller := include "miles-common.uninstallerServiceAccountName" . -}}
{{- if and .Values.rbac.create (eq (include "miles-workbench.serviceAccountName" .) $uninstaller) }}
{{- fail (printf "serviceAccount.name is %s, which is the fixed name of the account a finished run uninstalls itself as: two ServiceAccounts of that one name cannot both be created, and the workbench account must outlive the runs it launches" $uninstaller) }}
{{- end }}
{{- end }}

{{- define "miles-workbench.roleRules" -}}
- apiGroups: [""]
  resources: ["configmaps", "secrets", "serviceaccounts", "services"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["delete", "get", "list", "patch", "update", "watch"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create", "get"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["events", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["rbac.authorization.k8s.io"]
  resources: ["roles", "rolebindings"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
{{- if .Values.rbac.leaderWorkerSets }}
- apiGroups: ["leaderworkerset.x-k8s.io"]
  resources: ["leaderworkersets"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
{{- end }}
{{- end }}

{{- define "miles-workbench.uninstallerRoleRules" -}}
- apiGroups: [""]
  resources: ["configmaps", "secrets", "serviceaccounts", "services", "pods"]
  verbs: ["get", "list", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "delete"]
- apiGroups: ["rbac.authorization.k8s.io"]
  resources: ["roles", "rolebindings"]
  verbs: ["get", "list", "delete"]
{{- if .Values.rbac.leaderWorkerSets }}
- apiGroups: ["leaderworkerset.x-k8s.io"]
  resources: ["leaderworkersets"]
  verbs: ["get", "list", "delete"]
{{- end }}
{{- end }}

{{- define "miles-workbench.infraConfigMapName" -}}
{{- printf "%s-infra" (include "miles-workbench.fullname" .) }}
{{- end }}

{{- define "miles-workbench.infraValuesFileName" -}}
infra.yaml
{{- end }}

{{- define "miles-workbench.infraValuesDir" -}}
/etc/miles
{{- end }}

{{- define "miles-workbench.infraValuesVolumeName" -}}
infra-values
{{- end }}

{{- define "miles-workbench.assertInfraValuesMountIsReserved" -}}
{{- $root := . -}}
{{- $reservedName := include "miles-workbench.infraValuesVolumeName" . -}}
{{- $reservedDir := include "miles-workbench.infraValuesDir" . -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- if eq $volume.name $reservedName }}
{{- fail (printf "infra.volumes[%s] takes the volume name the chart itself uses for the %s mount, and kubernetes rejects a pod that declares one name twice: rename this volume" $reservedName $reservedDir) }}
{{- end }}
{{- range $index, $mount := ($volume.mounts | default list) }}
{{- $mountPath := include "miles-common.mountPath" (dict "root" $root "volume" $volume "index" $index "mount" $mount) }}
{{- if eq $mountPath $reservedDir }}
{{- fail (printf "infra.volumes[%s].mounts[%d].mountPath is %s, which the chart mounts the installed infra values on, and kubernetes rejects a pod that mounts one path twice: mount this volume elsewhere" $volume.name $index $reservedDir) }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-workbench.env" -}}
{{- $launch := dict
      "MILES_SCRIPT_CLUSTER_BACKEND" "kubernetes"
      "MILES_SCRIPT_NAMESPACE" .Release.Namespace
      "MILES_SCRIPT_HELM_VALUES" (printf "%s/%s" (include "miles-workbench.infraValuesDir" .) (include "miles-workbench.infraValuesFileName" .))
-}}
{{- include "miles-common.envBlock" (merge (include "miles-common.envBase" . | fromYaml) $launch) }}
{{- end }}
