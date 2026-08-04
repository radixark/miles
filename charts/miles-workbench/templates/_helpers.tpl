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

{{- define "miles-workbench.roleRules" -}}
- apiGroups: [""]
  resources: ["configmaps", "secrets", "serviceaccounts", "services"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["delete", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["events", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["statefulsets"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
{{- if .Values.rbac.leaderWorkerSets }}
- apiGroups: ["leaderworkerset.x-k8s.io"]
  resources: ["leaderworkersets"]
  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
{{- end }}
{{- end }}
