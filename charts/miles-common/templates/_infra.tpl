{{- define "miles-common.image" -}}
{{- $image := .Values.infra.image | default dict -}}
{{- printf "%s:%s" ($image.repository | default "") ($image.tag | default "") | quote }}
{{- end }}

{{- define "miles-common.imagePullSecrets" -}}
{{- with (.Values.infra.image | default dict).pullSecrets -}}
imagePullSecrets:
{{- range . }}
  - name: {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.scheduling" -}}
{{- with (.Values.infra.scheduling | default dict).nodeSelector -}}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with (.Values.infra.scheduling | default dict).tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with (.Values.infra.scheduling | default dict).affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-common.envBase" -}}
{{- $base := dict -}}
{{- with include "miles-common.codePythonPath" . }}{{- $base = dict "PYTHONPATH" . }}{{- end }}
{{- toYaml (merge (deepCopy (.Values.infra.env | default dict)) $base) }}
{{- end }}

{{- define "miles-common.env" -}}
{{- include "miles-common.envBlock" (include "miles-common.envBase" . | fromYaml) }}
{{- end }}

{{- define "miles-common.envBlock" -}}
{{- with . -}}
env:
{{- range $name, $value := . }}
  - name: {{ $name | quote }}
    value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.repoTargets" -}}
- name: miles
  path: /root/miles
- name: megatron
  path: /root/Megatron-LM
- name: sglang
  path: /sgl-workspace/sglang
{{- end }}

{{- define "miles-common.overriddenRepos" -}}
{{- $repos := ((.Values.infra.paths | default dict).repos | default dict) -}}
{{- $mounted := list -}}
{{- if ne ((.Values.infra.sharedStorage | default dict).type | default "none") "none" -}}
{{- range $target := include "miles-common.repoTargets" . | fromYamlArray }}
{{- with get $repos $target.name }}
{{- $mounted = append $mounted (dict "path" $target.path "subPath" .) }}
{{- end }}
{{- end }}
{{- end }}
{{- toYaml $mounted }}
{{- end }}

{{- define "miles-common.codeVolumeMounts" -}}
{{- range include "miles-common.overriddenRepos" . | fromYamlArray }}
- name: shared-storage
  mountPath: {{ .path | quote }}
  subPath: {{ .subPath | quote }}
{{- end }}
{{- end }}

{{- define "miles-common.codePythonPath" -}}
{{- $entries := list -}}
{{- range include "miles-common.overriddenRepos" . | fromYamlArray }}
{{- $entries = append $entries .path }}
{{- end }}
{{- join ":" $entries }}
{{- end }}

{{- define "miles-common.sharedStorageVolume" -}}
{{- if ne ((.Values.infra.sharedStorage | default dict).type | default "none") "none" -}}
- name: shared-storage
  {{- if eq (.Values.infra.sharedStorage | default dict).type "hostPath" }}
  hostPath:
    path: {{ (.Values.infra.sharedStorage | default dict).hostPath | quote }}
    type: Directory
  {{- else }}
  persistentVolumeClaim:
    claimName: {{ (.Values.infra.sharedStorage | default dict).pvcClaimName | quote }}
  {{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.sharedStorageVolumeMount" -}}
{{- if ne ((.Values.infra.sharedStorage | default dict).type | default "none") "none" -}}
- name: shared-storage
  mountPath: {{ (.Values.infra.sharedStorage | default dict).mountPath | quote }}
{{- end }}
{{- end }}
