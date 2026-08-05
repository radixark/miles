{{- define "miles-run.fullname" -}}
{{- include "miles-common.fullname" . }}
{{- end }}

{{- define "miles-run.componentName" -}}
{{- include "miles-common.componentName" . }}
{{- end }}

{{- define "miles-run.labels" -}}
{{- include "miles-common.labels" . }}
{{- end }}

{{- define "miles-run.selectorLabels" -}}
{{- include "miles-common.selectorLabels" . }}
{{- end }}

{{- define "miles-run.runDir" -}}
{{- $infra := .Values.infra -}}
{{- $root := trimSuffix "/" $infra.sharedStorage.mountPath -}}
{{- $subPath := trimSuffix "/" (($infra.paths | default dict).runsSubPath | default "") -}}
{{- with $subPath }}{{- $root = printf "%s/%s" $root . }}{{- end }}
{{- printf "%s/miles-runs/%s" $root .Values.run.id }}
{{- end }}

{{- define "miles-run.orchestratorExitFile" -}}
{{- printf "%s/state/orchestrator.exit" (include "miles-run.runDir" .) }}
{{- end }}

{{- define "miles-run.podDefaults" -}}
{{- $context := . -}}
{{- $scheduling := $context.Values.infra.scheduling | default dict -}}
enableServiceLinks: false
{{- with include "miles-common.imagePullSecrets" $context }}
{{ . }}
{{- end }}
{{- with $scheduling.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $scheduling.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $scheduling.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-run.podDefaultsWithAntiAffinity" -}}
{{- $context := .context -}}
{{- $scheduling := $context.Values.infra.scheduling | default dict -}}
enableServiceLinks: false
{{- with include "miles-common.imagePullSecrets" $context }}
{{ . }}
{{- end }}
{{- with $scheduling.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $scheduling.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
affinity:
  {{- toYaml (merge (dict "podAntiAffinity" .podAntiAffinity) (deepCopy ($scheduling.affinity | default dict))) | nindent 2 }}
{{- end }}

{{- define "miles-run.env" -}}
{{- $base := include "miles-common.envBase" .context | fromYaml -}}
{{- $run := deepCopy (.context.Values.run.env | default dict) -}}
{{- include "miles-common.envBlock" (merge (deepCopy (.entry.env | default dict)) $base $run) }}
{{- end }}

{{- define "miles-run.nodeLocalVolume" -}}
{{- with (.Values.infra.nodeLocalStorage | default dict).hostPath -}}
- name: node-local
  hostPath:
    path: {{ . | quote }}
    type: DirectoryOrCreate
{{- end }}
{{- end }}

{{- define "miles-run.nodeLocalVolumeMount" -}}
{{- with .Values.infra.nodeLocalStorage | default dict -}}
{{- if .hostPath }}
- name: node-local
  mountPath: {{ .mountPath | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-run.containerDefaults" -}}
image: {{ include "miles-common.image" . }}
imagePullPolicy: {{ .Values.infra.image.pullPolicy | quote }}
{{- $mounts := compact (list (include "miles-common.sharedStorageVolumeMount" . | trim) (include "miles-common.codeVolumeMounts" . | trim) (include "miles-run.nodeLocalVolumeMount" . | trim)) | join "\n" }}
{{- with $mounts }}
volumeMounts:
  {{- . | nindent 2 }}
{{- end }}
{{- end }}
