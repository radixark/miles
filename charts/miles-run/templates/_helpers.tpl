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

{{- /* Every container of the release: the run's own identity, from which a worker that has to
       reach another worker recomputes its address and builds a backend capability of its own. */ -}}
{{- define "miles-run.releaseEnv" -}}
{{- $identity := dict "MILES_K8S_NAMESPACE" .Release.Namespace "MILES_K8S_RELEASE" .Release.Name -}}
{{- $base := include "miles-common.envBase" . | fromYaml -}}
{{- $run := deepCopy (.Values.run.env | default dict) -}}
{{- toYaml (merge $identity $base $run) }}
{{- end }}

{{- define "miles-run.env" -}}
{{- $release := include "miles-run.releaseEnv" .context | fromYaml -}}
{{- include "miles-common.envBlock" (merge (deepCopy (.entry.env | default dict)) $release) }}
{{- end }}

{{- define "miles-run.envItems" -}}
{{- $release := include "miles-run.releaseEnv" .context | fromYaml -}}
{{- range $name, $value := (merge (deepCopy (.entry.env | default dict)) $release) }}
- name: {{ $name | quote }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}

{{- define "miles-run.cellIndexEnv" -}}
- name: MILES_CELL_INDEX
  valueFrom:
    fieldRef:
      fieldPath: metadata.labels['{{ .label }}']
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
