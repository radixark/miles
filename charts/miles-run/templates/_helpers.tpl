{{- define "miles-run.labels" -}}
{{- include "miles-common.labels" . }}
{{- end }}

{{- define "miles-run.selectorLabels" -}}
{{- include "miles-common.selectorLabels" . }}
{{- end }}

{{- define "miles-run.podDefaults" -}}
{{- include "miles-run.podDefaultsFor" (dict "context" . "gated" false) }}
{{- end }}

{{- define "miles-run.podDefaultsFor" -}}
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
{{- if not .gated }}
{{- with $scheduling.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
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

{{- /* Every container of the release: the run's own identity, from which a worker that has to
       reach another worker recomputes its address and builds a backend capability of its own. */ -}}
{{- define "miles-run.releaseEnv" -}}
{{- $identity := dict "MILES_K8S_NAMESPACE" .Release.Namespace "MILES_K8S_RELEASE" .Release.Name -}}
{{- with .Values.run.launchRecord }}{{- $identity = merge (dict "MILES_SCRIPT_ENV_REPORT" .) $identity }}{{- end }}
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

{{- define "miles-run.labelEnv" -}}
- name: {{ .name }}
  valueFrom:
    fieldRef:
      fieldPath: metadata.labels['{{ .label }}']
{{- end }}

{{- define "miles-run.annotationEnv" -}}
- name: {{ .name }}
  valueFrom:
    fieldRef:
      fieldPath: metadata.annotations['{{ .annotation }}']
{{- end }}

{{- define "miles-run.shmVolume" -}}
{{- with .Values.infra.devShm -}}
- name: dev-shm
  {{- include "miles-common.volumeSource" . | trim | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-run.shmVolumeMount" -}}
{{- with .Values.infra.devShm -}}
- name: dev-shm
  mountPath: {{ .mountPath | quote }}
{{- end }}
{{- end }}

{{- define "miles-run.containerDefaults" -}}
{{- include "miles-run.containerDefaultsWith" (dict "context" . "extraMounts" "") }}
{{- end }}

{{- define "miles-run.containerDefaultsWith" -}}
{{- $context := .context -}}
image: {{ include "miles-common.image" $context }}
imagePullPolicy: {{ $context.Values.infra.image.pullPolicy | quote }}
{{- $mounts := compact (list (include "miles-common.volumeMounts" $context | trim) (.extraMounts | trim)) | join "\n" }}
{{- with $mounts }}
volumeMounts:
  {{- . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-run.autoUninstallEnabled" -}}
{{- if .Values.run.autoUninstall.enabled -}}
true
{{- end }}
{{- end }}

{{- define "miles-run.uninstallManifestDir" -}}
/etc/miles-uninstall
{{- end }}

{{- define "miles-run.uninstallManifestFileName" -}}
uninstall-job.yaml
{{- end }}

{{- define "miles-run.uninstallManifestVolume" -}}
{{- if include "miles-run.autoUninstallEnabled" . -}}
- name: uninstall-manifest
  configMap:
    name: {{ .Values.run.objectNames.uninstallManifest | quote }}
{{- end }}
{{- end }}

{{- define "miles-run.uninstallManifestVolumeMount" -}}
{{- if include "miles-run.autoUninstallEnabled" . -}}
- name: uninstall-manifest
  mountPath: {{ include "miles-run.uninstallManifestDir" . | quote }}
  readOnly: true
{{- end }}
{{- end }}

