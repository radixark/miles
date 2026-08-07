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

{{- define "miles-run.colocateEnv" -}}
{{- $colocate := .Values.run.colocate | default dict -}}
{{- $env := dict -}}
{{- if $colocate.enabled }}
{{- $engine := dict }}
{{- range .Values.run.inferenceEngines }}{{- if eq .name $colocate.enginePool }}{{- $engine = . }}{{- end }}{{- end }}
{{- $trainer := dict }}
{{- range .Values.run.trainers }}{{- if eq .name $colocate.trainerPool }}{{- $trainer = . }}{{- end }}{{- end }}
{{- $env = dict
      "MILES_K8S_COLOCATE_ENGINE_COMPONENT" (include "miles-run.componentName" (dict "context" . "component" $colocate.enginePool))
      "MILES_K8S_COLOCATE_TRAINER_COMPONENT" (include "miles-run.componentName" (dict "context" . "component" $colocate.trainerPool))
      "MILES_K8S_COLOCATE_TRAINER_POOL" $colocate.trainerPool
      "MILES_K8S_COLOCATE_ENGINE_CELLS" (default 1 $engine.replicas | toString)
      "MILES_K8S_COLOCATE_TRAINER_CELLS" (default 1 $trainer.replicas | toString)
      "MILES_K8S_COLOCATE_PODS_PER_ENGINE_CELL" (default 1 $engine.size | toString)
      "MILES_K8S_COLOCATE_PODS_PER_TRAINER_CELL" (default 1 $trainer.size | toString) }}
{{- end }}
{{- toYaml $env }}
{{- end }}
