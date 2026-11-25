{{- define "miles-run.labels" -}}
{{- include "miles-common.labels" . }}
{{- end }}

{{- define "miles-run.selectorLabels" -}}
{{- include "miles-common.selectorLabels" . }}
{{- end }}

{{- define "miles-run.accountUsedByAWorker" -}}
{{- $name := .name -}}
{{- range $worker := .context.Values.run.staticWorkers -}}
{{- if eq (default "" $worker.serviceAccountName) $name -}}
used
{{- end -}}
{{- end -}}
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
{{- $affinity := $scheduling.affinity | default dict }}
{{- if .withoutPodAntiAffinity }}
{{- $affinity = omit $affinity "podAntiAffinity" }}
{{- end }}
{{- with $affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-run.podDefaultsWithAntiAffinity" -}}
{{- $context := .context -}}
{{- $scheduling := $context.Values.infra.scheduling | default dict -}}
{{- $affinity := deepCopy ($scheduling.affinity | default dict) -}}
{{- $antiAffinity := $affinity.podAntiAffinity | default dict -}}
{{- $required := concat (.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution | default list) ($antiAffinity.requiredDuringSchedulingIgnoredDuringExecution | default list) -}}
{{- $_ := set $affinity "podAntiAffinity" (set $antiAffinity "requiredDuringSchedulingIgnoredDuringExecution" $required) -}}
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
  {{- toYaml $affinity | nindent 2 }}
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
- name: MILES_POD_UID
  valueFrom:
    fieldRef:
      fieldPath: metadata.uid
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

{{- define "miles-run.interpolateShmPath" -}}
{{- $value := .value -}}
{{- range $reference := regexFindAll "\\$\\{[^}]*\\}" $value -1 -}}
{{- if and (ne $reference "${NAMESPACE}") (ne $reference "${RELEASE}") -}}
{{- fail (printf "%s is %s, which names the unknown variable %s: only ${NAMESPACE} and ${RELEASE} are allowed" $.field $value $reference) -}}
{{- end -}}
{{- end -}}
{{- $value | replace "${NAMESPACE}" .root.Release.Namespace | replace "${RELEASE}" .root.Release.Name -}}
{{- end }}

{{- define "miles-run.shmVolumeFor" -}}
{{- $context := .context -}}
{{- with $context.Values.infra.devShm -}}
{{- $source := . -}}
{{- if and $.colocated .colocateHostPath -}}
{{- $path := include "miles-run.interpolateShmPath" (dict "root" $context "field" "infra.devShm.colocateHostPath.path" "value" .colocateHostPath.path) -}}
{{- $source = dict "hostPath" (dict "path" $path "type" (.colocateHostPath.type | default "Directory")) -}}
{{- else if .hostPath -}}
{{- $path := include "miles-run.interpolateShmPath" (dict "root" $context "field" "infra.devShm.hostPath.path" "value" .hostPath.path) -}}
{{- $source = mustMergeOverwrite (deepCopy .) (dict "hostPath" (dict "path" $path)) -}}
{{- end }}
- name: dev-shm
  {{- include "miles-common.volumeSource" $source | trim | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-run.shmVolumeMountFor" -}}
{{- with .context.Values.infra.devShm -}}
- name: dev-shm
  mountPath: {{ .mountPath | quote }}
{{- end }}
{{- end }}

{{- define "miles-run.shmVolume" -}}
{{- include "miles-run.shmVolumeFor" (dict "context" . "colocated" false) }}
{{- end }}

{{- define "miles-run.shmVolumeMount" -}}
{{- include "miles-run.shmVolumeMountFor" (dict "context" . "colocated" false) }}
{{- end }}

{{- define "miles-run.poolNeedsGpuRuntime" -}}
{{- $pool := .pool -}}
{{- if hasKey $pool "needsGpuRuntime" -}}
{{- if $pool.needsGpuRuntime }}true{{ else }}false{{ end -}}
{{- else if .gated -}}
true
{{- else if gt (int (default 0 (index (($pool.resources | default dict).limits | default dict) "nvidia.com/gpu"))) 0 -}}
true
{{- else -}}
false
{{- end -}}
{{- end }}

{{- define "miles-run.gpuContainerSecurityContext" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    add:
      - SYS_PTRACE
      - SYS_NICE
{{- end }}

{{- define "miles-run.cpuContainerSecurityContext" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
{{- end }}

{{- define "miles-run.containerDefaults" -}}
{{- include "miles-run.containerDefaultsWith" (dict "context" . "extraMounts" "") }}
{{- end }}

{{- define "miles-run.containerDefaultsWith" -}}
{{- $context := .context -}}
image: {{ include "miles-common.image" $context }}
imagePullPolicy: {{ $context.Values.infra.image.pullPolicy | quote }}
workingDir: "/root/miles"
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

