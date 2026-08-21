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
{{- $platform := dict "PYTHONPATH" (include "miles-common.codePythonPath" .) -}}
{{- toYaml (merge (deepCopy (.Values.infra.env | default dict)) $platform) }}
{{- end }}

{{- define "miles-common.codePythonPath" -}}
/root/miles:/root/Megatron-LM:/sgl-workspace/sglang/python
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

{{- define "miles-common.volumes" -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
- name: {{ $volume.name | quote }}
  {{- include "miles-common.volumeSource" $volume | trim | nindent 2 }}
{{- end }}
{{- end }}

{{- /* The one place a values file's source keys become a kubernetes volume, so that infra.volumes
       and infra.devShm cannot drift into accepting different ones. */ -}}
{{- define "miles-common.volumeSource" -}}
{{- if .hostPath }}
hostPath:
  path: {{ .hostPath.path | quote }}
  type: {{ .hostPath.type | default "Directory" | quote }}
{{- end }}
{{- if .persistentVolumeClaim }}
persistentVolumeClaim:
  {{- toYaml .persistentVolumeClaim | nindent 2 }}
{{- end }}
{{- if hasKey . "emptyDir" }}
emptyDir:
  {{- toYaml (.emptyDir | default dict) | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-common.volumeMounts" -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- range $mount := ($volume.mounts | default list) }}
- name: {{ $volume.name | quote }}
  mountPath: {{ $mount.mountPath | quote }}
  {{- with $mount.subPath }}
  subPath: {{ . | quote }}
  {{- end }}
  {{- if $mount.readOnly }}
  readOnly: true
  {{- end }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.assertRunsRootIsMounted" -}}
{{- $runsRoot := (.Values.infra.paths | default dict).runsRoot | default "" -}}
{{- if $runsRoot }}
{{- $writable := list -}}
{{- $readOnly := list -}}
{{- $all := list -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- range $mount := ($volume.mounts | default list) }}
{{- $all = append $all $mount.mountPath }}
{{- if or (eq $runsRoot $mount.mountPath) (hasPrefix (printf "%s/" $mount.mountPath) $runsRoot) }}
{{- if hasKey $volume "emptyDir" }}
{{- fail (printf "infra.paths.runsRoot is %s, which falls under the emptyDir volume %s mounted at %s: an emptyDir belongs to a single pod, so the launcher, the orchestrator and every worker would each write into a directory of their own and no run would ever report a verdict" $runsRoot $volume.name $mountPath) }}
{{- end }}
{{- if $mount.readOnly }}
{{- $readOnly = append $readOnly $mount.mountPath }}
{{- else }}
{{- $writable = append $writable $mount.mountPath }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
{{- if not $writable }}
{{- if $readOnly }}
{{- fail (printf "infra.paths.runsRoot is %s, which only falls under the read-only mount %s: every run writes its state, values and exit file there" $runsRoot (join ", " $readOnly)) }}
{{- end }}
{{- fail (printf "infra.paths.runsRoot is %s, which falls under none of the infra.volumes mounts (%s), so a run would write its state file into the container's own filesystem where the launcher never sees it" $runsRoot (join ", " $all)) }}
{{- end }}
{{- end }}
{{- end }}
