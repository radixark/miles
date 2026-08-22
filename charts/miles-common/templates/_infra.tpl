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
{{- toYaml (.Values.infra.env | default dict) }}
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

{{- define "miles-common.interpolatePath" -}}
{{- $value := .value -}}
{{- range $reference := regexFindAll "\\$\\{[^}]*\\}" $value -1 -}}
{{- if ne $reference "${NAMESPACE}" -}}
{{- fail (printf "%s is %s, which names the unknown variable %s: ${NAMESPACE} is the only variable a path may name, and an unknown one left in place would give every namespace the same literal directory instead of one of its own" $.field $value $reference) -}}
{{- end -}}
{{- end -}}
{{- $value | replace "${NAMESPACE}" .root.Release.Namespace -}}
{{- end }}

{{- define "miles-common.assertInfraPathsResolve" -}}
{{- $_ := include "miles-common.volumes" . -}}
{{- $_ = include "miles-common.volumeMounts" . -}}
{{- $_ = include "miles-common.runsRoot" . -}}
{{- end }}

{{- define "miles-common.runsRoot" -}}
{{- with (.Values.infra.paths | default dict).runsRoot -}}
{{- include "miles-common.interpolatePath" (dict "root" $ "field" "infra.paths.runsRoot" "value" .) -}}
{{- end -}}
{{- end }}

{{- define "miles-common.volumes" -}}
{{- $root := . -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- $source := $volume }}
{{- with $volume.hostPath }}
{{- $path := include "miles-common.interpolatePath" (dict "root" $root "field" (printf "infra.volumes[%s].hostPath.path" $volume.name) "value" .path) }}
{{- $source = mustMergeOverwrite (deepCopy $volume) (dict "hostPath" (dict "path" $path)) }}
{{- end }}
- name: {{ $volume.name | quote }}
  {{- include "miles-common.volumeSource" $source | trim | nindent 2 }}
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
{{- $root := . -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- range $index, $mount := ($volume.mounts | default list) }}
- name: {{ $volume.name | quote }}
  mountPath: {{ include "miles-common.mountPath" (dict "root" $root "volume" $volume "index" $index "mount" $mount) | quote }}
  {{- with $mount.subPath }}
  subPath: {{ include "miles-common.interpolatePath" (dict "root" $root "field" (printf "infra.volumes[%s].mounts[%d].subPath" $volume.name $index) "value" .) | quote }}
  {{- end }}
  {{- if $mount.readOnly }}
  readOnly: true
  {{- end }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.mountPath" -}}
{{- include "miles-common.interpolatePath" (dict "root" .root "field" (printf "infra.volumes[%s].mounts[%d].mountPath" .volume.name (int .index)) "value" .mount.mountPath) -}}
{{- end }}

{{- define "miles-common.assertRunsRootIsMounted" -}}
{{- $root := . -}}
{{- $runsRoot := include "miles-common.runsRoot" . -}}
{{- if $runsRoot }}
{{- $writable := list -}}
{{- $readOnly := list -}}
{{- $all := list -}}
{{- range $volume := (.Values.infra.volumes | default list) }}
{{- range $index, $mount := ($volume.mounts | default list) }}
{{- $mountPath := include "miles-common.mountPath" (dict "root" $root "volume" $volume "index" $index "mount" $mount) }}
{{- $all = append $all $mountPath }}
{{- if or (eq $runsRoot $mountPath) (hasPrefix (printf "%s/" $mountPath) $runsRoot) }}
{{- if $mount.readOnly }}
{{- $readOnly = append $readOnly $mountPath }}
{{- else }}
{{- $writable = append $writable $mountPath }}
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
