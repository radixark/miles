{{- define "miles-run.colocateRole" -}}
{{- $colocate := .context.Values.run.colocate | default dict -}}
{{- if $colocate.enabled -}}
{{- if eq .name $colocate.enginePool }}engine{{ else if eq .name $colocate.trainerPool }}trainer{{ end }}
{{- end }}
{{- end }}

{{- define "miles-run.pool" -}}
{{- $context := .context }}
{{- $pool := .pool }}
{{- $role := include "miles-run.colocateRole" (dict "context" $context "name" $pool.name) }}
{{- $gated := eq $role "engine" }}
{{- $name := $pool.objectName }}
{{- $labels := dict "context" $context "component" $pool.name }}
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: {{ $name | quote }}
  namespace: {{ $context.Release.Namespace | quote }}
  labels:
    {{- include "miles-run.labels" $labels | nindent 4 }}
spec:
  replicas: {{ default 1 $pool.replicas }}
  startupPolicy: LeaderCreated
  leaderWorkerTemplate:
    size: {{ default 1 $pool.size }}
    restartPolicy: RecreateGroupOnPodRestart
    workerTemplate:
      metadata:
        labels:
          {{- include "miles-run.labels" $labels | nindent 10 }}
          miles.radixark.io/pool: {{ default $pool.name $pool.poolId | quote }}
        {{- with $pool.meta }}
        annotations:
          {{- range $key, $value := . }}
          miles.radixark.io/meta-{{ $key }}: {{ $value | quote }}
          {{- end }}
        {{- end }}
      spec:
        {{- include "miles-run.podDefaultsFor" (dict "context" $context "gated" $gated) | nindent 8 }}
        {{- if $role }}
        hostIPC: true
        {{- end }}
        {{- if $gated }}
        schedulingGates:
          - name: "miles.radixark.io/colocate-pairing"
        {{- end }}
        containers:
          - name: {{ $pool.containerName | default "worker" | quote }}
            {{- include "miles-run.containerDefaults" $context | nindent 12 }}
            command:
              {{- range $pool.command }}
              - {{ . | quote }}
              {{- end }}
            {{- $entry := $pool }}
            {{- if $gated }}
            {{- $entry = merge (dict "env" (merge (dict "NVIDIA_VISIBLE_DEVICES" "all") (deepCopy ($pool.env | default dict)))) (deepCopy $pool) }}
            {{- end }}
            env:
              {{- include "miles-run.cellIndexEnv" (dict "label" "leaderworkerset.sigs.k8s.io/group-index") | trim | nindent 14 }}
              {{- with include "miles-run.envItems" (dict "context" $context "entry" $entry) | trim }}
              {{- . | nindent 14 }}
              {{- end }}
            {{- with $pool.ports }}
            ports:
              {{- range . }}
              - name: {{ .name | quote }}
                containerPort: {{ .port }}
              {{- end }}
            {{- end }}
            {{- $resources := default dict $pool.resources }}
            {{- if $gated }}
            {{- $limits := deepCopy ($resources.limits | default dict) }}
            {{- $resources = deepCopy $resources }}
            {{- $_ := set $limits "nvidia.com/gpu" 0 }}
            {{- $ignored := set $resources "limits" $limits }}
            {{- end }}
            resources:
              {{- toYaml $resources | nindent 14 }}
        {{- $volumes := compact (list (include "miles-common.sharedStorageVolume" $context | trim) (include "miles-run.nodeLocalVolume" $context | trim)) | join "\n" }}
        {{- with $volumes }}
        volumes:
          {{- . | nindent 10 }}
        {{- end }}
{{- end }}
