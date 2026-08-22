{{- define "miles-run.colocateRole" -}}
{{- $colocate := .context.Values.run.colocate | default dict -}}
{{- $poolId := .poolId -}}
{{- $isInference := false -}}
{{- range $colocate.inference_pools }}{{- if eq .pool_id $poolId }}{{- $isInference = true }}{{- end }}{{- end }}
{{- if $isInference }}inference{{ else if eq $poolId ($colocate.trainer_pool_id | default "") }}trainer{{ end }}
{{- end }}

{{- define "miles-run.pool" -}}
{{- $context := .context }}
{{- $pool := .pool }}
{{- $role := include "miles-run.colocateRole" (dict "context" $context "poolId" (default $pool.name $pool.poolId)) }}
{{- $gated := eq $role "inference" }}
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
            {{- include "miles-run.containerDefaultsWith" (dict "context" $context "extraMounts" (include "miles-run.shmVolumeMount" $context)) | nindent 12 }}
            command:
              {{- range $pool.command }}
              - {{ . | quote }}
              {{- end }}
            {{- $entry := $pool }}
            {{- if $gated }}
            {{- $entry = merge (dict "env" (merge (dict "NVIDIA_VISIBLE_DEVICES" "all") (deepCopy ($pool.env | default dict)))) (deepCopy $pool) }}
            {{- end }}
            env:
              {{- include "miles-run.labelEnv" (dict "name" "MILES_CELL_INDEX" "label" "leaderworkerset.sigs.k8s.io/group-index") | trim | nindent 14 }}
              {{- include "miles-run.labelEnv" (dict "name" "MILES_POD_INDEX" "label" "leaderworkerset.sigs.k8s.io/worker-index") | trim | nindent 14 }}
              {{- if $gated }}
              {{- include "miles-run.annotationEnv" (dict "name" "MILES_BASE_GPU_ID" "annotation" "miles.radixark.io/base-gpu-id") | trim | nindent 14 }}
              {{- end }}
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
        {{- $volumes := compact (list (include "miles-common.volumes" $context | trim) (include "miles-run.shmVolume" $context | trim)) | join "\n" }}
        {{- with $volumes }}
        volumes:
          {{- . | nindent 10 }}
        {{- end }}
{{- end }}
