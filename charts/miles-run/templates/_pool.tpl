{{- define "miles-run.pool" -}}
{{- $context := .context }}
{{- $pool := .pool }}
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
        {{- include "miles-run.podDefaults" $context | nindent 8 }}
        containers:
          - name: {{ $pool.containerName | default "worker" | quote }}
            {{- include "miles-run.containerDefaults" $context | nindent 12 }}
            command:
              {{- range $pool.command }}
              - {{ . | quote }}
              {{- end }}
            {{- $entry := $pool }}
            env:
              {{- include "miles-run.labelEnv" (dict "name" "MILES_CELL_INDEX" "label" "leaderworkerset.sigs.k8s.io/group-index") | trim | nindent 14 }}
              {{- include "miles-run.labelEnv" (dict "name" "MILES_POD_INDEX" "label" "leaderworkerset.sigs.k8s.io/worker-index") | trim | nindent 14 }}
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
            resources:
              {{- toYaml $resources | nindent 14 }}
        {{- $volumes := compact (list (include "miles-common.sharedStorageVolume" $context | trim) (include "miles-run.nodeLocalVolume" $context | trim)) | join "\n" }}
        {{- with $volumes }}
        volumes:
          {{- . | nindent 10 }}
        {{- end }}
{{- end }}
