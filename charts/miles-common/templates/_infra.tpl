{{- define "miles-common.image" -}}
{{- $image := .Values.image | default dict -}}
{{- printf "%s:%s" ($image.repository | default "") ($image.tag | default "") | quote }}
{{- end }}

{{- define "miles-common.imagePullSecrets" -}}
{{- with (.Values.image | default dict).pullSecrets -}}
imagePullSecrets:
{{- range . }}
  - name: {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.scheduling" -}}
{{- with (.Values.scheduling | default dict).nodeSelector -}}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with (.Values.scheduling | default dict).tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with (.Values.scheduling | default dict).affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "miles-common.env" -}}
{{- with .Values.env -}}
env:
{{- range $name, $value := . }}
  - name: {{ $name | quote }}
    value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.sharedStorageVolume" -}}
{{- if ne ((.Values.sharedStorage | default dict).type | default "none") "none" -}}
- name: shared-storage
  {{- if eq (.Values.sharedStorage | default dict).type "hostPath" }}
  hostPath:
    path: {{ (.Values.sharedStorage | default dict).hostPath | quote }}
    type: Directory
  {{- else }}
  persistentVolumeClaim:
    claimName: {{ (.Values.sharedStorage | default dict).pvcClaimName | quote }}
  {{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.sharedStorageVolumeMount" -}}
{{- if ne ((.Values.sharedStorage | default dict).type | default "none") "none" -}}
- name: shared-storage
  mountPath: {{ (.Values.sharedStorage | default dict).mountPath | quote }}
{{- end }}
{{- end }}
