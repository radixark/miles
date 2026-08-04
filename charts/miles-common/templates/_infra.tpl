{{- define "miles-common.image" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag | quote }}
{{- end }}

{{- define "miles-common.imagePullSecrets" -}}
{{- with .Values.image.pullSecrets -}}
imagePullSecrets:
{{- range . }}
  - name: {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.scheduling" -}}
{{- with .Values.scheduling.nodeSelector -}}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.scheduling.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.scheduling.affinity }}
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
{{- if ne .Values.sharedStorage.type "none" -}}
- name: shared-storage
  {{- if eq .Values.sharedStorage.type "hostPath" }}
  hostPath:
    path: {{ .Values.sharedStorage.hostPath | quote }}
    type: Directory
  {{- else }}
  persistentVolumeClaim:
    claimName: {{ .Values.sharedStorage.pvcClaimName | quote }}
  {{- end }}
{{- end }}
{{- end }}

{{- define "miles-common.sharedStorageVolumeMount" -}}
{{- if ne .Values.sharedStorage.type "none" -}}
- name: shared-storage
  mountPath: {{ .Values.sharedStorage.mountPath | quote }}
{{- end }}
{{- end }}
