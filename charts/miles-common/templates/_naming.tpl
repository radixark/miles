{{- define "miles-common.fullname" -}}
{{- if contains .Chart.Name .Release.Name }}
{{- .Release.Name | trunc 52 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 52 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "miles-common.componentName" -}}
{{- $context := .context -}}
{{- $budget := sub 52 (add1 (len .component)) -}}
{{- $prefix := include "miles-common.fullname" $context | trunc (int $budget) | trimSuffix "-" -}}
{{- printf "%s-%s" $prefix .component }}
{{- end }}

{{- define "miles-common.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "miles-common.selectorLabels" -}}
{{- $context := .context -}}
app.kubernetes.io/name: {{ $context.Chart.Name | quote }}
app.kubernetes.io/instance: {{ $context.Release.Name | quote }}
app.kubernetes.io/component: {{ .component | quote }}
{{- end }}

{{- define "miles-common.labels" -}}
{{- $context := .context -}}
helm.sh/chart: {{ include "miles-common.chart" $context | quote }}
{{ include "miles-common.selectorLabels" . }}
app.kubernetes.io/version: {{ $context.Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ $context.Release.Service | quote }}
{{- end }}
