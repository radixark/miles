#!/usr/bin/env bash
# Submit a rendered JobSet (render_jobset.py), wait for Kueue to admit it, stream its pod's log and exit with its
# verdict. Adapted from sgl-project/sglang-torchtpu .github/workflows/_tpu-test.yml.
#
# Usage: run_jobset.sh <jobset.json> <label> <run-budget-seconds>
# Env: WIF_PROVIDER and CI_SA (to re-mint the cluster token), plus the GitHub OIDC request variables.
set -euo pipefail

job_file=$1
label=$2
run_budget_s=$3
ns=tpu-ci
# a hosted runner's job dies at 6 hours; queueing comes out of the same budget
job_ceiling_s=21600

# The kubeconfig holds a one-hour token minted at job start and org policy forbids longer ones; a job that queues
# outlives it, so re-mint it from a fresh GitHub OIDC token (the exchange google-github-actions/auth does).
refresh_cluster_token() {
  local oidc sts sa_token user
  oidc="$(curl -sSf -H "Authorization: bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN}" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=https://iam.googleapis.com/${WIF_PROVIDER}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')" || return 1
  sts="$(curl -sSf -X POST https://sts.googleapis.com/v1/token \
    -H 'Content-Type: application/json' \
    -d "{\"audience\":\"//iam.googleapis.com/${WIF_PROVIDER}\",
         \"grantType\":\"urn:ietf:params:oauth:grant-type:token-exchange\",
         \"requestedTokenType\":\"urn:ietf:params:oauth:token-type:access_token\",
         \"scope\":\"https://www.googleapis.com/auth/cloud-platform\",
         \"subjectTokenType\":\"urn:ietf:params:oauth:token-type:jwt\",
         \"subjectToken\":\"${oidc}\"}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')" || return 1
  # userinfo.email: without it GKE sees a numeric id and RBAC denies
  sa_token="$(curl -sSf -X POST \
    "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${CI_SA}:generateAccessToken" \
    -H "Authorization: Bearer ${sts}" -H 'Content-Type: application/json' \
    -d '{"scope":["https://www.googleapis.com/auth/cloud-platform","https://www.googleapis.com/auth/userinfo.email"],"lifetime":"3600s"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["accessToken"])')" || return 1
  user="$(kubectl config view --minify -o jsonpath='{.contexts[0].context.user}')"
  kubectl config set-credentials "$user" --token "$sa_token" >/dev/null
  echo "refreshed cluster credentials at $(date -u +%FT%TZ)"
}

# refresh every 40 minutes, or 45 s after a failed attempt
next_refresh=$(($(date +%s) + 2400))
maybe_refresh() {
  [ "$(date +%s)" -ge "$next_refresh" ] || return 0
  if refresh_cluster_token; then
    next_refresh=$(($(date +%s) + 2400))
  else
    echo "credential refresh failed; retrying in 45s"
    next_refresh=$(($(date +%s) + 45))
  fi
}

name="$(kubectl -n "$ns" create -f "$job_file" -o name)"
echo "created ${name}"
# a cancelled run's runner gets SIGTERM; exit through the EXIT trap so the JobSet goes too
trap 'kubectl -n '"$ns"' delete '"$name"' --ignore-not-found --wait=false' EXIT
trap 'exit 143' TERM INT
short="${name##*/}"
sel="jobset.sigs.k8s.io/jobset-name=${short}"

# No Workload means Kueue is not managing the JobSet: it would run with no admission control. Fail closed.
uid="$(kubectl -n "$ns" get "$name" -o jsonpath='{.metadata.uid}')"
wl_sel="kueue.x-k8s.io/job-uid=${uid}"
for _ in $(seq 1 15); do
  [ -n "$(kubectl -n "$ns" get workloads -l "$wl_sel" -o name 2>/dev/null)" ] && break
  sleep 2
done
if [ -z "$(kubectl -n "$ns" get workloads -l "$wl_sel" -o name 2>/dev/null)" ]; then
  echo "❌ ${label}: no Kueue Workload for ${short}; check the queue-name label and the LocalQueue"
  kubectl -n "$ns" get localqueues 2>/dev/null || true
  exit 1
fi

# What is left of the runner's 6 hours after the run itself, image pull (900 s) and setup (600 s) goes to queueing.
admit_s=$((job_ceiling_s - run_budget_s - 900 - 600))
echo "budgets for ${label}: admission ${admit_s}s, runtime ${run_budget_s}s"
echo "workloads in ${ns}:"
kubectl -n "$ns" get workloads 2>/dev/null || true
admit_deadline=$(($(date +%s) + admit_s))
auth_fails=0
while true; do
  # a dead kubectl looks like "no pods yet" with errors suppressed; count failures and re-mint credentials
  if pods="$(kubectl -n "$ns" get pods -l "$sel" -o name 2>&1)"; then
    auth_fails=0
    [ -n "$pods" ] && break
  else
    auth_fails=$((auth_fails + 1))
    if [ "$auth_fails" -ge 3 ]; then
      refresh_cluster_token || echo "credential refresh failed; will retry"
      auth_fails=0
    fi
  fi
  maybe_refresh
  if [ "$(date +%s)" -ge "$admit_deadline" ]; then
    echo "❌ ${label} NOT ADMITTED after ${admit_s}s: no CI host freed up"
    kubectl -n "$ns" get workloads 2>/dev/null || true
    exit 1
  fi
  sleep 10
done

# admitted; the pod still binds and pulls the image, which is not charged to the run
running_deadline=$(($(date +%s) + 900))
until kubectl -n "$ns" get pods -l "$sel" -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -q Running; do
  if [ "$(date +%s)" -ge "$running_deadline" ]; then
    echo "⚠️ ${label}: no pod Running 900s after admission; continuing"
    break
  fi
  sleep 5
done
kubectl -n "$ns" get pods -l "$sel" -o wide || true
deadline=$(($(date +%s) + run_budget_s))

# Stream the pod's log, re-attaching while the JobSet runs: the log is all that leaves the pod.
refresh_cluster_token || echo "credential refresh failed; continuing with the current token"
next_refresh=$(($(date +%s) + 2400))
log="${RUNNER_TEMP:-/tmp}/tpu-pod-logs.txt"
rm -f "$log"
since=""
while kubectl -n "$ns" get "$name" >/dev/null 2>&1 && [ "$(date +%s)" -lt "$deadline" ]; do
  conditions="$(kubectl -n "$ns" get "$name" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
  [[ " $conditions " == *" Completed "* || " $conditions " == *" Failed "* ]] && break
  if [ -n "$since" ]; then
    echo "--- [$(date +%T)] log stream ended while ${label} is still running; re-attaching ---"
    refresh_cluster_token || echo "credential refresh failed; continuing with the current token"
    next_refresh=$(($(date +%s) + 2400))
    range=(--since-time="$since")
  else
    range=(--tail=-1)
  fi
  kubectl -n "$ns" logs -f -l "$sel" --all-containers=true --pod-running-timeout=15m "${range[@]}" 2>&1 \
    | tee -a "$log" || true
  since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 10
done

# verdict: the JobSet's Completed or Failed condition
auth_fails=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  maybe_refresh
  if ! conditions="$(kubectl -n "$ns" get "$name" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)"; then
    auth_fails=$((auth_fails + 1))
    [ "$auth_fails" -eq 3 ] && { refresh_cluster_token || true; }
    if [ "$auth_fails" -ge 12 ]; then
      echo "❌ ${label}: lost cluster access while polling; JobSet outcome unknown"
      exit 1
    fi
    sleep 5
    continue
  fi
  auth_fails=0
  if [[ " $conditions " == *" Completed "* ]]; then
    echo "✅ ${label} PASSED"
    exit 0
  fi
  if [[ " $conditions " == *" Failed "* ]]; then
    echo "❌ ${label} FAILED"
    kubectl -n "$ns" get pods -l "$sel" -o wide || true
    kubectl -n "$ns" logs -l "$sel" --all-containers --tail=150 2>/dev/null || true
    kubectl -n "$ns" get events --sort-by=.lastTimestamp 2>/dev/null | tail -n 40 || true
    kubectl -n "$ns" describe "$name" | tail -n 100 || true
    exit 1
  fi
  sleep 5
done
echo "❌ ${label} TIMED OUT after ${run_budget_s}s"
kubectl -n "$ns" get pods -l "$sel" -o wide || true
kubectl -n "$ns" describe "$name" | tail -n 100 || true
exit 1
