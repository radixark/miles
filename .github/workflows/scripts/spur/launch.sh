#!/usr/bin/env bash
# Run one ROCm CI shard on an MI35x node scheduled through Spur (Slurm).
#
# Runs on the login-node runner: submits job.sh, streams the job's log into the
# step output, and exits with the job's exit code. The same approach as
# SGLang's scripts/ci/slurm/launch_amd_ci_mi35x.sh, which handles these Spur
# behaviours:
#   * batch output is not streamed back, so the job logs to shared storage and
#     this script re-reads that file by offset each poll (`tail -F` misses
#     appends made on another host over NFS);
#   * a job can be cancelled or held (JobHoldMaxRequeue) before it starts, so a
#     job that never started is resubmitted, up to SUBMIT_ATTEMPTS times;
#   * the controller sometimes answers with an empty queue, so one missing
#     state reading is not taken as "job finished";
#   * a two-field time limit parses as HH:MM, so TIME_LIMIT must be HH:MM:SS;
#   * the job's exit-code file can stay invisible on the login host for up to
#     a minute after the node writes it (NFS negative lookup caching).
#
# Usage: launch.sh [--gpus 1|2|4|8]
#   The workflow step provides CI_IMAGE and CI_COMMAND plus the variables
#   job.sh passes into the container; sbatch exports them to the job.
#
# Runner environment (the runner's .env):
#   SLURM_PARTITION, SLURM_ACCOUNT, SLURM_QOS
#   CI_NODE_SCRATCH  node-local directory for the checkout copy and the
#                    persistent model / dataset / HF cache
# Optional:
#   SLURM_EXCLUDE    comma-separated nodes to keep the scheduler off
#   SLURM_WHOLE_NODE_ONLY
#                    nodes kept for 8-GPU jobs; smaller jobs stay off them
#   TIME_LIMIT       default 03:00:00
#   CI_SHARED_DIR    shared-storage dir for job logs, default $HOME/.miles_ci_spur
#   SUBMIT_ATTEMPTS  default 3
#   HF_TOKEN, DOCKERHUB_USERNAME, DOCKERHUB_TOKEN

set -euo pipefail

GPUS=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            if [[ ! "$2" =~ ^(1|2|4|8)$ ]]; then
                echo "ERROR: --gpus takes 1, 2, 4 or 8 (got '$2')" >&2
                exit 2
            fi
            GPUS="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

: "${CI_IMAGE:?set by the workflow step}"
: "${CI_COMMAND:?set by the workflow step}"
: "${SLURM_PARTITION:?set in the runner environment}"
: "${SLURM_ACCOUNT:?set in the runner environment}"
: "${SLURM_QOS:?set in the runner environment}"
: "${CI_NODE_SCRATCH:?set in the runner environment}"
: "${GITHUB_WORKSPACE:?}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
SUBMIT_ATTEMPTS="${SUBMIT_ATTEMPTS:-3}"
if [[ ! "$TIME_LIMIT" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "ERROR: TIME_LIMIT must be HH:MM:SS or D-HH:MM:SS (got '$TIME_LIMIT'); Spur reads '30:00' as 30 hours" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# _run-ci-rocm.yml's cleanup step cancels by this prefix; keep the two in sync.
JOB_PREFIX="miles-ci-${RUNNER_NAME:-local}-${GITHUB_RUN_ID:-0}-"
JOB_NAME="${JOB_PREFIX}${GITHUB_JOB:-job}-${GITHUB_RUN_ATTEMPT:-1}"
RUN_DIR="${CI_SHARED_DIR:-$HOME/.miles_ci_spur}/${JOB_NAME}"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

# Read by job.sh; sbatch exports the submitting environment.
export CI_SRC="$GITHUB_WORKSPACE" CI_NODE_SCRATCH CI_GPUS="$GPUS"
export CI_EXIT_FILE="$RUN_DIR/exit_code" CI_SUMMARY_FILE="$RUN_DIR/summary.md"

JOB_ID=""
cleanup() {
    if [[ -n "$JOB_ID" && ! -f "$CI_EXIT_FILE" ]]; then
        echo "Cancelling cluster job $JOB_ID"
        scancel "$JOB_ID" 2>/dev/null || true
    fi
    if [[ -s "$CI_SUMMARY_FILE" && -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        cat "$CI_SUMMARY_FILE" >> "$GITHUB_STEP_SUMMARY" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LOG_OFFSET=0
stream_log() {  # print what the job appended to $LOG since the last call
    local size
    size="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
    if (( size > LOG_OFFSET )); then
        tail -c +"$((LOG_OFFSET + 1))" "$LOG" | head -c "$((size - LOG_OFFSET))"
        LOG_OFFSET="$size"
    fi
}

wait_for_exit_file() {  # up to 90s; listing the directory forces a fresh NFS lookup
    local _
    for _ in $(seq 1 18); do
        ls "$RUN_DIR" > /dev/null 2>&1 || true
        [[ -s "$CI_EXIT_FILE" ]] && return 0
        sleep 5
    done
    return 1
}

job_state() {  # "<STATE> <REASON>", or empty if the controller does not know the job
    scontrol show job "$1" 2>/dev/null \
        | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^(JobState|Reason)=/) {sub(/^[^=]*=/, "", $i); printf "%s ", $i}}'
}

for attempt in $(seq 1 "$SUBMIT_ATTEMPTS"); do
    LOG="$RUN_DIR/attempt${attempt}.log"
    : > "$LOG"
    rm -f "$CI_EXIT_FILE"
    SBATCH_ARGS=(-p "$SLURM_PARTITION" -A "$SLURM_ACCOUNT" -q "$SLURM_QOS"
                 -N1 -t "$TIME_LIMIT" -J "$JOB_NAME" --output "$LOG")
    # A whole node stays --exclusive. Smaller jobs get only their GPUs
    # (ROCR_VISIBLE_DEVICES), so two 4-GPU shards can share a node.
    EXCLUDE="${SLURM_EXCLUDE:-}"
    if (( GPUS == 8 )); then
        SBATCH_ARGS+=(--exclusive)
    else
        SBATCH_ARGS+=(--gres="gpu:$GPUS")
        [[ -n "${SLURM_WHOLE_NODE_ONLY:-}" ]] && EXCLUDE="${EXCLUDE:+$EXCLUDE,}$SLURM_WHOLE_NODE_ONLY"
    fi
    [[ -n "$EXCLUDE" ]] && SBATCH_ARGS+=(--exclude="$EXCLUDE")
    JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/job.sh")"
    JOB_ID="${JOB_ID%%;*}"
    echo "Submitted job $JOB_ID (attempt $attempt/$SUBMIT_ATTEMPTS): gpus=$GPUS image=$CI_IMAGE"

    LOG_OFFSET=0
    misses=0
    started=0
    while true; do
        sleep 15
        stream_log
        # The job writes its exit code last; COMPLETING can linger after that.
        [[ -f "$CI_EXIT_FILE" ]] && break
        read -r state reason <<< "$(job_state "$JOB_ID")" || true
        case "${state:-}" in
            PENDING)
                misses=0
                if [[ "${reason:-}" == JobHold* ]]; then
                    echo "Job $JOB_ID held ($reason) before starting; cancelling"
                    scancel "$JOB_ID" 2>/dev/null || true
                    break
                fi ;;
            RUNNING|CONFIGURING|COMPLETING) misses=0; started=1 ;;
            "") misses=$((misses + 1)); (( misses >= 4 )) && break ;;
            *) break ;;
        esac
    done
    if [[ "${state:-}" == PENDING ]]; then
        sleep 5          # held before starting: no exit code is coming
    else
        wait_for_exit_file || true
    fi
    stream_log           # also picks up the last lines of the log

    if [[ -s "$CI_EXIT_FILE" ]]; then
        rc="$(tr -dc 0-9 < "$CI_EXIT_FILE")"
        echo "Job $JOB_ID finished with exit code ${rc:-unknown}"
        exit "${rc:-1}"
    fi
    if (( started )) || grep -q '^\[miles-ci-job\] start' "$LOG"; then
        echo "ERROR: job $JOB_ID started but ended (${state:-gone} ${reason:-}) without writing an exit code" >&2
        exit 1
    fi
    echo "Job $JOB_ID never started (${state:-gone} ${reason:-}); resubmitting"
    JOB_ID=""
done
echo "ERROR: no job started after $SUBMIT_ATTEMPTS attempts" >&2
exit 1
