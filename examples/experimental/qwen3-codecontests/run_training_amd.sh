#!/usr/bin/env bash
#
# Usage:
#   bash run_training_amd.sh                  # full flow end-to-end
#   WANDB_KEY=xxx bash run_training_amd.sh
#   RUNTIME=/data/$USER/cc bash run_training_amd.sh   # runtime data on a big disk
#   bash run_training_amd.sh --fresh-data     # wipe extracted tasks + jsonls, rebuild
#   bash run_training_amd.sh --skip-setup     # reuse running containers / installed miles
#   bash run_training_amd.sh --skip-data      # reuse prepared data + model
#   bash run_training_amd.sh --reset          # restart the trainer between runs, then exit
#   bash run_training_amd.sh --teardown       # remove containers + task sandboxes, then exit
#   bash run_training_amd.sh --help
#
# Any flags after `--` pass straight through to run-qwen3-codecontests.py, e.g.:
#   bash run_training_amd.sh -- --num-rollout 50 --save-interval 5
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Config — all paths derived from the repo; every value overridable via env.
# --------------------------------------------------------------------------- #
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
EX="${EX:-$REPO_DIR/examples/experimental/qwen3-codecontests}"
# Runtime data (extracted tasks, Harbor trials, HF cache). Keep it OFF the repo so
# growing data never bloats a checkout; override with a big-disk path when available.
RUNTIME="${RUNTIME:-$EX/runtime}"
WORK_DIR="${WORK_DIR:-$RUNTIME/work}"                 # logs + Harbor trials (cc_trials/)
TASKS_DIR="${TASKS_DIR:-$RUNTIME/harbor_tasks_cc}"    # extracted Harbor task dirs
HF_CACHE="${HF_CACHE:-$RUNTIME/cache/hf}"             # HuggingFace cache (can be large)

# Container images (mirror the notebook; override for your registry/tags).
MILES_IMAGE="${MILES_IMAGE:-rlsys/miles:MI350-355-latest}"
AGENT_IMAGE="${AGENT_IMAGE:-aigmkt/aai-2026-harbor:v1}"
CC_BASE_IMAGE="${CC_BASE_IMAGE:-cc_base:v1}"          # shared task-sandbox base (fast spawn)

# ROCm device groups on the host (video/render). Override if your host differs.
VIDEO_GID="${VIDEO_GID:-44}"
RENDER_GID="${RENDER_GID:-109}"

# W&B: from the environment only — never hard-code a key. Empty => W&B disabled.
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"

# Model + training hyperparameters (match the notebook's async run; override via env
# or passthrough flags after --).
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3-30B-A3B}"
PROMPT_DATA="${PROMPT_DATA:-$EX/data/cc_train_all_sorted.jsonl}"  # full curriculum, easy->hard
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-4}"
NUM_EPOCH="${NUM_EPOCH:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3999}"
MAX_CONCURRENT="${MAX_CONCURRENT:-128}"               # Harbor trial concurrency

DOCKER="${DOCKER:-docker}"                            # set to "sudo docker" if needed

SKIP_SETUP=0
SKIP_DATA=0
FRESH_DATA=0
RESET_ONLY=0
TEARDOWN_ONLY=0
PASSTHRU=()

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-data)  SKIP_DATA=1; shift ;;
    --fresh-data) FRESH_DATA=1; shift ;;
    --reset)      RESET_ONLY=1; shift ;;
    --teardown)   TEARDOWN_ONLY=1; shift ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --) shift; PASSTHRU=("$@"); break ;;
    *) echo "unknown arg: $1 (use --help)"; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
note() { printf '\033[2m%s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- #
# reset / teardown (notebook cells 27 / 28)
# --------------------------------------------------------------------------- #
reset_trainer() {
  log "Reset trainer (restart miles_swe)"
  $DOCKER restart miles_swe
  $DOCKER exec miles_swe bash -lc 'rm -rf /tmp/ray/session_* 2>/dev/null; rm -f /app/aiter/aiter/jit/build/lock_* 2>/dev/null' || true
  echo "miles_swe reset; live ray/sglang: $($DOCKER exec miles_swe bash -lc 'pgrep -fc "raylet|sglang|train.py" 2>/dev/null' || echo 0)"
}

teardown() {
  log "Teardown (remove containers + task sandboxes)"
  $DOCKER rm -f miles_swe agent_env 2>/dev/null || true
  $DOCKER rm -f "$($DOCKER ps -aq --filter 'name=code_contests-')" 2>/dev/null || true
}

if [ "$RESET_ONLY" = 1 ]; then reset_trainer; exit 0; fi
if [ "$TEARDOWN_ONLY" = 1 ]; then teardown; exit 0; fi

export REPO_DIR EX RUNTIME WORK_DIR TASKS_DIR HF_CACHE WANDB_KEY
mkdir -p "$WORK_DIR" "$TASKS_DIR" "$HF_CACHE"

log "Config"
printf '%-12s= %s\n' REPO_DIR "$REPO_DIR" EX "$EX" RUNTIME "$RUNTIME" \
  WORK_DIR "$WORK_DIR" TASKS_DIR "$TASKS_DIR" HF_CACHE "$HF_CACHE" \
  MILES_IMAGE "$MILES_IMAGE" AGENT_IMAGE "$AGENT_IMAGE" PROMPT_DATA "$PROMPT_DATA"
echo "WANDB_KEY   = $([ -n "$WANDB_KEY" ] && echo set || echo '(empty -> W&B disabled)')"

# --------------------------------------------------------------------------- #
# 1+2+3. Host setup, launch containers, install miles (notebook cells 4, 6, 7, 9)
# --------------------------------------------------------------------------- #
if [ "$SKIP_SETUP" = 0 ]; then
  log "1. Host setup & clean slate"
  $DOCKER rm -f miles_swe agent_env 2>/dev/null || true
  $DOCKER rm -f "$($DOCKER ps -aq --filter 'name=code_contests-')" 2>/dev/null || true
  $DOCKER network create swe-net 2>/dev/null || echo "swe-net exists"

  log "2. Launch containers (miles_swe + agent_env)"
  # Repo AND $RUNTIME are IDENTITY-mounted (host path == container path) so work/
  # tasks/trials resolve in the sibling task containers Harbor spawns. All caches
  # live under $RUNTIME so growing data never fills '/'.
  $DOCKER run -d \
    --name miles_swe \
    --network swe-net \
    --hostname miles_swe \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --memory=0 \
    --memory-swap=0 \
    --privileged \
    --ulimit nofile=65535:65535 \
    -e WANDB_API_KEY="$WANDB_KEY" \
    -e HF_HOME="$HF_CACHE" \
    -v "$REPO_DIR":"$REPO_DIR" \
    -v "$RUNTIME":"$RUNTIME" \
    "$MILES_IMAGE" sleep infinity

  $DOCKER run -d --name agent_env --hostname agent_env \
    --network swe-net \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$REPO_DIR":"$REPO_DIR" \
    -v "$RUNTIME":"$RUNTIME" \
    "$AGENT_IMAGE" sleep infinity

  $DOCKER ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' | grep -E 'miles_swe|agent_env'

  log "3. Install miles in miles_swe"
  $DOCKER exec -e REPO_DIR="$REPO_DIR" miles_swe bash -lc 'cd "$REPO_DIR" && pip install -e . --no-deps --no-build-isolation'
  $DOCKER exec miles_swe bash -lc 'python3 -c "import miles, miles_plugins.mbridge"' && echo "miles import ok"
else
  note "--skip-setup: reusing running containers + installed miles"
fi

# --------------------------------------------------------------------------- #
# 4. Start Harbor (detached background service); wait for /health (cell 11)
# --------------------------------------------------------------------------- #
log "4. Start Harbor (background)"
if $DOCKER exec miles_swe bash -lc 'curl -sf http://agent_env:11000/health' >/dev/null 2>&1; then
  note "Harbor already responding; not relaunching"
else
  $DOCKER exec -d -e EX="$EX" -e REPO_DIR="$REPO_DIR" -e WORK_DIR="$WORK_DIR" -e TASKS_DIR="$TASKS_DIR" agent_env bash -lc ' \
      export OPENAI_API_KEY=dummy \
      MSWEA_API_KEY=dummy \
      MSWEA_CONFIG_FILE=$EX/harbor/codecontests.yaml \
      HARBOR_EXTRA_DOCKER_COMPOSE=$EX/harbor/swe_net_override.yaml \
      HARBOR_DELETE_CONTAINERS=false \
      HARBOR_ASYNC_TEARDOWN=1 \
      HARBOR_TASKS_DIR=$TASKS_DIR \
      HARBOR_TRIALS_DIR=$WORK_DIR/cc_trials; \
      cd $EX && PYTHONPATH=$REPO_DIR python3 harbor/server.py --port 11000 --max-concurrent '"$MAX_CONCURRENT"' > $WORK_DIR/harbor.log 2>&1'

  # health check runs from inside miles_swe (host can't reach swe-net directly)
  if $DOCKER exec miles_swe bash -lc 'for i in $(seq 1 30); do python3 -c "import urllib.request; urllib.request.urlopen(\"http://agent_env:11000/health\")" && exit 0; sleep 2; done; exit 1'; then
    echo " <- harbor up"
  else
    echo "harbor NOT up; last 15 lines of harbor.log:"; tail -n 15 "$WORK_DIR/harbor.log" 2>/dev/null || true
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# 5+6. Data prep, task base image, model pre-fetch (cells 13/14/16/21; guarded)
# --------------------------------------------------------------------------- #
if [ "$SKIP_DATA" = 0 ]; then
  if [ "$FRESH_DATA" = 1 ]; then
    log "5.0 Fresh data prep: wiping extracted tasks + jsonls + HF dataset cache"
    $DOCKER exec -e EX="$EX" -e TASKS_DIR="$TASKS_DIR" -e HF_CACHE="$HF_CACHE" miles_swe bash -lc '
      rm -rf "$TASKS_DIR"/code_contests-*
      rm -f  "$EX"/data/cc_train_*.jsonl
      rm -rf "$HF_CACHE"/hub/datasets--open-thoughts--CodeContests*
      echo "after clean: tasks=$(ls "$TASKS_DIR" 2>/dev/null | wc -l)  jsonls=$(ls "$EX"/data/cc_train_*.jsonl 2>/dev/null | wc -l)"
    '
  fi

  log "5. Data preparation (download + extract + build jsonls; guarded)"
  $DOCKER exec -e EX="$EX" -e TASKS_DIR="$TASKS_DIR" -e HF_CACHE="$HF_CACHE" -e HF_HOME="$HF_CACHE" miles_swe bash -lc '
    [ -d "$TASKS_DIR"/code_contests-0000 ] || \
      python3 "$EX"/data_prep/extract_codecontests.py --dataset open-thoughts/CodeContests --out "$TASKS_DIR"
    # build_cc_jsonl.py emits the per-difficulty splits AND the combined
    # cc_train_all_sorted.jsonl (increasing difficulty, unrated last) in one pass.
    [ -f "$EX"/data/cc_train_all_sorted.jsonl ] || \
      python3 "$EX"/data_prep/build_cc_jsonl.py --tasks "$TASKS_DIR" --out-dir "$EX"/data
  '

  log "5b. Faster sandbox spawn: build task base image $CC_BASE_IMAGE + rewrite task Dockerfiles"
  # Bake python3/pip once into a shared base image on the HOST docker daemon (the one
  # Harbor uses via the mounted socket), then rewrite each task Dockerfile to FROM it
  # (no per-task apt) so environment_setup drops toward the ~5s container-create floor.
  if ! $DOCKER image inspect "$CC_BASE_IMAGE" >/dev/null 2>&1; then
    $DOCKER build -t "$CC_BASE_IMAGE" - <<'EOF'
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
EOF
  else
    note "task base image $CC_BASE_IMAGE already present"
  fi
  $DOCKER exec -e EX="$EX" -e TASKS_DIR="$TASKS_DIR" -e CC_BASE_IMAGE="$CC_BASE_IMAGE" miles_swe bash -lc \
    'python3 "$EX"/data_prep/extract_codecontests.py --rewrite-only --out "$TASKS_DIR" --base-image "$CC_BASE_IMAGE"'

  log "6. Pre-fetch $HF_CHECKPOINT into HF cache"
  $DOCKER exec -e HF_HOME="$HF_CACHE" -e HF_CHECKPOINT="$HF_CHECKPOINT" miles_swe bash -lc \
    'python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"HF_CHECKPOINT\"])"'
else
  note "--skip-data: reusing prepared data + model"
fi

# --------------------------------------------------------------------------- #
# 7. Training run (disaggregated async GRPO, 4 train + 4 rollout GPUs) — cell 23.
# Foreground, streamed to terminal + $WORK_DIR/train.log. Use `--reset` between runs.
# --------------------------------------------------------------------------- #
log "7. Training run (foreground; also logging to $WORK_DIR/train.log)"
$DOCKER exec -e EX="$EX" -e REPO_DIR="$REPO_DIR" -e WORK_DIR="$WORK_DIR" -e TASKS_DIR="$TASKS_DIR" \
  -e HF_CACHE="$HF_CACHE" -e HF_HOME="$HF_CACHE" -e WANDB_KEY="$WANDB_KEY" \
  -e HF_CHECKPOINT="$HF_CHECKPOINT" -e PROMPT_DATA="$PROMPT_DATA" \
  -e NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" -e TRAIN_NUM_GPUS="$TRAIN_NUM_GPUS" \
  -e NUM_EPOCH="$NUM_EPOCH" -e ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
  -e N_SAMPLES_PER_PROMPT="$N_SAMPLES_PER_PROMPT" -e GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  -e SAVE_INTERVAL="$SAVE_INTERVAL" \
  miles_swe bash -lc 'cd "$EX"; \
    export WANDB_KEY=$WANDB_KEY \
    MILES_ROUTER_EXTERNAL_HOST=miles_swe \
    AGENT_SERVER_URL=http://agent_env:11000 \
    HARBOR_TASKS_DIR=$TASKS_DIR \
    WANDB_DIR=$WORK_DIR/wandb; \
    PYTHONPATH=$REPO_DIR python3 run-qwen3-codecontests.py \
    --async-mode \
    --num-gpus-per-node "$NUM_GPUS_PER_NODE" \
    --train-num-gpus "$TRAIN_NUM_GPUS" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --prompt-data "$PROMPT_DATA" \
    --num-epoch "$NUM_EPOCH" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-interval "$SAVE_INTERVAL" '"${PASSTHRU[*]:-}"' 2>&1' \
  | tee "$WORK_DIR/train.log"

log "Training finished. Trials under $WORK_DIR/cc_trials  |  log: $WORK_DIR/train.log"
