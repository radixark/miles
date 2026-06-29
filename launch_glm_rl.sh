#!/usr/bin/env bash
# ============================================================================
#  launch_glm_rl.sh  — self-contained launcher for FULL GLM-5.2 (744B) LoRA RL.
#
#  Every parameter is set EXPLICITLY here (no reliance on the GLM-wrapper eval
#  wiring that was reverted in c5051fe18). Eval is enabled by passing the eval
#  flags straight through to miles `train.py` via --extra-args — those args live
#  in miles core (arguments.py), which the revert did NOT touch.
#
#  Drives the multi-node flow (manual Ray cluster + MILES_SCRIPT_EXTERNAL_RAY=1),
#  delegating per-role to scripts/run_glm5_lora.py for all GLM-specific handling
#  (KEEP_MOE_LORA target-drop, qkv-format from backend, INDEXER_ROPE_NEOX_STYLE).
#
#  THIS RUN:
#    * MODEL=GLM-5.2 (full 78L 744B-A40B)         * 8 nodes x 8 H200 = 64 GPU
#    * LoRA = ATTENTION-ONLY (KEEP_MOE_LORA=0)    * MB backend = slime (fused, thd)
#    * task=gsm8k, response-len=256               * --lora-base-cpu-backup (ON)
#    * eval ON (gsm8k held-out test split, every EVAL_INTERVAL steps)
#    * 10 rollout steps, SAVE every step (inspect LoRA per-step)
#    * wandb ONLINE (export WANDB_API_KEY before running)
#
#  USAGE (per pod, via `rx devbox run <box> --rank N -- bash -lc '...'`):
#    HEAD_IP=<rank0 bond0 ip> WANDB_API_KEY=<secret> bash launch_glm_rl.sh head
#    HEAD_IP=<rank0 bond0 ip>                         bash launch_glm_rl.sh worker   # on each other rank
#    HEAD_IP=<rank0 bond0 ip> WANDB_API_KEY=<secret> bash launch_glm_rl.sh launch    # on rank 0, after 8 nodes joined
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" && "$ROLE" != "launch" ]]; then
  echo "usage: $0 {head|worker|launch}" >&2; exit 2
fi
cd "$(dirname "$0")"   # miles repo root (so scripts/run_glm5_lora.py resolves)

# ---- topology / model -------------------------------------------------------
MODEL="${MODEL:-GLM-5.2}"
NODES="${NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"          # ray --num-gpus per pod; == TP=EP intra-node
BACKEND="${BACKEND:-megatron-bridge}"        # UNFUSED DSA (auto --qkv-format bshd; LoRA-A natively differentiable, no slime all-gather grad-block). "slime" = fused (thd).
LORA_RANK="${LORA_RANK:-16}"

# ---- task / rollout ---------------------------------------------------------
TASK="${TASK:-gsm8k}"
RESP_LEN="${RESP_LEN:-256}"                  # --rollout-max-response-len
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"            # number of train steps
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"          # save a LoRA adapter EVERY step
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-32}"   # full bf16 -> multi-node sglang engine
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"

# ---- eval (direct miles-core flags; gsm8k -> held-out test split) -----------
# ⚠ EVAL on the FULL GLM-5.2 dp-attention colocate config HANGS (sglang scheduler
#   freezes, util 0): a known unsolved issue — the windowing fix (c3782692a) was
#   reverted (d4f043185) because eval still stalls; documented workaround = EVAL=off.
#   Default off here for the full model; set EVAL=on only on configs without dp-attention.
EVAL="${EVAL:-off}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
DATA_DIR="${DATA_DIR:-/personal/datasets}"
if [ "$EVAL" = "on" ]; then
  EVAL_ARGS="--eval-interval ${EVAL_INTERVAL} --eval-top-p 0.7 \
--eval-prompt-data gsm8k ${DATA_DIR}/gsm8k/test.parquet \
--eval-input-key messages --eval-label-key label \
--n-samples-per-eval-prompt 1 --eval-max-response-len ${RESP_LEN}"
else
  EVAL_ARGS=""
fi

# ---- parallelism (full 744B: TP8 x DP8 / PP1 / CP1 / EP32) ------------------
PARALLEL_EXTRA="${PARALLEL_EXTRA:---pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32}"
# Recompute. ⚠ The unfused "megatron-bridge" backend runs bshd, and GLM-5.2 cross-layer DSA REJECTS
# bshd + activation recompute at forward time (the per-microbatch top-k holder is not recompute-safe).
# So default recompute OFF for megatron-bridge, ON for slime (thd, recompute-safe). Override via RECOMPUTE.
# (gsm8k seq is short, so the unfused backend fits comfortably without recompute.)
if [ "$BACKEND" = "slime" ]; then RECOMPUTE="${RECOMPUTE:-on}"; else RECOMPUTE="${RECOMPUTE:-off}"; fi
RECOMPUTE_ARGS=""
[ "$RECOMPUTE" = "on" ] && RECOMPUTE_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"

# ---- LoRA / memory knobs (read by run_glm5_lora.py via env) -----------------
export KEEP_MOE_LORA="${KEEP_MOE_LORA:-0}"           # 0 => ATTENTION-ONLY LoRA
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
LORA_BASE_CPU_BACKUP="${LORA_BASE_CPU_BACKUP:-on}"   # required for train<->rollout base consistency

# ---- infra ------------------------------------------------------------------
HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/${MODEL}}"
HEAD_IP="${HEAD_IP:?set HEAD_IP to the rank-0 bond0 IP}"
NCCL_IFNAME="${NCCL_IFNAME:-bond0}"
RAY_PORT="${RAY_PORT:-6379}"
DASH_PORT="${DASH_PORT:-8265}"
JOB_ID="${JOB_ID:-glm5_full_lora_rl_$(date +%y%m%d-%H%M%S)}"

# 64-GPU colocate job spawns many workers -> raise open-file limit before ray.
ulimit -n "${ULIMIT_NOFILE:-$(ulimit -Hn)}" 2>/dev/null || true

case "$ROLE" in
  head)
    echo "[head] MODEL=$MODEL NODES=$NODES x $GPUS_PER_NODE (world=$((GPUS_PER_NODE*NODES)))  BACKEND=$BACKEND  LoRA=attn-only(KEEP_MOE_LORA=$KEEP_MOE_LORA)"
    ray stop --force >/dev/null 2>&1 || true; sleep 3
    ray start --head --node-ip-address "$HEAD_IP" --port "$RAY_PORT" \
      --dashboard-host 0.0.0.0 --dashboard-port "$DASH_PORT" \
      --num-gpus "$GPUS_PER_NODE" --disable-usage-stats
    export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
    echo "[head] waiting for $NODES nodes..."
    while :; do n="$(ray status 2>/dev/null | grep -cE '^ 1 node_' || true)"; echo "[head]  joined $n/$NODES"; [ "$n" -ge "$NODES" ] && break; sleep 5; done
    echo "[head] cluster ready. now run: bash launch_glm_rl.sh launch  (on rank 0)"
    ;;

  worker)
    echo "[worker] joining $HEAD_IP:$RAY_PORT with $GPUS_PER_NODE GPUs"
    ray stop --force >/dev/null 2>&1 || true; sleep 3
    ray start --address="$HEAD_IP:$RAY_PORT" --num-gpus "$GPUS_PER_NODE"
    ;;

  launch)
    [ -z "${WANDB_API_KEY:-}" ] && { echo "[launch] FATAL: export WANDB_API_KEY (online wandb)"; exit 3; }
    EXTRA_ARGS="--actor-num-nodes ${NODES} --num-gpus-per-node ${GPUS_PER_NODE} \
--save-interval ${SAVE_INTERVAL} ${PARALLEL_EXTRA} ${RECOMPUTE_ARGS} \
--moe-token-dispatcher-type alltoall --sglang-lora-backend triton"
    [ "$LORA_BASE_CPU_BACKUP" = "on" ] && EXTRA_ARGS="${EXTRA_ARGS} --lora-base-cpu-backup"
    EXTRA_ARGS="${EXTRA_ARGS} ${EVAL_ARGS}"

    export HF_HOME=/cluster-storage/models PYTHONUNBUFFERED=1
    export NCCL_SOCKET_IFNAME="$NCCL_IFNAME" GLOO_SOCKET_IFNAME="$NCCL_IFNAME"
    export MILES_SCRIPT_EXTERNAL_RAY=1
    export RAY_ADDRESS="http://${HEAD_IP}:${DASH_PORT}"
    export MILES_RAY_SUBMIT_NO_WAIT=1 MILES_RAY_SUBMISSION_ID="$JOB_ID"
    export WANDB_API_KEY                                   # online wandb (run_glm5_lora.py enable_wandb defaults True)

    echo "[launch] JOB_ID=$JOB_ID  task=$TASK resp=$RESP_LEN steps=$NUM_ROLLOUT save_interval=$SAVE_INTERVAL eval_interval=$EVAL_INTERVAL"
    echo "[launch] extra-args: $EXTRA_ARGS"
    python scripts/run_glm5_lora.py train \
      --model-name "$MODEL" \
      --hf-checkpoint "$HF_CHECKPOINT" \
      --dsa-attention-backend "$BACKEND" \
      --lora-rank "$LORA_RANK" \
      --num-gpus-per-node "$GPUS_PER_NODE" \
      --num-rollout "$NUM_ROLLOUT" \
      --data-dir "$DATA_DIR" \
      --task "$TASK" --rollout-max-response-len "$RESP_LEN" \
      --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
      --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
      --extra-args "$EXTRA_ARGS"
    echo "[launch] submitted '$JOB_ID'.  monitor: RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job logs $JOB_ID --follow"
    ;;

  *) echo "usage: $0 {head|worker|launch}" >&2; exit 2 ;;
esac
