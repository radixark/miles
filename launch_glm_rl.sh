#!/usr/bin/env bash
# ============================================================================
#  launch_glm_rl.sh — one-command multi-node GLM-5.2 (744B) attention-only LoRA RL,
#  wired to the miles_lora rx-devbox flow (devbox_config.sh / rx devbox run --rank N).
#
#  One-line launch of the whole multi-node training from your local box (the one with rx):
#     RX_GPU_COUNT=64 RX_DEVBOX_NAME=miles-exp WANDB_API_KEY=xxx bash launch_glm_rl.sh
#
#  DSA backend default = unfused (megatron-bridge). Switch to fused: BACKEND=slime bash launch_glm_rl.sh
#
#  Other roles (the orchestrator calls these on each pod automatically; usually no manual use):
#     bash launch_glm_rl.sh head|worker|launch        # run on a pod
#     bash launch_glm_rl.sh stop                       # abort the job (keep the nodes)
#
#  PREREQ: nodes acquired via 0_launch_node.sh, this repo synced to /personal/miles via
#          1_sync_files.sh, and every rank installed via 2_pre-install.sh (or PREINSTALL=1 here).
#          This file must live inside the synced miles repo so it appears in /personal/miles/.
# ============================================================================
set -euo pipefail
ROLE="${1:-all}"
ulimit -n "${ULIMIT_NOFILE:-$(ulimit -Hn)}" 2>/dev/null || true

# Reuse the miles_lora devbox config (sourced on the local box; skipped on pods where it is absent).
DEVBOX_CONFIG="${DEVBOX_CONFIG:-}"
if [ -z "$DEVBOX_CONFIG" ]; then
  for c in "$HOME/Downloads/miles_lora/devbox_config.sh" \
           "${MILES_DEV_ROOT:-}/devbox_config.sh"; do
    [ -n "$c" ] && [ -f "$c" ] && { DEVBOX_CONFIG="$c"; break; }
  done
fi
[ -n "$DEVBOX_CONFIG" ] && [ -f "$DEVBOX_CONFIG" ] && source "$DEVBOX_CONFIG"

# Topology (derived from devbox_config; overridable via env vars).
RX_DEVBOX_NAME="${RX_DEVBOX_NAME:-miles-exp}"
REMOTE_ROOT="${REMOTE_ROOT:-/personal}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"                       # 8 GPU per H200 node
NODES="${NODES:-$(( ${RX_GPU_COUNT:-8} / GPUS_PER_NODE ))}"; [ "$NODES" -ge 1 ] || NODES=1
REPO_DIR="${REPO_DIR:-$REMOTE_ROOT/miles}"
SCRIPT_NAME="${SCRIPT_NAME:-launch_glm_rl.sh}"
HEAD_IP="${HEAD_IP:-}"                                    # empty => auto-detect
NCCL_IFNAME="${NCCL_IFNAME:-bond0}"
RAY_PORT="${RAY_PORT:-6379}"; DASH_PORT="${DASH_PORT:-8265}"

# DSA attention backend (DEFAULT = unfused / megatron-bridge):
#   unfused (megatron-bridge): megatron-core DSA kernels, no tilelang dependency.
#   slime  (fused TileLang SparseMLA + lighting_indexer), requires tilelang.
BACKEND="${BACKEND:-megatron-bridge}"
# qkv-format + activation recompute must follow the backend (full GLM-5.2 has DSA cross-layer
# index sharing => unfused bshd + recompute raises a forward AssertionError; only slime/thd is
# recompute-safe):
if [ "$BACKEND" = "slime" ]; then
  QKV_FORMAT="${QKV_FORMAT:-thd}";  RECOMPUTE="${RECOMPUTE:-on}"
else
  QKV_FORMAT="${QKV_FORMAT:-bshd}"; RECOMPUTE="${RECOMPUTE:-off}"   # WARN: recompute off on the full model may OOM
fi
RECOMPUTE_ARGS=()
[ "$RECOMPUTE" = "on" ] && RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)

# Paths / ids / wandb.
HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/GLM-5.2}"
SAVE_DIR="${SAVE_DIR:-/personal/checkpoints}"
DATA_DIR="${DATA_DIR:-/personal/datasets}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
TRAIN_PY="${TRAIN_PY:-$REPO_DIR/train.py}"
RUN_ID="${RUN_ID:-$(date -u +%y%m%d-%H%M%S)-$(printf '%03d' $((RANDOM % 1000)))}"
JOB_ID="${JOB_ID:-glm5_full_lora_rl_$(date +%y%m%d-%H%M%S)}"
WANDB="${WANDB:-online}"; WANDB_PROJECT="${WANDB_PROJECT:-miles-run_glm5_lora}"

# ════════════════════════════════════════════════════════════════════════════
#  UPPER HALF — train.py args (only backend/qkv-format/recompute follow the switches above; rest fixed)
# ════════════════════════════════════════════════════════════════════════════
MODEL_ARGS=(
  --spec miles_plugins.models.glm5.glm5 get_glm5_spec
  --moe-layer-freq "[0]*3+[1]*75" --num-experts 256 --moe-router-topk 8
  --moe-shared-expert-intermediate-size 2048 --moe-ffn-hidden-size 2048
  --moe-grouped-gemm --moe-permute-fusion
  --moe-router-score-function sigmoid --moe-router-pre-softmax --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0 --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk-scaling-factor 2.5 --moe-aux-loss-coeff 0 --moe-router-dtype fp32
  --make-vocab-size-divisible-by 16
  --num-layers 78 --hidden-size 6144 --ffn-hidden-size 12288 --num-attention-heads 64
  --disable-bias-linear --swiglu --untie-embeddings-and-output-weights
  --position-embedding-type rope --no-position-embedding --normalization RMSNorm --qk-layernorm
  --multi-latent-attention --q-lora-rank 2048 --kv-lora-rank 512 --qk-head-dim 192 --v-head-dim 256
  --kv-channels 192 --qk-pos-emb-head-dim 64 --vocab-size 154880 --rotary-base 8000000 --enable-experimental
)

# MoE-expert LoRA layer scope (OOM control). The bare names gate_proj/up_proj/down_proj put LoRA on
# ALL 75 MoE layers, whose grouped-GEMM backward activations (recompute is OFF on bshd) OOM the
# step-0 backward at full 744B (rank GPU hits ~100%). Restrict to the LAST $MOE_LORA_LAST_N MoE layers
# via Megatron-Bridge ModuleMatcher wildcards (*.layers.<N>.*.linear_fc1/linear_fc2). Attention LoRA
# stays on all layers. Set MOE_LORA_LAST_N=0 to LoRA every MoE layer (bare names, the OOM-prone path).
MOE_LORA_LAST_N="${MOE_LORA_LAST_N:-10}"
ATTN_TM="q_proj,k_proj,v_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
if [ "${MOE_LORA_LAST_N:-0}" -gt 0 ] 2>/dev/null && [ "$MOE_LORA_LAST_N" -lt 78 ]; then
  MOE_TM=""; for n in $(seq $((78 - MOE_LORA_LAST_N)) 77); do MOE_TM="$MOE_TM,*.layers.$n.*.linear_fc1,*.layers.$n.*.linear_fc2"; done
  TARGET_MODULES="${ATTN_TM}${MOE_TM}"
else
  TARGET_MODULES="${ATTN_TM},gate_proj,up_proj,down_proj"
fi

TRAIN_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT" --megatron-to-hf-mode bridge --dsa-attention-backend "$BACKEND"
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.0
  # LoRA targets = attention (all layers) + MoE experts (last MOE_LORA_LAST_N layers; TARGET_MODULES
  # built above). MUST stay quoted -- the *.layers.N.* wildcards would otherwise glob-expand in bash.
  # MoE-expert LoRA needs BOTH flags below ON together or the sglang rollout crashes ("scheduler died":
  # expert gate_up LoRA-B dim vs base mismatch under EP=32 / dp-attention):
  #   --experts-shared-outer-loras       (TRAIN side; arguments.py auto-mirrors --sglang-experts-shared-outer-loras)
  #   --sglang-lora-use-virtual-experts  (SERVE side; in the sglang block below) -- NOT auto-set, added explicitly.
  --target-modules "$TARGET_MODULES"
  --experts-shared-outer-loras
  --no-gradient-accumulation-fusion
  --rm-type math --prompt-data "$DATA_DIR/gsm8k/train.parquet" --input-key messages --label-key label
  --apply-chat-template --rollout-shuffle
  --num-rollout 10000 --rollout-batch-size 8 --n-samples-per-prompt 16 --rollout-max-response-len 256
  --rollout-temperature 1.0 --global-batch-size 64
  --advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00
  --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 --use-rollout-routing-replay
  --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  # Parallelism (TP8 / EP32 / PP1 / CP1, world=64, DP8); qkv-format follows the backend (unfused=bshd).
  --tensor-model-parallel-size 8 --expert-model-parallel-size 32 --expert-tensor-parallel-size 1
  --pipeline-model-parallel-size 1 --context-parallel-size 1 --sequence-parallel
  --qkv-format "$QKV_FORMAT" --micro-batch-size 1
  # rollout / sglang (always sglang, independent of the train backend).
  --rollout-num-gpus-per-engine 32 --sglang-mem-fraction-static 0.7
  --sglang-enable-dp-attention --sglang-ep-size 32 --sglang-dp-size 32 --sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head
  --sglang-attention-backend nsa --sglang-nsa-decode-backend flashmla_sparse --sglang-nsa-prefill-backend flashmla_sparse
  --sglang-page-size 64 --sglang-cuda-graph-max-bs 64 --sglang-max-running-requests 512
  --sglang-chunked-prefill-size 65536 --sglang-watchdog-timeout 3600
  --sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion
  --sglang-max-lora-rank 16 --sglang-lora-backend triton --sglang-lora-use-virtual-experts
  --lora-base-cpu-backup
  # colocate / multi-node.
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --calculate-per-token-loss --use-miles-router --colocate
  --actor-num-nodes "$NODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" --num-gpus-per-node "$GPUS_PER_NODE"
  --moe-token-dispatcher-type alltoall --save-interval 20
  --save "$SAVE_DIR/$RUN_ID"
)

# ════════════════════════════════════════════════════════════════════════════
#  LOWER HALF — orchestration (aligned with the miles_lora rx-devbox flow)
# ════════════════════════════════════════════════════════════════════════════
_detect_head_ip() {
  local ip node
  # 1) a real dotted IP on the rank-0 / Nodes line (clusters that surface IPs directly)
  ip="$(rx devbox status "$RX_DEVBOX_NAME" 2>/dev/null | grep -iE '\[rank 0\]|^[[:space:]]*Nodes:' | head -n1 \
        | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n1 || true)"
  [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && { echo "$ip"; return; }
  # 2) dash-form rank-0 node name (e.g. gpu1-10-220-51-74 / h200-10-220-51-1) -> dotted IP (this cluster)
  node="$(rx devbox status "$RX_DEVBOX_NAME" 2>/dev/null | grep -iE '\[rank 0\]' \
          | grep -oE '[A-Za-z0-9]+(-[0-9]{1,3}){4}' | head -n1 || true)"
  if [ -n "$node" ]; then
    ip="$(printf '%s' "$node" | awk -F- '{print $(NF-3)"."$(NF-2)"."$(NF-1)"."$NF}')"
    [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && { echo "$ip"; return; }
  fi
  # 3) last resort: ask the pod (only works if `ip` is installed in the container)
  rx devbox run "$RX_DEVBOX_NAME" --rank 0 -- bash -lc "ip -o -4 addr show $NCCL_IFNAME" 2>/dev/null \
    | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n1 || true
}
_pod() {  # _pod <rank> "<env prefix>" <role>
  rx devbox run "$RX_DEVBOX_NAME" --rank "$1" -- bash -lc \
    "cd '$REPO_DIR' && HEAD_IP='$HEAD_IP' NODES='$NODES' GPUS_PER_NODE='$GPUS_PER_NODE' NCCL_IFNAME='$NCCL_IFNAME' RAY_PORT='$RAY_PORT' DASH_PORT='$DASH_PORT' $2 bash '$SCRIPT_NAME' $3"
}

case "$ROLE" in
  all)  # orchestrator (run on the local / rx-enabled control box)
    command -v rx >/dev/null || { echo "[all] FATAL: rx CLI not found (run on the control box)"; exit 2; }
    [ "$WANDB" = "online" ] && [ -z "${WANDB_API_KEY:-}" ] && { echo "[all] FATAL: WANDB=online requires WANDB_API_KEY (or use WANDB=offline)"; exit 3; }
    [ -z "$HEAD_IP" ] && { echo "[all] detecting rank-0 IP..."; HEAD_IP="$(_detect_head_ip)"; }
    [ -n "$HEAD_IP" ] || { echo "[all] FATAL: could not determine HEAD_IP (set HEAD_IP=... manually)"; exit 4; }
    rx devbox run "$RX_DEVBOX_NAME" --rank 0 -- bash -lc "test -f '$REPO_DIR/$SCRIPT_NAME'" \
      || { echo "[all] FATAL: $REPO_DIR/$SCRIPT_NAME not on rank-0; run ./1_sync_files.sh to sync it to /personal/miles first"; exit 5; }
    echo "[all] BOX=$RX_DEVBOX_NAME NODES=$NODES x $GPUS_PER_NODE HEAD_IP=$HEAD_IP BACKEND=$BACKEND qkv=$QKV_FORMAT recompute=$RECOMPUTE RUN_ID=$RUN_ID JOB_ID=$JOB_ID"

    # Pre-run cleanup: kill stale ray/sglang on EVERY rank (avoids leftover raylets that show up as
    # extra/double-counted nodes, plus GPU still held by a dead run), then clear old checkpoints so
    # /personal does not fill up mid-run. Set CLEAN_CKPT=0 to keep existing checkpoints.
    echo "[all] pre-run cleanup: kill ray/sglang on all $NODES ranks ..."
    for r in $(seq 0 $((NODES - 1))); do
      rx devbox run "$RX_DEVBOX_NAME" --rank "$r" -- bash -lc \
        'ray stop --force >/dev/null 2>&1 || true; for p in sglang "ray::" raylet train.py redis; do pkill -9 -f "$p" 2>/dev/null || true; done; true' || true
    done
    if [ "${CLEAN_CKPT:-1}" = "1" ]; then
      echo "[all] clearing checkpoints under $SAVE_DIR (set CLEAN_CKPT=0 to keep) ..."
      rx devbox run "$RX_DEVBOX_NAME" --rank 0 -- bash -lc "rm -rf '$SAVE_DIR'/* 2>/dev/null || true; mkdir -p '$SAVE_DIR'; df -h /personal | tail -1" || true
    fi

    if [ "${PREINSTALL:-0}" = "1" ]; then
      echo "[all] per-rank editable install ..."
      for r in $(seq 0 $((NODES - 1))); do
        rx devbox run "$RX_DEVBOX_NAME" --rank "$r" -- bash -lc \
          "export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0; \
           [ -d $REMOTE_ROOT/sglang/python ] && (cd $REMOTE_ROOT/sglang && pip install -e python --no-deps --no-build-isolation); \
           [ -e $REMOTE_ROOT/Megatron-Bridge/pyproject.toml ] && pip install -e $REMOTE_ROOT/Megatron-Bridge --no-deps --no-build-isolation; \
           [ -e $REMOTE_ROOT/miles/pyproject.toml ] && pip install -e $REMOTE_ROOT/miles --no-deps; true"
      done
    fi

    echo "[all] starting ray head (rank 0)"; _pod 0 "" head
    for r in $(seq 1 $((NODES - 1))); do echo "[all] joining worker rank $r"; _pod "$r" "" worker; done

    echo "[all] waiting for $NODES nodes to register ..."
    while :; do
      n="$(rx devbox run "$RX_DEVBOX_NAME" --rank 0 -- bash -lc "RAY_ADDRESS='$HEAD_IP:$RAY_PORT' ray status 2>/dev/null | grep -cE '^ 1 node_'" 2>/dev/null || echo 0)"
      n="$(printf '%s' "$n" | tr -dc '0-9')"; n="${n:-0}"
      echo "[all]   joined ${n}/${NODES}"; [ "$n" -ge "$NODES" ] && break; sleep 5
    done

    echo "[all] submitting training job (rank 0)"
    _pod 0 "JOB_ID='$JOB_ID' RUN_ID='$RUN_ID' BACKEND='$BACKEND' QKV_FORMAT='$QKV_FORMAT' RECOMPUTE='$RECOMPUTE' WANDB='$WANDB' WANDB_PROJECT='$WANDB_PROJECT' WANDB_API_KEY='${WANDB_API_KEY:-}'" launch

    echo "[all] DONE. monitor:"
    echo "  rx devbox run $RX_DEVBOX_NAME --rank 0 -- bash -lc \"RAY_ADDRESS=http://$HEAD_IP:$DASH_PORT ray job logs $JOB_ID --follow\""
    echo "[all] abort: bash $SCRIPT_NAME stop   |  fully release nodes: ~/Downloads/miles_lora/4_release_kill.sh"
    ;;

  head)
    : "${HEAD_IP:?set HEAD_IP}"
    ray stop --force >/dev/null 2>&1 || true; sleep 3
    ray start --head --node-ip-address "$HEAD_IP" --port "$RAY_PORT" \
      --dashboard-host 0.0.0.0 --dashboard-port "$DASH_PORT" --num-gpus "$GPUS_PER_NODE" --disable-usage-stats
    ;;

  worker)
    : "${HEAD_IP:?set HEAD_IP}"
    ray stop --force >/dev/null 2>&1 || true; sleep 3
    ray start --address="$HEAD_IP:$RAY_PORT" --num-gpus "$GPUS_PER_NODE"
    ;;

  launch)
    : "${HEAD_IP:?set HEAD_IP}"
    WANDB_ARGS=()
    case "$WANDB" in
      online)  [ -n "${WANDB_API_KEY:-}" ] || { echo "[launch] FATAL: WANDB=online requires WANDB_API_KEY"; exit 3; }
               WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$RUN_ID" --wandb-key "$WANDB_API_KEY" --disable-wandb-random-suffix) ;;
      offline) WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$RUN_ID" --wandb-mode offline --disable-wandb-random-suffix) ;;
    esac
    pkill -9 sglang || true; sleep 3; pkill -9 miles || true; sleep 3; pkill -9 miles || true; pkill -9 redis || true; true
    export RAY_ADDRESS="http://$HEAD_IP:$DASH_PORT" PYTHONUNBUFFERED=1 HF_HOME=/cluster-storage/models
    RUNTIME_ENV_JSON="$(cat <<JSON
{"env_vars":{"PYTHONPATH":"$MEGATRON_PATH","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"1","NCCL_SOCKET_IFNAME":"$NCCL_IFNAME","GLOO_SOCKET_IFNAME":"$NCCL_IFNAME","no_proxy":"127.0.0.1,127.0.0.1","MASTER_ADDR":"127.0.0.1","MILES_EXPERIMENTAL_ROLLOUT_REFACTOR":"1","INDEXER_ROPE_NEOX_STYLE":"0","SGLANG_NSA_FORCE_MLA":"1","NVSHMEM_DISABLE_NCCL":"0"}}
JSON
)"
    echo "[launch] JOB_ID=$JOB_ID RUN_ID=$RUN_ID BACKEND=$BACKEND qkv=$QKV_FORMAT recompute=$RECOMPUTE -> $TRAIN_PY"
    ray job submit --no-wait --submission-id "$JOB_ID" --runtime-env-json="$RUNTIME_ENV_JSON" \
      -- python3 "$TRAIN_PY" "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "${RECOMPUTE_ARGS[@]}" "${WANDB_ARGS[@]}"
    ;;

  stop)  # abort the job (keep the nodes), across all ranks
    command -v rx >/dev/null || { echo "[stop] rx required"; exit 2; }
    for r in $(seq 0 $((NODES - 1))); do
      rx devbox run "$RX_DEVBOX_NAME" --rank "$r" -- bash -lc \
        'ray stop --force 2>/dev/null || true; pkill -9 -f sglang||true; pkill -9 -f "ray::"||true; pkill -9 -f train.py||true; true' || true
    done
    echo "[stop] aborted; nodes kept. To fully release nodes use ~/Downloads/miles_lora/4_release_kill.sh"
    ;;

  *) echo "usage: $0 {all|head|worker|launch|stop}" >&2; exit 2 ;;
esac
