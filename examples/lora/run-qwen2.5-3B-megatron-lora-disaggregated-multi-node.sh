#!/bin/bash

# Distributed (multi-node) LoRA disaggregated training for Qwen2.5-3B.
#
# Disaggregated layout: the training GPUs and the rollout (SGLang) GPUs live on
# *different* nodes, all joined into one Ray cluster. Weights are transferred
# from the Megatron actor to the remote SGLang engine either by NCCL broadcast
# or by RDMA point-to-point (p2p).
#
# Cluster size is derived from the GPU counts in the "Cluster topology" section
# below — to scale, edit those numbers (or pass them as env vars); you do not
# need to touch the rest of the script. The default is 2 nodes x 1 GPU
# (1 training GPU + 1 rollout GPU).
#
# Usage:
#   bash run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh <MODE> <NODE_RANK>
#
#   MODE          : broadcast | p2p
#   NODE_RANK     : 0 (head node) | 1..N (worker nodes)
#
# Run the script once per node, with NODE_RANK 0 on the head node and a distinct
# non-zero rank on every other node. Only the head submits the training job; all
# other nodes simply join the Ray cluster and contribute their GPUs.
#
# The head node IP is hardcoded below (HEAD_NODE_IP) — edit it for your cluster,
# or override it on the command line: HEAD_NODE_IP=10.0.0.2 bash ... broadcast 0
#
# Examples (default 2 nodes x 1 GPUs layout):
#   # broadcast
#   bash run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh broadcast 0   # head
#   bash run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh broadcast 1   # worker
#   # p2p (RDMA)
#   bash run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh p2p 0         # head
#   bash run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh p2p 1         # worker
#
# Example (4 nodes x 8 GPUs = 32 GPUs: 16 train GPUs + 16 rollout GPUs):
#   ENV="GPUS_PER_NODE=8 NUM_TRAIN_GPUS=16 NUM_ROLLOUT_GPUS=16"
#   env ${ENV} bash ...-multi-node.sh broadcast 0   # head    (train node 0)
#   env ${ENV} bash ...-multi-node.sh broadcast 1   # worker 1 (train node 1)
#   env ${ENV} bash ...-multi-node.sh broadcast 2   # worker 2 (rollout node 0)
#   env ${ENV} bash ...-multi-node.sh broadcast 3   # worker 3 (rollout node 1)
#   # 2 train nodes -> actor-num-nodes 2; 16 rollout GPUs split into engines via
#   # ROLLOUT_GPUS_PER_ENGINE (default 1 -> 16 engines; set to 8 for 2 TP=8 engines).
#   # With >1 training GPU also raise the parallelism in PERF_ARGS accordingly.
#
# Example (asymmetric, 2 nodes: 1 train GPU on node A, a 2-GPU rollout engine on
# node B). Train and rollout nodes have different GPU counts:
#   ENV="TRAIN_GPUS_PER_NODE=1 NUM_TRAIN_GPUS=1 \
#        ROLLOUT_GPUS_PER_NODE=2 NUM_ROLLOUT_GPUS=2 ROLLOUT_GPUS_PER_ENGINE=2"
#   env ${ENV} bash ...-multi-node.sh broadcast 0   # head    (train node,   1 GPU)
#   env ${ENV} bash ...-multi-node.sh broadcast 1   # worker 1 (rollout node, 2 GPUs)
#   # Ranks [0, NUM_TRAIN_NODES) are training nodes; the rest are rollout nodes,
#   # so rank 0 exposes 1 GPU and rank 1 exposes 2 GPUs to Ray automatically.

export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONBUFFERED=1

# ---------------------------------------------------------------------------
# Cluster topology — edit these (or pass as env vars) to size the cluster
# ---------------------------------------------------------------------------
# IP address of the head node, reachable from every node. Hardcoded here but
# can be overridden from the environment (HEAD_NODE_IP=... bash ...).
HEAD_NODE_IP="${HEAD_NODE_IP:-10.0.0.1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-$GPUS_PER_NODE}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-$GPUS_PER_NODE}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_GPUS:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"

# Network interface NCCL/Gloo use for cross-node sockets.
apt-get install -y iproute2.
SOCKET_IFNAME="${SOCKET_IFNAME:-$(ip -o -4 route get "${HEAD_NODE_IP}" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
echo "Using SOCKET_IFNAME=${SOCKET_IFNAME} for NCCL/Gloo cross-node sockets"

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
MODE="$1"              # broadcast | p2p
NODE_RANK="$2"         # 0 = head, 1..N = workers

if [ "$MODE" != "broadcast" ] && [ "$MODE" != "p2p" ]; then
    echo "MODE must be 'broadcast' or 'p2p' (got '$MODE')"
    exit 1
fi

# ---------------------------------------------------------------------------
# Derived cluster sizing (no need to edit — computed from the values above)
# ---------------------------------------------------------------------------
# Training nodes occupy ranks [0, NUM_TRAIN_NODES); rollout nodes the rest.
if [ $((NUM_TRAIN_GPUS % TRAIN_GPUS_PER_NODE)) -ne 0 ]; then
    echo "NUM_TRAIN_GPUS (${NUM_TRAIN_GPUS}) must be divisible by TRAIN_GPUS_PER_NODE (${TRAIN_GPUS_PER_NODE})"
    exit 1
fi
if [ $((NUM_ROLLOUT_GPUS % ROLLOUT_GPUS_PER_NODE)) -ne 0 ]; then
    echo "NUM_ROLLOUT_GPUS (${NUM_ROLLOUT_GPUS}) must be divisible by ROLLOUT_GPUS_PER_NODE (${ROLLOUT_GPUS_PER_NODE})"
    exit 1
fi
NUM_TRAIN_NODES=$((NUM_TRAIN_GPUS / TRAIN_GPUS_PER_NODE))
NUM_ROLLOUT_NODES=$((NUM_ROLLOUT_GPUS / ROLLOUT_GPUS_PER_NODE))
NNODES=$((NUM_TRAIN_NODES + NUM_ROLLOUT_NODES))
EXPECTED_GPUS=$((NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS))

# This node is a training node if its rank is in [0, NUM_TRAIN_NODES), else a
# rollout node — that decides how many GPUs it exposes to Ray.
if [ "${NODE_RANK}" -lt "${NUM_TRAIN_NODES}" ]; then
    THIS_NODE_GPUS=${TRAIN_GPUS_PER_NODE}
else
    THIS_NODE_GPUS=${ROLLOUT_GPUS_PER_NODE}
fi

# Expose this node's GPUs to Ray (default: all of them, 0..THIS_NODE_GPUS-1).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((THIS_NODE_GPUS - 1)))}"

# Optional: set to 1 to skip the post-update weight-equality check.
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"

# ---------------------------------------------------------------------------
# Cleanup stale processes (every node cleans up its own)
# ---------------------------------------------------------------------------
pkill sglang || true
ray stop --force || true
sleep 5 # Wait for processes to terminate gracefully
# Force kill any remaining processes.
# Note: `pkill -9 python` is broad and can be risky.
pkill -9 sglang || true
pkill -9 ray || true
pkill -9 python || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-3B-Instruct/
   --megatron-to-hf-mode bridge
)

LORA_ARGS=(
   --lora-rank 32                    # LoRA rank (typical values: 8, 16, 32, 64)
   --lora-alpha 32                   # LoRA alpha (usually 2x rank)
   --lora-dropout 0.0                # LoRA dropout (0.0 for RL training)
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss # if use kl loss, should use --ref-load
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # --lr 1e-6
   --lr 1e-5                         # Higher LR often works better for LoRA
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project miles-lora-megatron
   --wandb-group qwen2.5-3B-lora-disaggregate-${NNODES}node-${MODE}
)

# SGLang serves the rollout on the dedicated rollout GPUs.
SGLANG_ARGS=(
   --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.7

   # --sglang-enable-deterministic-inference
   # --sglang-attention-backend flashinfer
   # --deterministic-mode
)
# p2p (RDMA) transfer needs the SGLang remote weight loader to seed via the
# transfer engine. Broadcast (NCCL) does not.
if [ "$MODE" = "p2p" ]; then
   SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
fi

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   # disaggregated layout: training spans NUM_TRAIN_NODES nodes, rollout the rest
   --actor-num-nodes ${NUM_TRAIN_NODES}
   --actor-num-gpus-per-node ${TRAIN_GPUS_PER_NODE}
   --update-weight-transfer-mode ${MODE}
   --update-weight-buffer-size $((1 * 1024 * 1024 * 1024))
)
# Verify the rollout engine received identical weights after each update.
if [ "$SKIP_VALIDATION" -eq 0 ]; then
   MISC_ARGS+=(--check-weight-update-equal)
fi

# ---------------------------------------------------------------------------
# Launch Ray
#   - head node (rank 0) starts the cluster head and submits the job
#   - worker nodes only join the cluster and contribute their GPUs
# Each node exposes THIS_NODE_GPUS, which may differ between train and rollout.
# ---------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
   ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus ${THIS_NODE_GPUS} \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
   # Give the head node a moment to come up before joining.
   sleep 20
   ray start --address="${HEAD_NODE_IP}:6379" --num-gpus ${THIS_NODE_GPUS} --disable-usage-stats
fi

# ---------------------------------------------------------------------------
# Head node: wait for all GPUs to join, then submit the training job.
# Worker nodes: nothing else to do — Ray keeps them attached to the cluster.
# ---------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
   echo "Waiting for ${EXPECTED_GPUS} GPUs in Ray cluster..."
   set +x
   while true; do
      AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null || echo 0)
      echo "  ... detected ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs"
      if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
         break
      fi
      sleep 5
   done
   set -x
   echo "All ${EXPECTED_GPUS} GPUs available. Submitting job."

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{
        "env_vars": {
           "PYTHONPATH": "/root/Megatron-LM",
           "CUDA_DEVICE_MAX_CONNECTIONS": "1",
           "NCCL_ALGO": "Ring",
           "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "NCCL_SOCKET_IFNAME": "'"${SOCKET_IFNAME}"'",
           "GLOO_SOCKET_IFNAME": "'"${SOCKET_IFNAME}"'"
        }
      }' \
      -- python3 train.py \
      --calculate-per-token-loss \
      --use-miles-router \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${LORA_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${GRPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]} \
      ${ROLLOUT_ARGS[@]}
fi
