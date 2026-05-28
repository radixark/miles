#!/bin/bash

# for rerun the task
# pkill -9 sglang
# sleep 3
# # K8s: Ray is started by ray_init_simple.sh — keep the following lines commented.
# # ray stop --force
# # pkill -9 ray
# pkill -9 miles
# sleep 3
# pkill -9 miles
# pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${MILES_ROOT}/scripts/models/qwen3-0.6B.sh"

# Disaggregated: 1 train node + (NUM_NODES - 1) rollout nodes (default 2 nodes total).
NUM_NODES=${SLURM_JOB_NUM_NODES:-2}
TRAIN_NUM_NODES=${TRAIN_NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_NODES=$((NUM_NODES - TRAIN_NUM_NODES))
ROLLOUT_GPUS=$((ROLLOUT_NUM_NODES * NUM_GPUS_PER_NODE))
echo "Disagg: train ${TRAIN_NUM_NODES} node(s), rollout ${ROLLOUT_NUM_NODES} node(s) (${ROLLOUT_GPUS} rollout GPUs)"

MODEL_NAME=Qwen3-0.6B
DATA_NAME=swegym
VARIANT=${DATA_NAME}_${MODEL_NAME}_agentic_async
log_dir=/home/yangchengyi/data/logs
mkdir -p "${log_dir}"
log_path=${log_dir}/${MODEL_NAME}_${DATA_NAME}_agentic_async_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log
exec > >(tee -a "${log_path}") 2>&1
CKPT_ARGS=(
   --hf-checkpoint /home/yangchengyi/data/models/${MODEL_NAME}
   --ref-load /home/yangchengyi/data/models/${MODEL_NAME}_torch_dist
   --save /home/yangchengyi/data/ckpts/${MODEL_NAME}_agentic_async/
   --save-interval 5
)

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data /home/yangchengyi/data/datasets/${DATA_NAME}.jsonl
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-temperature 0.8
   --rollout-max-response-len 8192
   --max-seq-len 16384
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
   --global-batch-size 32
   --balance-data
   --pause-generation-mode in_place
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-agentic
   --wandb-group ${VARIANT}
   --wandb-key ${WANDB_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.80
   --sglang-tool-call-parser qwen25
   --sglang-reasoning-parser qwen3
   --use-miles-router
   --sglang-router-port 31000
)

AGENT_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path swe_agent_function.run
   --custom-rm-path generate.reward_func
   --tito-model qwen3
   --use-session-server
   --session-server-port 30000
   --tito-allowed-append-roles user tool
)

PROMETHEUS_ARGS=(
   --use-prometheus
   --prometheus-port 9090
   --prometheus-run-name ${VARIANT}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-transfer-mode broadcast
   --update-weight-buffer-size $((2 * 1024 * 1024 * 1024))
   --actor-num-nodes ${TRAIN_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_GPUS}
   --grad-reduce-in-bf16
   --use-fault-tolerance
   --rollout-health-check-first-wait 1800
   --dump-details /home/yangchengyi/data/trajs/${VARIANT}
   --debug-rollout-only
)

# Agent / Harbor (passed into Ray job)
export AGENT_SERVER_URL=${AGENT_SERVER_URL:-http://houjue-harbor-server-ycy.houjue.svc.cluster.local:11000}
export AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-${MODEL_NAME}}
export HARBOR_TASKS_DIR=${HARBOR_TASKS_DIR:-/home/yangchengyi/data/harbor/datasets/swegym}
export MILES_ROUTER_EXTERNAL_HOST=${MILES_ROUTER_EXTERNAL_HOST:-miles-session.nlp-train.svc.cluster.local}
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=false
export MILES_SCRIPT_EXTERNAL_RAY=${MILES_SCRIPT_EXTERNAL_RAY:-true}

export PYTHONPATH=${MILES_ROOT}:${MEGATRON_PATH:-/root/Megatron-LM}:${SCRIPT_DIR}:${MILES_ROOT}/examples/fully_async:${PYTHONPATH:-}

# launch ray head (local debug). K8s: use ray_init_simple.sh and only set MASTER_ADDR / RAY_ADDRESS.
export MASTER_ADDR=${MASTER_ADDR:-${POD_IP:-127.0.0.1}}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ADDRESS=${RAY_ADDRESS:-http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}}

if [ "${MILES_SCRIPT_EXTERNAL_RAY}" != "true" ]; then
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK\": \"false\",
    \"AGENT_SERVER_URL\": \"${AGENT_SERVER_URL}\",
    \"AGENT_MODEL_NAME\": \"${AGENT_MODEL_NAME}\",
    \"HARBOR_TASKS_DIR\": \"${HARBOR_TASKS_DIR}\",
    \"MILES_ROUTER_EXTERNAL_HOST\": \"${MILES_ROUTER_EXTERNAL_HOST}\"
  }
}"

RAY_JOB_ADDRESS=${RAY_ADDRESS}
if [[ "${RAY_JOB_ADDRESS}" != http* ]]; then
   RAY_JOB_ADDRESS="http://127.0.0.1:8265"
fi

ray job submit --address="${RAY_JOB_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_ROOT}/train_async.py \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${AGENT_ARGS[@]} \
   ${PROMETHEUS_ARGS[@]} \
   ${MISC_ARGS[@]}
