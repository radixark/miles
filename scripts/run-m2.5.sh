#!/bin/bash

# Tear down previous local Ray/Sglang/Python before re-run. When Ray was started outside this
# script (e.g. code/ray_init_simple.sh on K8s), set MILES_USE_EXTERNAL_RAY=1 to skip this block
# and the embedded `ray start` / SSH worker bootstrap below.
export MILES_USE_EXTERNAL_RAY=1
export MASTER_ADDR="$POD_IP"  
if [[ "${MILES_USE_EXTERNAL_RAY:-0}" != "1" ]]; then
  # for rerun the task
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
fi

set -ex

# if base folder not set raise error
# if [ -z "${BASE_FOLDER}" ]; then
#   echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
#   exit 1
# fi

if [[ "${MILES_USE_EXTERNAL_RAY:-0}" == "1" ]] && [[ -z "${MASTER_ADDR:-}" ]] && [[ -n "${POD_IP:-}" ]]; then
  export MASTER_ADDR="${POD_IP}"
fi

if [ -z "${MASTER_ADDR:-}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

# ray job submit talks to Dashboard (8265). Default to head IP so it matches ray_init_simple.sh
# binding on --dashboard-host=0.0.0.0. Override with RAY_JOB_SUBMIT_ADDRESS if needed.
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_JOB_SUBMIT_ADDRESS="${RAY_JOB_SUBMIT_ADDRESS:-http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}}"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# Prefer WANDB_API_KEY; never hardcode keys in this repo.
WANDB_KEY_VALUE="${WANDB_API_KEY:-${WANDB_KEY:-}}"

# Single root for host paths (override in K8s / shared FS jobs).
MILES_DATA_ROOT="${MILES_DATA_ROOT:-/home/yangchengyi/data}"
DATA_ROOT="${MILES_DATA_ROOT}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
model_name=MiniMax-M2.5
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax-m2.5.sh"
model_dir="${DATA_ROOT}/models"
ckpt_dir="${DATA_ROOT}/ckpts"
interval=50
project_name=minmax-dev
variant=${model_name}_minmax_moe_test
CKPT_ARGS=(
   --hf-checkpoint ${model_dir}/${model_name}
   --ref-load ${model_dir}/${model_name}_torch_dist
   --load ${ckpt_dir}/${variant}
   --save ${ckpt_dir}/${variant}
   --save-interval ${interval}
)
data_dir="${DATA_ROOT}/datasets"
data_path=${data_dir}/deepmath-103k_miles.jsonl
log_dir="${DATA_ROOT}/logs"
log_path=${log_dir}/${model_name}_dapo_miles_$(date -d '+8 hours' +%Y%m%d_%H%M%S).log
WORKING_DIR="${DATA_ROOT}/miles"
# --rollout-function-path fully_async_rollout.* 需要能 import fully_async_rollout（见 examples/fully_async/）。
# 官方示例：PYTHONPATH 包含 examples/fully_async。Ray 打包时可用 py_modules，避免各节点绝对路径不一致。
FULLY_ASYNC_ROLLOUT_DIR="${FULLY_ASYNC_ROLLOUT_DIR:-${WORKING_DIR}/examples/fully_async}"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_path}") 2>&1

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${data_path}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   # Keep <= --max-tokens-per-gpu under --use-dynamic-batch-size (Miles docs).
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)
eval_path=${data_dir}/qwen-cot
mapfile -t eval_files < <(ls "${eval_path}"/*.jsonl 2>/dev/null | sort)
if [ ${#eval_files[@]} -eq 0 ]; then
  echo "No eval jsonl files found under ${eval_path}"
  exit 1
fi

# EVAL_ARGS=(
#    --eval-interval ${interval}
#    --n-samples-per-eval-prompt 4
#    --eval-max-response-len 8192
#    --eval-top-p 1
# )

EVAL_PROMPT_DATA_ARGS=()
for eval_file in "${eval_files[@]}"; do
  eval_name="$(basename "${eval_file}" .jsonl)"
  EVAL_PROMPT_DATA_ARGS+=("${eval_name}" "${eval_file}")
done
EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA_ARGS[@]}")

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   # MiniMax-M2.5: num_layers=62 -> PP must divide 62 (valid: 1,2,31,62). PP=2 is safe.
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   # Lower than 8192 to reduce OOM under --colocate with large MoE + grad buffers.
   --max-tokens-per-gpu 2048
   --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator gspo
   #--use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
   --use-rollout-routing-replay 
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)


WANDB_ARGS=()
if [[ -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${project_name}"
    --wandb-group "${variant}"
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  echo "WARN: WANDB_API_KEY / WANDB_KEY unset; skipping --use-wandb"
fi
TB_ARGS=(
   --use-tensorboard
   --tb-dir "${DATA_ROOT}/tb"
   --tb-project-name ${project_name}
   --tb-experiment-name ${variant}
)
# SGLang: MoE + colocate — leave headroom for Megatron init (Miles walkthrough).
# Minimum practical engine width is often 4 GPUs (tp_size=4); tune via env if needed.

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 512


   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   

   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"

if [[ "${MILES_USE_EXTERNAL_RAY:-0}" != "1" ]]; then
  # launch the master node of ray in container and workers via SSH (MPI rack hostfile)
  ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
    if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
  done
  wait
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"working_dir\": \"${WORKING_DIR}\",
  \"py_modules\": [\"${FULLY_ASYNC_ROLLOUT_DIR}\"],
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${FULLY_ASYNC_ROLLOUT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="${RAY_JOB_SUBMIT_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 8 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 64 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
