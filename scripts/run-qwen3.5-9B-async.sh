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
export WANDB_KEY=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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
model_name=Qwen3.5-9B
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-9B.sh"
model_dir=/home/yangchengyi/data/models
ckpt_dir=/home/yangchengyi/data/ckpts

project_name=miles-dev
variant=${model_name}_miles_test
CKPT_ARGS=(
   --hf-checkpoint ${model_dir}/${model_name}
   --ref-load ${model_dir}/${model_name}_torch_dist
   --load ${ckpt_dir}/${variant}
   --save ${ckpt_dir}/${variant}
   --save-interval 50
)
data_dir=/home/yangchengyi/data/datasets
data_path=${data_dir}/deepmath-103k_miles.jsonl
log_dir=/home/yangchengyi/data/logs
log_path=${log_dir}/${model_name}_dapo_miles_$(date +%Y%m%d_%H%M%S).log
WORKING_DIR=/home/yangchengyi/data/miles
mkdir -p "${log_dir}"
exec > >(tee -a "${log_path}") 2>&1

ROLLOUT_ARGS=(
   --prompt-data ${data_path}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)
eval_path=${data_dir}/qwen-cot
mapfile -t eval_files < <(ls "${eval_path}"/*.jsonl 2>/dev/null | sort)
if [ ${#eval_files[@]} -eq 0 ]; then
  echo "No eval jsonl files found under ${eval_path}"
  exit 1
fi

EVAL_ARGS=(
   --eval-interval 50
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 8192
   --eval-top-p 1
)

EVAL_PROMPT_DATA_ARGS=()
for eval_file in "${eval_files[@]}"; do
  eval_name="$(basename "${eval_file}" .jsonl)"
  EVAL_PROMPT_DATA_ARGS+=("${eval_name}" "${eval_file}")
done
EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA_ARGS[@]}")

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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
   --wandb-project ${project_name}
   --wandb-group ${variant}
   --wandb-key ${WANDB_KEY}
)
TB_ARGS=(
   --use-tensorboard
   --tb-dir /home/yangchengyi/data/tb
   --tb-project-name ${project_name}
   --tb-experiment-name ${variant}
)
SGLANG_ARGS=(
   # Workaround: SGLang TP>1 produces garbage output for Qwen3.5 (https://github.com/sgl-project/sglang/issues/21039)
   # This is fixed in sglang main branch & dev docker, but miles still uses 0.5.9, so use TP=1 per engine for now
   --rollout-num-gpus-per-engine 1
   # Colocate：为 logits/激活留出余量；math_eval 曾在 mem 0.6 + 高并发下 OOM（见 logits_processor）
   --sglang-mem-fraction-static 0.52
   --sglang-max-running-requests 96
   --sglang-chunked-prefill-size 4096
   # 限制 generate 侧 semaphore（默认 512×GPU 数 在 eval 时仍偏大）
   --sglang-server-concurrency 96
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
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="${RAY_JOB_SUBMIT_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
