#!/bin/bash

# When Ray was started outside this script (e.g. code/ray_init_simple.sh), set MILES_USE_EXTERNAL_RAY=1
# to skip tearing down Ray and skip `ray start --head` below.
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-4B.sh"
model_dir=/home/yangchengyi/data/models
ckpt_dir=/home/yangchengyi/data/ckpts
model_name=Qwen3.5-4B
variant=${model_name}_miles_test
CKPT_ARGS=(
   --hf-checkpoint ${model_dir}/${model_name}
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load ${model_dir}/${model_name}_torch_dist
   --load ${ckpt_dir}/${variant}
   --save ${ckpt_dir}/${variant}
   --save-interval 20
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
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)
eval_path=${data_dir}/math-benches-deepmath-eval.jsonl
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data math_eval ${eval_path}
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 8192
   --eval-top-p 1
)

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
   --wandb-project miles-dev
   --wandb-group qwen3-4B-instruct-test
   --wandb-key ${WANDB_KEY}
)
TB_ARGS=(
   --use-tensorboard
   --tb-dir /home/yangchengyi/data/tb
   --tb-project-name miles-agentic
   --tb-experiment-name qwen3-4B-instruct-test
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Lower static fraction to leave ~36GB headroom per GPU. With 0.7 we saw
   # OOM with only ~3GB free when the prefill logits gather (~4.5 GiB) ran.
   --sglang-mem-fraction-static 0.55
   # Cap concurrent requests per engine. SGLang default 2048 was way above
   # what KV cache (732K tokens) can sustain, so it kept retracting and
   # leaving the allocator fragmented.
   --sglang-max-running-requests 128
   # Bound a single prefill batch token count so the logits tensor
   # (vocab=151936 * tokens * 2B) stays well under available memory.
   --sglang-chunked-prefill-size 4096
   # Cap Miles-side in-flight HTTP dispatch (default 512) to match the
   # decoding cap above. Prevents queue buildup that triggers retracts.
   --sglang-server-concurrency 128
)


PROMETHEUS_ARGS=(
   # --use-prometheus
   # --prometheus-port 9090
   # --prometheus-run-name "qwen3-4B-exp"
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

# launch the master node of ray in container (skipped when using external Ray cluster)
# export CUDA_VISIBLE_DEVICES=3,4,6,7
gpu_num=8
if [[ "${MILES_USE_EXTERNAL_RAY:-0}" != "1" ]]; then
  export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${gpu_num} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

if [[ "${MILES_USE_EXTERNAL_RAY:-0}" == "1" ]] && [[ -z "${MASTER_ADDR:-}" ]] && [[ -n "${POD_IP:-}" ]]; then
  export MASTER_ADDR="${POD_IP}"
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_JOB_SUBMIT_ADDRESS="${RAY_JOB_SUBMIT_ADDRESS:-http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}}"

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${WORKING_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
ray job submit --address="${RAY_JOB_SUBMIT_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${gpu_num} \
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
   ${PROMETHEUS_ARGS[@]} \
   ${MISC_ARGS[@]}
