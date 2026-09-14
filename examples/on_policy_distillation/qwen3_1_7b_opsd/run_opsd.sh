#!/bin/bash

set -ex

MODEL_DIR=${MODEL_DIR:-/root}
DATA_DIR=${DATA_DIR:-/root}
TRAIN_DATA=${TRAIN_DATA:-/tmp/opsd-train.jsonl}
EVAL_DATA=${EVAL_DATA:-/tmp/opsd-aime24.jsonl}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

python3 "${SCRIPT_DIR}/prepare_data.py" \
    "${MODEL_DIR}/Qwen3-1.7B" \
    "${DATA_DIR}/openthoughts-math" \
    "${DATA_DIR}/aime24" \
    "$TRAIN_DATA" \
    "$EVAL_DATA"

TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path ${MODEL_DIR}/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.25 \
    > "$LOG_FILE" 2>&1 &
TEACHER_PID=$!
trap 'kill "$TEACHER_PID" 2>/dev/null || true' EXIT

echo "Starting teacher model server..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    sleep 5
done
echo "Teacher model server is up at $TEACHER_IP:$TEACHER_PORT."

export PYTHONUNBUFFERED=1

MODEL_ARGS_LINE="$(python3 "${SCRIPT_DIR}/../../../miles/utils/external_utils/model_args_utils.py" "qwen3-1.7B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-1.7B
   --ref-load ${MODEL_DIR}/Qwen3-1.7B_torch_dist
   --load ${MODEL_DIR}/Qwen3-1.7B_opsd/
   --save ${MODEL_DIR}/Qwen3-1.7B_opsd/
   --save-interval 25
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --rollout-max-response-len 1024
   --rollout-temperature 1.1
   --rollout-top-k 20

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.qwen3_1_7b_opsd.rm.reward_func
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   --eval-interval 25
   --eval-prompt-data aime24 ${EVAL_DATA}
   --eval-label-key label
   --n-samples-per-eval-prompt 12
   --eval-max-response-len 38912
   --eval-temperature 1.0
   --eval-top-p 0.95
   --eval-top-k -1
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 128
   --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
   --lora-dropout 0.0
   --megatron-to-hf-mode bridge
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
   --max-tokens-per-gpu 16384
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --opd-log-prob-top-k 256
   --opd-divergence forward_kl
   --loss-type custom_loss
   --custom-loss-function-path examples.on_policy_distillation.qwen3_1_7b_opsd.loss.loss_function
   --disable-compute-advantages-and-returns
   --custom-config-path "${SCRIPT_DIR}/config.yaml"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --clip-grad 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 5 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${OPD_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
