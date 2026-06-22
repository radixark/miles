#!/bin/bash

# Teacher-ensemble OPD: every sample is scored by a GROUP of teachers whose
# distributions are combined as a weighted mixture in probability space
# (logsumexp of weighted logprobs), with the exact tail-bucket top-k KL.
#
# Student:    Qwen3-8B (2 training GPUs + 4 rollout GPUs)
# Teacher A:  Qwen3-32B            (GPU 6) weight 2.0
# Teacher B:  Qwen3-30B-A3B        (GPU 7) weight 1.0
#
# Both teachers score every sample in parallel (asyncio.gather), so scoring
# wall clock is max(teacher latencies), not the sum, and training-step time is
# unchanged versus single-teacher OPD. Ensemble only same-domain peers — for
# task specialists, use per-sample routing instead (see
# run-qwen3-8B-opd-multi-teacher.sh); the two compose: each routed name can be
# its own ensemble group. All teachers must share the student's tokenizer
# (scoring sends input_ids), which holds for the Qwen3 family used here.
#
# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd-ensemble.sh

set -ex

TEACHER_A_IP="127.0.0.1"
TEACHER_A_PORT=13141
TEACHER_B_IP="127.0.0.1"
TEACHER_B_PORT=13142

start_teacher() {
    local gpu=$1 model_path=$2 port=$3
    local log_file
    log_file="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
    CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --host 0.0.0.0 \
        --port "$port" \
        --tp 1 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.6 \
        > "$log_file" 2>&1 &
    echo "$log_file"
}

wait_teacher() {
    local ip=$1 port=$2 log_file=$3
    until curl -sf "http://$ip:$port/health_generate" > /dev/null; do
        echo "Waiting for teacher server at $ip:$port..."
        tail -n 10 "$log_file"
        sleep 5
    done
    curl "http://$ip:$port/get_model_info"
    echo "Teacher server is up at $ip:$port."
}

A_LOG=$(start_teacher 6 /root/Qwen3-32B "$TEACHER_A_PORT")
B_LOG=$(start_teacher 7 /root/Qwen3-30B-A3B "$TEACHER_B_PORT")
wait_teacher "$TEACHER_A_IP" "$TEACHER_A_PORT" "$A_LOG"
wait_teacher "$TEACHER_B_IP" "$TEACHER_B_PORT" "$B_LOG"
sleep 10


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/qwen3-8B.sh"


CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_miles/
   --save /root/Qwen3-8B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path miles.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
   # One 'default' group of two teachers, mixed 2:1 in probability space.
   # Every sample is scored by both members in parallel.
   --opd-teacher-urls
       default=http://$TEACHER_A_IP:$TEACHER_A_PORT/generate@2.0,http://$TEACHER_B_IP:$TEACHER_B_PORT/generate@1.0
   # Bound teacher scoring independently of the shared router timeout.
   --opd-scoring-timeout-secs 600
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
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
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --opd-log-prob-top-k 16
   # Ensembles require the student-side token set: every member is scored at
   # the same per-position student top-k ids so raw probabilities mix exactly.
   --opd-top-k-strategy only-student
   --opd-reward-weight-mode student_p
   # Exact (k+1)-bucket reverse KL: top-k ids + tail mass, no renormalization.
   --opd-topk-tail-bucket
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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
   #--use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen3-8B-opd-ensemble
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


# launch the master node of ray in container
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
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}


####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
