#!/bin/bash

# Multi-teacher OPD over LoRA teachers: route each sample to a task-specific
# adapter sharing one frozen base.
#
# Student:  Qwen3-8B, trained from M0 (2 training GPUs + 4 rollout GPUs)
# Teachers: M0 + math_lora / code_lora / agentic_lora, all on ONE server (GPU 7)
#
# Each teacher is a LoRA adapter over the same frozen M0 the student starts from,
# so N teachers cost one base copy plus N adapters instead of N model replicas,
# and tokenizer agreement with the student is automatic rather than assumed.
#
# Routing is per sample via sample.metadata["opd_teacher"] and
# --opd-teacher-adapters; every teacher shares the endpoint in --rm-url and is
# selected per request by lora_path. Scoring cost is identical to single-teacher
# OPD: each rollout is scored once, by exactly one teacher.
#
# Adapters here are the directories written by the multi-LoRA Tinker gateway's
# save_weights_for_sampler (see examples/multi_lora); any PEFT adapter over the
# same base works too.
#
# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd-lora-teachers.sh

set -ex

TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141

# The frozen base every teacher adapter sits on. This is M0 — the same checkpoint
# the student starts from — so student and teachers share a tokenizer by construction.
TEACHER_BASE=/root/Qwen3-8B
TEACHER_ADAPTER_DIR=/root/opd-teachers
TEACHER_LORA_RANK=32

# One server for all teachers. Pinning keeps every adapter resident: SGLang caps
# pinned adapters at max-loras-per-batch - 1 (to stop pinning starving the pool),
# and the base model itself occupies a slot, so 3 pinned teachers need 4 slots.
# The triton backend is used because the default csgmv backend can skip MoE layers.
start_teacher() {
    local gpu=$1 port=$2
    local log_file
    log_file="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
    CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server \
        --model-path "$TEACHER_BASE" \
        --host 0.0.0.0 \
        --port "$port" \
        --tp 1 \
        --enable-lora \
        --lora-paths \
            "{\"lora_name\":\"math_lora\",\"lora_path\":\"$TEACHER_ADAPTER_DIR/math\",\"pinned\":true}" \
            "{\"lora_name\":\"code_lora\",\"lora_path\":\"$TEACHER_ADAPTER_DIR/code\",\"pinned\":true}" \
            "{\"lora_name\":\"agentic_lora\",\"lora_path\":\"$TEACHER_ADAPTER_DIR/agentic\",\"pinned\":true}" \
        --max-loras-per-batch 4 \
        --max-lora-rank "$TEACHER_LORA_RANK" \
        --lora-backend triton \
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

TEACHER_LOG=$(start_teacher 7 "$TEACHER_PORT")
wait_teacher "$TEACHER_IP" "$TEACHER_PORT" "$TEACHER_LOG"
sleep 10


# Build a mixed prompt set with per-sample teacher tags. Each row's metadata
# column (--metadata-key, default "metadata") carries the teacher name under
# the --opd-teacher-key (default "opd_teacher"):
#   {"prompt": "...", "metadata": {"opd_teacher": "math"}}
MIXED_DATA=/root/opd-lora-teachers/mixed.jsonl
mkdir -p "$(dirname $MIXED_DATA)"
python3 - << 'PYEOF'
import json
import os

sources = [
    ("/root/dapo-math-17k/dapo-math-17k.jsonl", "math"),
    # Add the prompt sets for your other specialists here, e.g.:
    # ("/root/code-prompts/code-prompts.jsonl", "code"),
    # ("/root/agentic-prompts/agentic-prompts.jsonl", "agentic"),
]

out_path = "/root/opd-lora-teachers/mixed.jsonl"
n = 0
with open(out_path, "w") as out:
    for path, teacher in sources:
        if not os.path.exists(path):
            print(f"skip missing {path}")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row.setdefault("metadata", {})["opd_teacher"] = teacher
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
print(f"wrote {n} tagged prompts to {out_path}")
PYEOF


export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_ARGS_LINE="$(python3 "${SCRIPT_DIR}/../../miles/utils/external_utils/model_args_utils.py" "qwen3-8B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"


CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_miles/
   --save /root/Qwen3-8B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $MIXED_DATA
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
   # Every teacher lives on this one endpoint; they differ only by adapter.
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
   # Per-task teacher routing; 'default' catches untagged/unknown samples.
   --opd-teacher-adapters
       math=math_lora
       code=code_lora
       agentic=agentic_lora
       default=math_lora
   --opd-teacher-key opd_teacher
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
   --opd-top-k-strategy only-student
   --opd-reward-weight-mode student_p
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
   # --wandb-group qwen3-8B-lora-teachers
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
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MILES_USE_LEGACY_ROLLOUT_V1": "1"
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
