import os

import miles.utils.misc as U
from miles.utils.external_utils.command_utils import execute_train_npu

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-4B-Instruct-2507")

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "4"))
EXTERNAL_RAY = int(os.environ.get("MILES_SCRIPT_EXTERNAL_RAY", "0"))
TRAIN_BACKEND = os.environ.get("MILES_SCRIPT_TRAIN_BACKEND", "fsdp").lower()
assert TRAIN_BACKEND in {"fsdp", "megatron"}

DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT = "/root/dataset/geo3k_imgurl_processed"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")


def get_megatron_model_type(model_name: str) -> str:
    megatron_model_type = {
        "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
        "Qwen3-4B-Base": "qwen3-4B",
        "Qwen3-4B": "qwen3-4B",
    }[model_name]
    return megatron_model_type


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    data_missing = not os.path.exists(TRAIN_DATA_PATH)
    if data_missing:
        U.exec_command(f"hf download --repo-type dataset {DATASET_NAME} --local-dir {DATA_ROOT}")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found. Expected local dataset at {TRAIN_DATA_PATH}; ")


def execute():
    ckpt_args = "--hf-checkpoint /root/model/Qwen3-4B-Instruct-2507/ "

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project miles-dev "
            "--wandb-group geo3k_vlm_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    rollout_args = (
        "--prompt-data /root/dataset/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # By default it is thinking mode
        # """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    # eval_args = (
    #     "--eval-interval 20 "
    #     f"--eval-prompt-data geo3k_eval {TRAIN_DATA_PATH}@[0:64] "
    #     "--n-samples-per-eval-prompt 1 "
    #     "--eval-max-response-len 4096 "
    #     "--eval-top-k 1 "
    # )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [4, 8] + list(range(16, 257, 8))))} "
        "--sglang-mm-attention-backend ascend_attn "
        "--sglang-device npu "
        "--sglang-disable-radix-cache "
        "--sglang-chunked-prefill-size 32768 "
        "--sglang-max-prefill-tokens 4000 "
        "--sglang-max-total-tokens 327680 "
    )

    megatron_args = (
        "--train-backend megatron "
        "--load /root/model/Qwen3-4B-Instruct-2507/ "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--no-gradient-accumulation-fusion "
        "--use-flash-attn "
    )

    backend_args = megatron_args
    megatron_model_type = get_megatron_model_type(MODEL_NAME)
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{misc_args} "
        f"{wandb_args} "
        # f"{get_default_wandb_args(__file__)} "
    )

    execute_train_npu(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=megatron_model_type,
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
        megatron_path="/root/Megatron-LM/",
    )


if __name__ == "__main__":
    # prepare()
    execute()
