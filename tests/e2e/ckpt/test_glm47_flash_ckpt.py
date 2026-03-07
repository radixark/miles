import os
from argparse import ArgumentParser

import miles.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))
USE_DEEPEP = bool(int(os.environ.get("MILES_TEST_USE_DEEPEP", "0")))

MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-flash"
NUM_GPUS = 8


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")


def _get_latest_checkpointed_iteration() -> int:
    latest_path = f"/root/models/{MODEL_NAME}_miles/latest_checkpointed_iteration.txt"
    with open(latest_path, encoding="utf-8") as f:
        latest_text = f.read().strip()
    if not latest_text.isdigit():
        raise ValueError(f"Invalid latest checkpoint value: {latest_text}")
    return int(latest_text)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    # GLM-4.7-Flash requires a newer transformers version.
    U.exec_command(
        "pip install git+https://github.com/huggingface/transformers.git@"
        "76732b4e7120808ff989edbd16401f61fa6a0afa --break-system-packages"
    )
    U.exec_command(f"hf download zai-org/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"rm -rf /root/models/{MODEL_NAME}_miles")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute(mode: str = "", ckpt_step: int | None = None):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    if mode == "save":
        ckpt_args += f"--save /root/models/{MODEL_NAME}_miles "
        ckpt_args += "--save-interval 2 "
    elif mode == "async_save":
        ckpt_args += f"--save /root/models/{MODEL_NAME}_miles "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
        ckpt_args += "--use-persistent-ckpt-worker "
    elif mode == "load":
        ckpt_args += f"--load /root/models/{MODEL_NAME}_miles "
        ckpt_args += f"--ckpt-step {ckpt_step} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = ""
    if ENABLE_EVAL:
        eval_args = (
            "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 2048 "
            "--eval-top-k 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if TIGHT_HOST_MEMORY else 32768} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        f"{'' if TIGHT_HOST_MEMORY else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-rollout-routing-replay "
        "--use-miles-router "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        f"--sglang-mem-fraction-static {0.7 if TIGHT_HOST_MEMORY else 0.8} "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )

    if USE_DEEPEP:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "

    mtp_args = "--enable-mtp-training --mtp-loss-scaling-factor 0.2 "

    ci_args = "--ci-test "
    if mode in {"save", "async_save"}:
        ci_args += "--ci-save-model-hash "
    if mode == "load":
        ci_args += "--ci-check-model-hash "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    if USE_DEEPEP:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{mtp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "MILES_TEST_R3_THRESHOLD": "1.0",
        },
    )


if __name__ == "__main__":
    args = parser.parse_args()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("save" if not args.async_save else "async_save")
    latest_step = _get_latest_checkpointed_iteration()
    execute("load", ckpt_step=latest_step)
