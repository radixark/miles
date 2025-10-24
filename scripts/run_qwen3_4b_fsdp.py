import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U

MODEL_NAME = "Qwen3-4B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")


def execute():
    # load_save_path = (
    #     f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}__{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/"
    # )
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        # f"--ref-load /root/{MODEL_NAME}_torch_dist "
        # f"--load {load_save_path} "
        # f"--save {load_save_path} "
        # "--save-interval 20 "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        # TODO temp
        # "--rollout-max-response-len 8192 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = (
        # TODO temp
        # "--eval-interval 20 "
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 0.7 "
    )

    perf_args = "--use-dynamic-batch-size " "--max-tokens-per-gpu 4096 "

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 "

    fsdp_args = (
        "--train-backend fsdp "
        # "--gradient-checkpointing "
        # "--attn-implementation flash_attention_2 "
        f"--update-weights-bucket-size {512 * 1024 * 1024} "
        # "--fsdp-full-params "
    )

    misc_args = "--actor-num-nodes 1 " "--actor-num-gpus-per-node 8 " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fsdp_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=None,
    )


if __name__ == "__main__":
    prepare()
    execute()
