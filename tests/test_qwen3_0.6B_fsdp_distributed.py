import os
import command_utils as U

MODEL_NAME = "Qwen3-0.6B"


FEW_GPU = bool(int(os.environ.get("MILES_TEST_FEW_GPU", "1")))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/Qwen3-0.6B --local-dir /root/models/{MODEL_NAME}")
    U.exec_command("huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 128 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        #"--use-kl-loss "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {2 if FEW_GPU else 4} "
        "--colocate "
        "--slime-backend fsdp "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=2 if FEW_GPU else 4,
        model_type=None,
    )


if __name__ == "__main__":
    prepare()
    execute()
