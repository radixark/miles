import os

import miles.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen2.5-3B"
MODEL_TYPE = "qwen2.5-3B"
NUM_GPUS = 2


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 128 "
        "--global-batch-size 4 "
    )

    grpo_args = "--advantage-estimator grpo " "--rm-type math "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--sglang-mem-fraction-static 0.6 "
    )

    checker_args = "--enable-weight-checker " "--update-weights-interval 1 "

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    train_args = (
        f"{ckpt_args} " f"{rollout_args} " f"{grpo_args} " f"{sglang_args} " f"{checker_args} " f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
