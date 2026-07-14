from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8

register_cuda_ci(
    est_time=600,
    suite="stage-c-8-gpu-h100",
    labels=["megatron", "weight-update"],
)
register_rocm_ci(
    est_time=600,
    suite="stage-c-8-gpu-mi300x",
    labels=["megatron", "weight-update"],
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        "--rollout-num-gpus-per-engine 2 " f"--rollout-num-gpus {NUM_GPUS // 2} " "--sglang-mem-fraction-static 0.8 "
    )

    ci_args = "--ci-test "

    # Single node, so the "shared" publish dir and the host-local checkpoint are
    # both plain local paths; the delta pipeline is exercised end to end
    # (baseline pull + reload, per-sync publish -> /pull_weights apply -> reload).
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS // 2} "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        "--update-weight-transfer-mode disk-delta "
        "--update-weight-disk-dir /root/delta-updates "
        "--update-weight-local-checkpoint-dir /root/delta-local-ckpt "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    execute()
