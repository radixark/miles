import os
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_fully_async_ckpt import assert_checkpoint_replayed, read_checkpoint_sample_indices

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["ckpt", "fully-async"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
SAVE_DIR = Path(f"/root/models/{MODEL_NAME}_fully_async_ckpt")
EVENT_DIR = Path("/root/dumps/fully_async_ckpt_events")
NUM_ROLLOUT = 4


def prepare() -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu(f"rm -rf {SAVE_DIR} {EVENT_DIR}")
    U.hf_download_dataset("zhuzilin/gsm8k")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute(mode: str, ckpt_step: int | None = None) -> None:
    U = command_utils.default_config().create_backend()
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--save {SAVE_DIR} "
        "--save-interval 2 "
    )
    if mode == "load":
        ckpt_args += f"--load {SAVE_DIR} --ckpt-step {ckpt_step} "
    else:
        ckpt_args += "--debug-exit-after-rollout 2 "

    rollout_args = (
        "--fully-async "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--max-weight-staleness 1 "
        "--async-unused-samples-handler retry "
        "--pause-generation-mode in_place "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.65 "

    audit_args = f"--save-debug-event-data {EVENT_DIR} "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 6 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{audit_args} "
        "--ci-test "
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
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("save")
    checkpoint_id = read_checkpoint_tracker_iteration(SAVE_DIR)
    assert checkpoint_id is not None
    assert checkpoint_id == 1
    saved_indices = read_checkpoint_sample_indices(save_dir=SAVE_DIR, rollout_id=checkpoint_id)
    execute("load", ckpt_step=checkpoint_id)
    assert_checkpoint_replayed(
        event_dir=EVENT_DIR, saved_indices=saved_indices, rollout_ids={None: checkpoint_id}, num_rollout=NUM_ROLLOUT
    )
