"""E2E test for --stream-optimizer-state-to-disk (on top of --offload-train-target=disk).

Same colocate RL loop as test_qwen3_4B_offload_disk.py, but the fp32 main params and
Adam moments additionally live in per-bucket NVMe files and are streamed through the
GPU bucket-by-bucket during each optimizer step. Completing the run is only half the
check: a run where the store silently never engaged (flag lost on the way to the
actor, plugin import failure swallowed) trains just as happily, so `execute` also
asserts that every rank logged actual streaming steps, on top of the disk-offload
armed assertion inherited from the base test.
"""

import glob
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 4
OFFLOAD_DIR = "/root/train_offload_disk_stream"

register_cuda_ci(
    est_time=600,
    suite="stage-c-4-gpu-h200",
    labels=["miles-plugin"],
)
register_rocm_ci(
    est_time=600,
    suite="stage-c-4-gpu-mi350",
    labels=["miles-plugin", "amd"],
    disabled="FIXME: re-enable once this case passes on the MI350 runners.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def _assert_offloaded_to_disk():
    """Every rank must have armed disk offload under its own directory."""
    logs = glob.glob("/tmp/ray/session_latest/logs/worker-*")
    assert logs, "no Ray worker logs to check for the disk-offload path"

    armed = set()
    for path in logs:
        with open(path, errors="ignore") as f:
            for line in f:
                if "Train disk-offload reclaim armed" in line:
                    armed.add(line.split("reclaim armed for ")[1].split()[0])

    expected = {os.path.join(OFFLOAD_DIR, f"cell0_rank{rank}") for rank in range(NUM_GPUS)}
    assert armed == expected, f"expected disk offload armed for {sorted(expected)}, saw {sorted(armed)}"
    print(f"disk offload armed for {len(armed)} ranks under {OFFLOAD_DIR}")


def _assert_streamed():
    """Every rank must have executed real streaming steps.

    The store logs one `NVMe streaming step: ...` line per optimizer step from each
    training actor process, so the number of distinct worker logs carrying it equals
    the number of ranks. Reads Ray's post-job worker logs, same as the disk assertion.
    """
    logs = glob.glob("/tmp/ray/session_latest/logs/worker-*")
    assert logs, "no Ray worker logs to check for the streaming path"

    streamed = set()
    for path in logs:
        with open(path, errors="ignore") as f:
            if any("NVMe streaming step:" in line for line in f):
                streamed.add(path)

    assert (
        len(streamed) == NUM_GPUS
    ), f"expected {NUM_GPUS} ranks to log streaming steps, saw {len(streamed)}: {sorted(streamed)}"
    print(f"optimizer state streaming ran on {len(streamed)} ranks")


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
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
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

    # The feature under test: paused-actor disk offload plus NVMe optimizer-state
    # streaming during the step.
    offload_args = (
        "--offload-train "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {OFFLOAD_DIR} "
        "--offload-train-disk-chunk-mb 64 "
        "--stream-optimizer-state-to-disk "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{offload_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)

    _assert_offloaded_to_disk()
    _assert_streamed()


if __name__ == "__main__":
    prepare()
    execute()
