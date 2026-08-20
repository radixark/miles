"""E2E test for dist_muon with its optimizer state on disk.

The colocate loop from test_qwen3_4B_offload_disk_stream.py with --optimizer dist_muon.
Completing the run is only half the check: if the disk backend silently never engaged, the
run trains just as happily against pinned host memory, so `execute` also asserts every rank
logged real disk-backed steps.

At this rollout size every sample truncates, so the metric gates below sit at zero and
cannot catch a regression on their own -- as in the Adam test this mirrors.
"""

import glob
import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

from miles.utils.external_utils import command_utils
from miles.utils.workers.naming import format_name_index

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 4
OFFLOAD_DIR = "/root/train_offload_muon_disk"

register_cuda_ci(
    est_time=600,
    suite="stage-c-4-gpu-h200",
    labels=["miles-plugin", "megatron"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="rollout/raw_reward")


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def _assert_disk_backed_steps():
    """Every rank must have run optimizer steps against file-backed state."""
    logs = glob.glob("/tmp/ray/session_latest/logs/worker-*")
    assert logs, "no Ray worker logs to check for the disk-backed state path"

    backed = set()
    for path in logs:
        with open(path, errors="ignore") as f:
            if any("Muon disk state step:" in line for line in f):
                backed.add(path)

    assert (
        len(backed) == NUM_GPUS
    ), f"expected {NUM_GPUS} ranks to log disk-backed optimizer steps, saw {len(backed)}: {sorted(backed)}"
    print(f"Muon optimizer state was file-backed on {len(backed)} ranks")


def _assert_offloaded_to_disk():
    """Every rank must have armed the paused-actor disk offload under its own directory."""
    logs = glob.glob("/tmp/ray/session_latest/logs/worker-*")
    assert logs, "no Ray worker logs to check for the disk-offload path"

    armed = set()
    for path in logs:
        with open(path, errors="ignore") as f:
            for line in f:
                if "Train disk-offload reclaim armed" in line:
                    armed.add(line.split("reclaim armed for ")[1].split()[0])

    expected = {
        os.path.join(OFFLOAD_DIR, f"cell{format_name_index(0)}_rank{format_name_index(rank)}")
        for rank in range(NUM_GPUS)
    }
    assert armed == expected, f"expected disk offload armed for {sorted(expected)}, saw {sorted(armed)}"
    print(f"disk offload armed for {len(armed)} ranks under {OFFLOAD_DIR}")


def execute():
    U = command_utils.default_config().create_backend()
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
        "--optimizer dist_muon "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # The feature under test: Muon's chunked optimizer state in files rather than
    # pinned host memory, on top of paused-actor disk offload.
    offload_args = (
        "--offload-train "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {OFFLOAD_DIR} "
        "--offload-train-disk-chunk-mb 64 "
        "--chunked-optimizer-state-offload "
        "--optimizer-state-offload-fraction 1.0 "
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
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)

    _assert_offloaded_to_disk()
    _assert_disk_backed_steps()


if __name__ == "__main__":
    prepare()
    execute()
