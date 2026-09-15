import logging
import os
from argparse import Namespace
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.audit_utils.event_analyzer.analyzer import run_sample_ownership_analysis
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.check import completed_actor_steps
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import SampleOwnershipViolation
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.external_utils import command_utils

logger = logging.getLogger(__name__)

register_cuda_ci(
    est_time=1200, suite="stage-c-8-gpu-h100", labels=["ckpt", "fully-async"], hardware=["hopper", "blackwell"]
)
register_rocm_ci(est_time=1200, suite="nightly-stage-c-8-gpu-mi350", labels=["ckpt", "fully-async"])

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8
ROLLOUT_BATCH_SIZE = 4
N_SAMPLES_PER_PROMPT = 2
SAMPLE_OWNERSHIP_GRACE_STEPS = 2


def _get_latest_checkpointed_iteration() -> int:
    latest_path = f"/root/models/{MODEL_NAME}_miles/latest_checkpointed_iteration.txt"
    with open(latest_path, encoding="utf-8") as f:
        latest_text = f.read().strip()
    if not latest_text.isdigit():
        raise ValueError(f"Invalid latest checkpoint value: {latest_text}")
    return int(latest_text)


def _prepare() -> None:
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu(f"rm -rf /root/models/{MODEL_NAME}_miles")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def _execute(mode: str, *, missing_training_step: bool = False) -> None:
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    if mode == "save":
        ckpt_args += f"--save /root/models/{MODEL_NAME}_miles "
        ckpt_args += "--save-interval 2 "
    elif mode == "load":
        ckpt_args += f"--load /root/models/{MODEL_NAME}_miles "
        ckpt_args += f"--ckpt-step {_get_latest_checkpointed_iteration()} "
        ckpt_args += "--low-memory-resume "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 12 "
        f"--rollout-batch-size {ROLLOUT_BATCH_SIZE} "
        f"--n-samples-per-prompt {N_SAMPLES_PER_PROMPT} "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
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

    sglang_args = "--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.7 --sglang-cuda-graph-bs 1 2 4 8 16 "

    ci_args = f"--ci-test --sample-ownership-grace-steps {SAMPLE_OWNERSHIP_GRACE_STEPS} "
    if mode == "save":
        ci_args += "--ci-save-model-hash --debug-exit-after-rollout 4 "
    if mode == "load":
        ci_args += "--ci-check-model-hash "

    if missing_training_step:
        ci_args += "--ci-inject-missing-prefetched-batch-bug "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 --fully-async --pause-generation-mode in_place "
        f"--save-debug-event-data /root/models/{MODEL_NAME}_miles/events "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
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


def run(*, missing_training_step: bool = False) -> None:
    _prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    _execute("save")
    if missing_training_step:
        try:
            _execute("load", missing_training_step=True)
        except Exception:
            logger.exception("Training failed; checking that the sample ownership checker rejected the injected loss")
            _assert_missing_sample()
        else:
            raise AssertionError("The sample ownership checker missed the injected batch loss")
    else:
        _execute("load")


def _assert_missing_sample() -> None:
    directory = Path(f"/root/models/{MODEL_NAME}_miles/events")
    assert completed_actor_steps(read_events(directory, strict=True)), "the resumed run recorded no completed step"
    args = Namespace(
        enable_sample_ownership_checker=True,
        sample_ownership_grace_steps=SAMPLE_OWNERSHIP_GRACE_STEPS,
        ci_test=True,
    )
    try:
        run_sample_ownership_analysis(args=args, event_dir=directory)
    except SampleOwnershipViolation as violation:
        untrained = [
            issue for issue in violation.issues if issue.description == "source sample had no training outcome"
        ]
        assert len(untrained) >= ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT, violation.issues
    else:
        raise AssertionError("The injected batch loss left no sample ownership violation behind")


if __name__ == "__main__":
    run()
