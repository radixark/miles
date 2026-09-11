import logging
import os
from datetime import timedelta
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci

from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.check import check
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore
from miles.utils.external_utils import command_utils

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["ckpt", "fully-async"])

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


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
        ckpt_args += "--ckpt-step 3 "
        ckpt_args += "--low-memory-resume "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 12 "
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

    ci_args = "--ci-test --sample-ownership-check-interval-seconds 0.001 "
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
    if not missing_training_step:
        _execute("load")
        return
    try:
        _execute("load", missing_training_step=True)
    except Exception:
        logger.exception("Training failed; checking for the injected sample loss")
        U = command_utils.default_config().create_backend()
        U.exec_command_cpu(
            "python -c 'from tests.e2e.ckpt.test_fully_async_ckpt import _assert_missing_sample; _assert_missing_sample()'"
        )
    else:
        raise AssertionError("The sample ownership checker missed the injected batch loss")


def _assert_missing_sample() -> None:
    store = SampleOwnershipEventStore(
        EventLogger(
            log_dir=Path(f"/root/models/{MODEL_NAME}_miles/events"),
            source=SimpleProcessIdentity(component="rollout_executor"),
        )
    )
    snapshot = store.read_current()
    assert snapshot is not None and snapshot.marker.mature_before is not None
    issues = check(store.read_events(), grace_period=timedelta(), now=snapshot.marker.mature_before)
    assert any(issue.description == "source sample had no training outcome" for issue in issues), issues


if __name__ == "__main__":
    run()
