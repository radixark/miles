import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.ft.conftest_ft.execution import DATA_DIR, MODEL_DIR
from tests.utils.ft.launch import get_fully_async_args
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.core.utils import API_SERVER_ARGS

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainConfig, LaunchGuard
from miles.utils.pydantic_utils import FrozenStrictBaseModel

FT_COMPONENTS: tuple[str, ...] = ("train", "rollout")
DEFAULT_SEED: int = 42
DEFAULT_NUM_ROLLOUT: int = 250
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 600.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 1200.0
# Must stay identical to the threshold asserted by the no-fault baseline
# tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py: fault recovery must not cost accuracy.
DEFAULT_METRIC_THRESHOLD: float = 0.55
MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE: str = "qwen2.5-0.5B"
# Same disaggregated layout as the kill_train__dp2_cp2__moe_5layer mode: 2 cells x CP2 on
# 4 training GPUs, plus 4 rollout engines x 1 GPU.
TRAIN_GPUS: int = 4
ROLLOUT_GPUS: int = 4
CONTEXT_PARALLEL_SIZE: int = 2
ROLLOUT_GPUS_PER_ENGINE: int = 1


class Gsm8kLaunchSpec(FrozenStrictBaseModel):
    config: ExecuteTrainConfig
    train_args: str
    fully_async: bool = False


@dataclass(frozen=True)
class Gsm8kRun:
    base_url: str
    dump_dir: str
    evidence_dir: Path
    launch_spec: Gsm8kLaunchSpec
    event_log: EventLog

    @property
    def events_dir(self) -> Path:
        return Path(self.dump_dir) / EVENTS_DIRNAME


def prepare_gsm8k_run(
    *,
    config: command_utils.ExecuteTrainConfig,
    test_name: str,
    seed: int,
    num_rollout: int,
    build_extra_train_args: Callable[[str], str],
    metric_threshold: float = DEFAULT_METRIC_THRESHOLD,
    fully_async: bool = False,
    enable_fault_tolerance: bool = True,
) -> Gsm8kRun:
    raise NotImplementedError


async def execute_gsm8k_session(run: Gsm8kRun) -> LaunchOutcome:
    raise NotImplementedError


async def launch(spec: Gsm8kLaunchSpec, *, guard: LaunchGuard | None = None) -> None:
    raise NotImplementedError


def prepare_gsm8k(U: BaseCommandBackend) -> None:
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=TRAIN_GPUS,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}",
        dir_dst=MODEL_DIR,
        megatron_path=os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM"),
    )
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)


def get_gsm8k_train_args(
    *,
    seed: int,
    num_rollout: int,
    test_name: str,
    metric_threshold: float = DEFAULT_METRIC_THRESHOLD,
    fully_async: bool = False,
    enable_fault_tolerance: bool = True,
) -> str:
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    ) + get_fully_async_args(fully_async=fully_async)

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data gsm8k {DATA_DIR}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        # Parallelism mirrors the kill_train__dp2_cp2__moe_5layer mode (2 cells x CP2), not
        # the no-fault baseline test.
        "--context-parallel-size 2 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus {ROLLOUT_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = API_SERVER_ARGS
    if enable_fault_tolerance:
        fault_tolerance_args += (
            "--use-fault-tolerance " f"--ft-components {' '.join(FT_COMPONENTS)} " "--mini-ft-controller-enable "
        )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        f"--ci-metric-checker-threshold {metric_threshold} "
    )

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
        f"--actor-num-gpus-per-node {TRAIN_GPUS} "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(f'test_{test_name}.py', run_name_prefix=f'seed{seed}')} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fault_tolerance_args} "
        f"{ci_args} "
        f"{misc_args} "
    )
