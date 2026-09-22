import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tests.e2e.ft.conftest_ft.execution import get_fully_async_args, get_train_script, launch_training
from tests.fast.cluster_backends import create_backend_for_run
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.types import SoakForms, SoakObserver
from tests.utils.soak.core.utils import API_SERVER_ARGS, DATA_DIR, MODEL_DIR, assert_fresh_dump_dir, resolve_dump_dir

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainConfig, LaunchGuard
from miles.utils.pydantic_utils import FrozenStrictBaseModel

FT_COMPONENTS: tuple[str, ...] = ("train", "rollout")
DEFAULT_SEED: int = 42
DEFAULT_NUM_ROLLOUT: int = 250
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 600.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 1200.0
DEFAULT_METRIC_THRESHOLD: float = 0.55
MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE: str = "qwen2.5-0.5B"
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
    event_log: EventLog = field(default_factory=EventLog)

    @property
    def events_dir(self) -> Path:
        return Path(self.dump_dir) / EVENTS_DIRNAME


@dataclass(frozen=True)
class Gsm8kOutcome:
    run: Gsm8kRun
    injector: SoakRunner
    forms: SoakForms


CreateSoakFormsFn = Callable[[Gsm8kRun], SoakForms]


async def run_realistic_gsm8k(
    *,
    config: command_utils.ExecuteTrainConfig,
    test_name: str,
    seed: int,
    num_rollout: int,
    mean_interval_seconds_of_kind: dict[str, float],
    expected_counts: dict[str, int],
    create_forms: CreateSoakFormsFn,
    build_extra_train_args: Callable[[str], str],
    metric_threshold: float = DEFAULT_METRIC_THRESHOLD,
    fully_async: bool = False,
    enable_fault_tolerance: bool = True,
    create_observer: Callable[[Gsm8kRun, SoakForms], SoakObserver] | None = None,
    execute_session: Callable[[Gsm8kRun], Coroutine[Any, Any, None]] | None = None,
) -> Gsm8kOutcome:
    raise NotImplementedError


def _prepare_gsm8k_run(*, config: command_utils.ExecuteTrainConfig, test_name: str) -> str:
    prepare_gsm8k(create_backend_for_run(config))
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    dump_dir: str = resolve_dump_dir(test_name, run_id=config.run_id)
    assert_fresh_dump_dir(Path(dump_dir))
    return dump_dir


def _build_gsm8k_train_args(
    *,
    dump_dir: str,
    seed: int,
    num_rollout: int,
    metric_threshold: float,
    fully_async: bool,
    test_name: str,
    enable_fault_tolerance: bool,
    build_extra_train_args: Callable[[str], str],
) -> str:
    train_args = get_gsm8k_train_args(
        seed=seed,
        num_rollout=num_rollout,
        metric_threshold=metric_threshold,
        fully_async=fully_async,
        test_name=test_name,
        enable_fault_tolerance=enable_fault_tolerance,
    )
    train_args += f"--save-debug-event-data {dump_dir}/{EVENTS_DIRNAME} "
    return train_args + build_extra_train_args(dump_dir)


async def execute_gsm8k_session(run: Gsm8kRun) -> Literal["finished", "replaced"]:
    raise NotImplementedError


async def launch(spec: Gsm8kLaunchSpec, *, guard: LaunchGuard | None = None) -> None:
    await asyncio.to_thread(
        launch_training,
        train_args=spec.train_args,
        num_gpus_per_node=TRAIN_GPUS + ROLLOUT_GPUS,
        megatron_model_type=MODEL_TYPE,
        config=spec.config,
        train_script=get_train_script(fully_async=spec.fully_async),
        guard=guard,
    )


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
        f"--context-parallel-size {CONTEXT_PARALLEL_SIZE} " "--use-dynamic-batch-size " "--max-tokens-per-gpu 9216 "
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
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = API_SERVER_ARGS
    fault_tolerance_args += "--update-weight-transfer-mode p2p "
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
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
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
