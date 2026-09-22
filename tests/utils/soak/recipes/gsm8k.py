from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.types import SoakForms, SoakObserver

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


async def execute_gsm8k_session(run: Gsm8kRun) -> Literal["finished", "replaced"]:
    raise NotImplementedError


async def launch(spec: Gsm8kLaunchSpec, *, guard: LaunchGuard | None = None) -> None:
    raise NotImplementedError


def prepare_gsm8k(U: BaseCommandBackend) -> None:
    raise NotImplementedError
