from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.scheduler import POLL_INTERVAL_SECONDS, QUIESCENT_POLLS_REQUIRED
from tests.utils.soak.core.types import SoakForms, SoakObserver

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig


async def run_soak(
    *,
    config: ExecuteTrainConfig,
    dump_dir: Path,
    seed: int,
    mean_interval_seconds_of_kind: dict[str, float],
    expected_counts: dict[str, int],
    training: Coroutine[Any, Any, Any],
    runner_config: SoakRunnerConfig,
    forms: SoakForms,
    event_log: EventLog,
    observer: SoakObserver,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
    evidence_dir: Path | None = None,
) -> SoakRunner:
    raise NotImplementedError
