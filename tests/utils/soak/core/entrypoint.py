from collections.abc import Awaitable
from pathlib import Path

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.types import SoakForms, SoakObserver

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig


async def run_soak(
    *,
    config: ExecuteTrainConfig,
    dump_dir: Path,
    sut_run: Awaitable[object],
    runner_config: SoakRunnerConfig,
    forms: SoakForms,
    event_log: EventLog,
    observer: SoakObserver,
    evidence_dir: Path,
) -> SoakRunner:
    raise NotImplementedError
