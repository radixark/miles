from collections.abc import Awaitable
from pathlib import Path

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.entrypoint import run_soak
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.types import SoakForms
from tests.utils.soak.core.utils import compute_base_url
from tests.utils.soak.ft.observers import create_cell_observer

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig


async def run_cell_soak(
    *,
    config: ExecuteTrainConfig,
    dump_dir: Path,
    sut_run: Awaitable[object],
    runner_config: SoakRunnerConfig,
    event_log: EventLog,
    evidence_dir: Path,
    forms: SoakForms,
) -> SoakRunner:
    base_url = compute_base_url(config)
    cell_types = set(runner_config.target_configs)
    forms = {kind: forms[kind] for kind in cell_types}

    return await run_soak(
        config=config,
        dump_dir=dump_dir,
        sut_run=sut_run,
        runner_config=runner_config,
        forms=forms,
        event_log=event_log,
        observer=create_cell_observer(base_url=base_url, cell_types=cell_types, forms=forms, config=config),
        evidence_dir=evidence_dir,
    )
