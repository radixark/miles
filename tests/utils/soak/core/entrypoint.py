from collections.abc import Awaitable
from functools import partial
from pathlib import Path

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakRunContext, SoakRunContextEvent
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.sut_events import SutEventFeed
from tests.utils.soak.core.teardown import teardown_run
from tests.utils.soak.core.types import SoakForms, SoakObserver
from tests.utils.soak.core.utils import compute_base_url

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
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
    runner = _create_runner(
        config=config,
        dump_dir=dump_dir,
        runner_config=runner_config,
        forms=forms,
        event_log=event_log,
        observer=observer,
    )

    await runner.run(
        sut_run,
        teardown=partial(teardown_run, config=config, event_log=runner.event_log, evidence_dir=evidence_dir),
    )

    return runner


def _create_runner(
    *,
    config: ExecuteTrainConfig,
    dump_dir: Path,
    runner_config: SoakRunnerConfig,
    forms: SoakForms,
    event_log: EventLog,
    observer: SoakObserver,
) -> SoakRunner:
    sources = {"training_events": dump_dir / EVENTS_DIRNAME}

    event_log.append(
        SoakRunContextEvent(
            context=SoakRunContext(
                base_url=compute_base_url(config),
                config=runner_config,
                form_names={kind: [form.name for form in kind_forms] for kind, kind_forms in forms.items()},
                train_config=config,
            ),
            sources=sources,
        )
    )

    return SoakRunner(
        observer=observer,
        forms=forms,
        event_log=event_log,
        config=runner_config,
        sut_events=SutEventFeed(
            directory=sources["training_events"],
            file_patterns=("trainer_controller_*.jsonl", "rollout_executor.jsonl"),
            event_types=(TrainGroupStepEndEvent, CellReconfigureEvent),
        ),
    )
