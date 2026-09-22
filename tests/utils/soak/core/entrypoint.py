import random
from collections.abc import Coroutine
from functools import partial
from pathlib import Path
from typing import Any

from tests.utils.soak.core.checkers.tail_completeness import assert_tail_complete
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetPolicy
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakRunContext, SoakRunContextEvent
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.scheduler import POLL_INTERVAL_SECONDS, QUIESCENT_POLLS_REQUIRED, SoakActionScheduler
from tests.utils.soak.core.sut_events import SutEventFeed
from tests.utils.soak.core.teardown import teardown_run
from tests.utils.soak.core.types import SoakForms, SoakObserver
from tests.utils.soak.core.utils import compute_base_url, evidence_directory

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
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
    evidence_dir = evidence_dir if evidence_dir is not None else evidence_directory(dump_dir)

    runner = _create_runner(
        config=config,
        dump_dir=dump_dir,
        seed=seed,
        mean_interval_seconds_of_kind=mean_interval_seconds_of_kind,
        expected_counts=expected_counts,
        runner_config=runner_config,
        forms=forms,
        event_log=event_log,
        observer=observer,
        poll_interval_seconds=poll_interval_seconds,
        quiescent_polls_required=quiescent_polls_required,
        evidence_dir=evidence_dir,
    )

    await runner.run(
        training,
        teardown=partial(teardown_run, config=config, event_log=runner.event_log, evidence_dir=evidence_dir),
    )

    assert_tail_complete(runner.event_log.events)
    return runner


def _create_runner(
    *,
    config: ExecuteTrainConfig,
    dump_dir: Path,
    seed: int,
    mean_interval_seconds_of_kind: dict[str, float],
    expected_counts: dict[str, int],
    runner_config: SoakRunnerConfig,
    forms: SoakForms,
    event_log: EventLog,
    observer: SoakObserver,
    poll_interval_seconds: float,
    quiescent_polls_required: int,
    evidence_dir: Path,
) -> SoakRunner:
    resolved_config = runner_config.model_copy(
        update={
            "target_policies": {
                kind: SoakTargetPolicy(expected_count=count)
                for kind, count in expected_counts.items()
                if kind in mean_interval_seconds_of_kind
            }
        }
    )
    event_log.persist_to(evidence_dir / "events.jsonl")
    sources = {"training_events": dump_dir / EVENTS_DIRNAME}

    event_log.append(
        SoakRunContextEvent(
            context=SoakRunContext(
                base_url=compute_base_url(config),
                seed=seed,
                config=resolved_config,
                mean_intervals=mean_interval_seconds_of_kind,
                form_names={kind: [form.name for form in kind_forms] for kind, kind_forms in forms.items()},
                train_config=config,
            ),
            sources=sources,
        )
    )

    return SoakRunner(
        observer=observer,
        scheduler=SoakActionScheduler(
            rng=random.Random(seed),
            mean_intervals=mean_interval_seconds_of_kind,
            forms=forms,
            quiescent_polls_required=quiescent_polls_required,
            config=resolved_config,
        ),
        forms={kind: forms[kind] for kind in mean_interval_seconds_of_kind},
        event_log=event_log,
        config=resolved_config,
        poll_interval_seconds=poll_interval_seconds,
        sut_events=SutEventFeed(directory=sources["training_events"]),
    )
