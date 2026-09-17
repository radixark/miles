# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import asyncio
import builtins
import random
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from tests.utils.soak.config import SoakPolicy, SoakTailPolicy, SoakTimeouts
from tests.utils.soak.core import POLL_INTERVAL_SECONDS, QUIESCENT_POLLS_REQUIRED, SoakActionScheduler
from tests.utils.soak.fault_forms import CellFaultForms, ExecSigkillFaultForm
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import EventLog, SoakRunContextEvent

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend, DeployComponent

API_SERVER_PORT: int = 18080
# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
SHUTDOWN_TIMEOUT_SECONDS: float = 180.0


class SoakSession:
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds_of_cell_type: dict[str, float],
        cell_fault_forms: CellFaultForms,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        namespace: str | None = None,
        release: str | None = None,
        event_log: EventLog | None = None,
        observer: SoakObserver | None = None,
        evidence_path: Path | None = None,
        quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
        policy: SoakPolicy | None = None,
        training_events_dir: Path | None = None,
        tail_policy: SoakTailPolicy | None = None,
        timeouts: SoakTimeouts | None = None,
    ) -> None:
        self.event_log = event_log if event_log is not None else EventLog()
        self.evidence_path = evidence_path
        self.timeouts = timeouts if timeouts is not None else SoakTimeouts()
        if evidence_path is not None:
            self.event_log.persist_to(evidence_path)
        self.cell_fault_forms = cell_fault_forms
        self._cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
        target_forms = {
            kind: [
                form.victim_form if isinstance(form, HookFaultForm) and form.victim_form is not None else form
                for form in cell_fault_forms[kind]
            ]
            for kind in self._cell_types
        }
        fault_target_types = {
            kind
            for kind, forms in target_forms.items()
            if any(form.name.startswith(("inject_fault:", "hook:")) for form in forms)
        }
        if any(
            isinstance(form, HookFaultForm) and form.victim_form is not None
            for kind in self._cell_types
            for form in cell_fault_forms[kind]
        ):
            fault_target_types.add("actor")
        self._runner = SoakRunner(
            observer=(
                replace(
                    observer,
                    cell_types=(self._cell_types | fault_target_types) - {"deployment"},
                    fault_target_cell_types=frozenset(fault_target_types),
                    process_patterns_of_type={
                        kind: {
                            container: pattern
                            for form in target_forms[kind]
                            if isinstance(form, ExecSigkillFaultForm)
                            for container, pattern in form.process_patterns.items()
                        }
                        for kind in self._cell_types
                    },
                )
                if observer is not None
                else SoakObserver(
                    base_url=base_url,
                    cell_types=self._cell_types | fault_target_types,
                    namespace=namespace,
                    release=release,
                    fault_target_cell_types=frozenset(fault_target_types),
                    process_patterns_of_type={
                        kind: {
                            container: pattern
                            for form in target_forms[kind]
                            if isinstance(form, ExecSigkillFaultForm)
                            for container, pattern in form.process_patterns.items()
                        }
                        for kind in self._cell_types
                    },
                )
            ),
            scheduler=SoakActionScheduler(
                rng=random.Random(seed),
                mean_intervals=mean_interval_seconds_of_cell_type,
                forms=cell_fault_forms,
                quiescent_polls_required=quiescent_polls_required,
                policy=policy,
            ),
            forms={kind: cell_fault_forms[kind] for kind in self._cell_types},
            event_log=self.event_log,
            poll_interval_seconds=poll_interval_seconds,
            training_events_dir=training_events_dir,
            tail_policy=tail_policy,
            timeouts=self.timeouts,
        )

    async def run(self, training: Coroutine[Any, Any, None], *, teardown: Callable[[], Awaitable[None]]) -> None:
        started = asyncio.Event()
        cleaning = asyncio.Event()
        owned = asyncio.create_task(
            self._run(training=training, teardown=teardown, started=started, cleaning=cleaning)
        )
        cancelled: asyncio.CancelledError | None = None
        forwarded = False
        while not owned.done():
            try:
                if cancelled is not None and not forwarded:
                    if not started.is_set():
                        await asyncio.sleep(0)
                        continue
                    if not cleaning.is_set():
                        owned.cancel()
                        forwarded = True
                await asyncio.shield(owned)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        try:
            owned.result()
        except BaseException as error:
            if cancelled is not None and not forwarded and not isinstance(error, asyncio.CancelledError):
                raise builtins.BaseExceptionGroup("Soak cancellation and cleanup failed", [cancelled, error]) from None
            raise
        if cancelled is not None:
            raise cancelled

    async def _run(
        self,
        *,
        training: Coroutine[Any, Any, None],
        teardown: Callable[[], Awaitable[None]],
        started: asyncio.Event,
        cleaning: asyncio.Event,
    ) -> None:
        started.set()
        stopped = asyncio.Event()
        errors: list[BaseException] = []
        try:
            async with asyncio.TaskGroup() as tasks:
                observing = tasks.create_task(self._runner.run(stopped))
                launched = tasks.create_task(training)
                try:
                    async with asyncio.timeout(self.timeouts.run_seconds):
                        await launched
                finally:
                    self.event_log.close_admission()
                    stopped.set()
                    async with asyncio.timeout(SHUTDOWN_TIMEOUT_SECONDS):
                        await observing
        except BaseException as error:
            errors.append(error)
        finally:
            cleaning.set()
            training.close()
            try:
                self.event_log.close_admission()
            except BaseException as error:
                errors.append(error)
            for cleanup in (self._runner.finish, teardown, self.event_log.finish):
                try:
                    await cleanup()
                except BaseException as error:
                    errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise builtins.BaseExceptionGroup("Soak session failed", errors)


def create_soak_session(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    cell_fault_forms: CellFaultForms,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    config: ExecuteTrainConfig | None = None,
    event_log: EventLog | None = None,
    observer: SoakObserver | None = None,
    evidence_path: Path | None = None,
    sources: dict[str, Path] | None = None,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
    policy: SoakPolicy | None = None,
    tail_policy: SoakTailPolicy | None = None,
    timeouts: SoakTimeouts | None = None,
) -> SoakSession:
    use_kubernetes = config is not None and config.cluster_backend is ClusterBackend.KUBERNETES
    handle = SoakSession(
        timeouts=timeouts,
        tail_policy=tail_policy,
        training_events_dir=(sources or {}).get("training_events"),
        event_log=event_log,
        observer=observer,
        evidence_path=evidence_path,
        policy=policy,
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
        poll_interval_seconds=poll_interval_seconds,
        namespace=config.namespace if use_kubernetes else None,
        release=(
            ReleaseName(
                run_id=config.run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
            ).serialize()
            if use_kubernetes
            else None
        ),
        quiescent_polls_required=quiescent_polls_required,
    )
    handle.event_log.note_context(
        SoakRunContextEvent(
            details={
                "base_url": base_url,
                "seed": seed,
                "tail_policy": tail_policy.model_dump(mode="json") if tail_policy is not None else None,
                "timeouts": handle.timeouts.model_dump(mode="json"),
                "mean_intervals": mean_interval_seconds_of_cell_type,
                "forms": {kind: [form.name for form in forms] for kind, forms in cell_fault_forms.items()},
                "config": asdict(config) if config is not None else None,
            },
            sources=sources or {},
        )
    )
    return handle
