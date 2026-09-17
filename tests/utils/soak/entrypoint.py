# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import random
from dataclasses import asdict, replace
from pathlib import Path

from tests.utils.soak.config import SoakPolicy, SoakTailPolicy, SoakTimeouts
from tests.utils.soak.core import POLL_INTERVAL_SECONDS, QUIESCENT_POLLS_REQUIRED, SoakActionScheduler
from tests.utils.soak.fault_forms import CellFaultForms
from tests.utils.soak.observer import SoakObserver
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import EventLog, SoakRunContextEvent

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend, DeployComponent

API_SERVER_PORT: int = 18080


def _create_soak_runner(
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
) -> SoakRunner:
    event_log = event_log if event_log is not None else EventLog()
    timeouts = timeouts if timeouts is not None else SoakTimeouts()
    if evidence_path is not None:
        event_log.persist_to(evidence_path)
    cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
    fault_target_types = {
        target_kind
        for kind in cell_types
        for form in cell_fault_forms[kind]
        for target_kind in form.fault_target_types(kind)
    }
    process_patterns = {
        kind: {
            container: pattern
            for form in cell_fault_forms[kind]
            for container, pattern in form.process_patterns.items()
        }
        for kind in cell_types
    }
    return SoakRunner(
        observer=(
            replace(
                observer,
                cell_types=(cell_types | fault_target_types) - {"deployment"},
                fault_target_cell_types=frozenset(fault_target_types),
                process_patterns_of_type=process_patterns,
            )
            if observer is not None
            else SoakObserver(
                base_url=base_url,
                cell_types=cell_types | fault_target_types,
                namespace=namespace,
                release=release,
                fault_target_cell_types=frozenset(fault_target_types),
                process_patterns_of_type=process_patterns,
            )
        ),
        scheduler=SoakActionScheduler(
            rng=random.Random(seed),
            mean_intervals=mean_interval_seconds_of_cell_type,
            forms=cell_fault_forms,
            quiescent_polls_required=quiescent_polls_required,
            policy=policy,
        ),
        forms={kind: cell_fault_forms[kind] for kind in cell_types},
        event_log=event_log,
        poll_interval_seconds=poll_interval_seconds,
        training_events_dir=training_events_dir,
        tail_policy=tail_policy,
        timeouts=timeouts,
    )


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
) -> SoakRunner:
    use_kubernetes = config is not None and config.cluster_backend is ClusterBackend.KUBERNETES
    handle = _create_soak_runner(
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
