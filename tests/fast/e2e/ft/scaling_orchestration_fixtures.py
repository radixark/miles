from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import scaling
from tests.fast.utils.soak.soak_fakes import _at, _write_sut_lines
from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakActionRequestedEvent, SoakEvent
from tests.utils.soak.core.types import SoakActionRequest
from tests.utils.soak.ft import entrypoint
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails

from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


def install_cluster_evidence(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str | None
) -> Callable[[list[SoakEvent], Path], dict[int, int]]:
    config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="test", run_id="same")
    dump_dir = tmp_path / "dumps"
    monkeypatch.setattr(scaling.command_utils, "default_config", lambda: config)
    monkeypatch.setattr(scaling, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(scaling, "resolve_dump_dir", lambda *args, **kwargs: str(dump_dir))
    monkeypatch.setattr(scaling, "run_training", lambda **kwargs: None)

    async def run_soak(**kwargs: Any) -> SimpleNamespace:
        await kwargs["sut_run"]
        log = kwargs["event_log"]
        for index, step in enumerate(scaling.SCALING_SCHEDULE):
            if corruption == "missing_resize" and index == 1:
                continue
            request = SoakActionRequest(
                target=PoolTarget(identity="actor-pool", alive=True, ready=True, replicas=2 if index == 0 else 3),
                form_name="resize:actor",
                details=ResizeDetails(step=step),
            )
            log.append(SoakActionRequestedEvent(request=request))
            log.append(SoakActionAppliedEvent(request_id=request.request_id, evidence=PoolResizedEvidence()))
        (dump_dir / "events").mkdir()
        _write_sut_lines(
            dump_dir / "events/controller.jsonl",
            [
                MetricEvent(
                    timestamp=_at(i),
                    source=TrainerControllerProcessIdentity(trainer_id="actor"),
                    rollout_id=i,
                    attempt=0,
                    metrics={"train/grad_norm": 0.0 if corruption == "zero_gradient" and i == 0 else 1.0},
                )
                for i in range(scaling.SCALING_NUM_ROLLOUTS)
            ],
        )
        return SimpleNamespace(event_log=log)

    monkeypatch.setattr(entrypoint, "run_soak", run_soak)

    def counts(events: list[SoakEvent], path: Path) -> dict[int, int]:
        assert path == dump_dir
        return {i: 2 if corruption == "counts" or i < 2 or i >= 7 else 3 for i in range(12)}

    return counts
