from pathlib import Path

import pytest
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.ft import entrypoint as ft_entrypoint
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.observers import CellObserver

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import FailureMode
from miles.utils.workers.types import ClusterBackend


class TestRunCellSoak:
    async def test_only_the_configured_kinds_are_injected_and_observed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trainer-only soak hands the core runner trainer forms and a trainer observer on the run's api port."""
        recorded: dict[str, object] = {}

        async def run_soak(**kwargs: object) -> None:
            recorded.update(kwargs)

        monkeypatch.setattr(ft_entrypoint, "run_soak", run_soak)
        actor_form = InjectFaultForm(base_url="http://localhost:18080", failure_mode=FailureMode.EXIT)
        rollout_form = InjectFaultForm(base_url="http://localhost:18080", failure_mode=FailureMode.SIGKILL)
        monkeypatch.setattr(
            ft_entrypoint,
            "create_cell_fault_forms",
            lambda *, base_url, config: {"actor": [actor_form], "rollout": [rollout_form]},
        )
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id="260926-120000-000")
        runner_config = SoakRunnerConfig(
            seed=0, target_configs={"actor": SoakTargetConfig(expected_count=2, mean_interval_seconds=60.0)}
        )

        await run_cell_soak(
            config=config,
            dump_dir=tmp_path / "dump",
            sut_run=_noop(),
            runner_config=runner_config,
            event_log=EventLog(tmp_path / "events.jsonl"),
            evidence_dir=tmp_path / "evidence",
        )

        assert recorded["forms"] == {"actor": [actor_form]}
        assert recorded["runner_config"] is runner_config
        assert recorded["config"] is config
        observer = recorded["observer"]
        assert isinstance(observer, CellObserver)
        assert (observer.base_url, observer.cell_types, observer.release) == (
            "http://localhost:18080",
            {"actor"},
            None,
        )
        recorded["sut_run"].close()


async def _noop() -> None:
    return None
