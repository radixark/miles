from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import comparisons


@pytest.fixture
def comparison_primitives_but_reconfigure(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []
    for name in (
        "assert_metrics_classified",
        "compare_metrics",
        "compare_dumps",
        "compare_inference_engine_checksums",
        "assert_engine_weights_moved",
        "assert_gradients_nonzero",
    ):
        monkeypatch.setattr(comparisons, name, lambda *args, _name=name, **kwargs: called.append(_name))
    return called


@pytest.fixture
def scaling_launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    from types import SimpleNamespace

    from tests.e2e.ft.conftest_ft import scaling
    from tests.utils.soak.core.utils import REPLACED_LAUNCH_EXIT_CODE
    from tests.utils.soak.deploy.session import start_launch

    from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError
    from miles.utils.workers.types import ClusterBackend

    config = SimpleNamespace(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace="test",
        run_id="test",
        deploy_component="all",
        deploy_instance_id="default",
    )
    monkeypatch.setattr(scaling.command_utils, "default_config", lambda: config)
    monkeypatch.setattr(scaling, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(scaling, "resolve_dump_dir", lambda *args, **kwargs: str(tmp_path / "dumps"))
    monkeypatch.setattr(scaling, "assert_resizes_follow_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(scaling, "assert_gradients_nonzero", lambda *args, **kwargs: None)
    monkeypatch.setattr(scaling, "assert_counts_follow_schedule", lambda *args, **kwargs: None)

    def launch(**kwargs: Any) -> None:
        raise RunExitedError(exit_code=REPLACED_LAUNCH_EXIT_CODE)

    async def finished() -> None:
        return None

    async def run_soak(*, sut_run: Any, event_log: Any, forms: Any, **kwargs: Any) -> SimpleNamespace:
        [form] = next(iter(forms.values()))
        start_launch(finished(), event_log=event_log, request_id="resize", chain=form.chain)
        await sut_run
        return SimpleNamespace(event_log=event_log)

    monkeypatch.setattr(scaling, "run_training", launch)
    monkeypatch.setattr(scaling, "run_cell_soak", run_soak)
    return scaling
