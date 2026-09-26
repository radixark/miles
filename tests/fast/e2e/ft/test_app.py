import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import app as app_module
from tests.e2e.ft.conftest_ft.app import RunSideRequest, run_pipeline
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from typer.testing import CliRunner

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


@pytest.fixture
def pipeline_dump_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    dump_dir = tmp_path / "comparison"
    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name, *, run_id: str(dump_dir))
    monkeypatch.setattr(app_module, "prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        command_utils, "default_config", lambda: command_utils.ExecuteTrainConfig(run_id="shared-release")
    )
    return dump_dir


class TestRunPipeline:
    def test_each_side_transforms_its_config_before_the_context_and_launch(self, pipeline_dump_dir: Path) -> None:
        """A side-specific release has to be shared by its target context and the launch it drives."""
        requests: list[RunSideRequest] = []
        contexts: list[command_utils.ExecuteTrainConfig] = []

        @contextlib.contextmanager
        def target_context(
            _mode: FTTestMode, _dump_dir: str, config: command_utils.ExecuteTrainConfig
        ) -> Iterator[None]:
            contexts.append(config)
            yield

        _run_pipeline(
            target_side_context=target_context,
            config_for_side=lambda side, config: dataclasses.replace(config, run_id=f"{config.run_id}-{side}"),
            run_side=requests.append,
        )

        assert [request.config.run_id for request in requests] == ["shared-release-baseline", "shared-release-target"]
        assert [request.dump_dir for request in requests] == [
            f"{pipeline_dump_dir}/baseline",
            f"{pipeline_dump_dir}/target",
        ]
        assert len(contexts) == 1
        assert contexts[0] is requests[1].config

    def test_each_side_is_released_before_the_next_starts_and_compared_last(self, pipeline_dump_dir: Path) -> None:
        """A finished baseline cannot overlap the target's GPU requests while its release tears down."""
        events: list[str] = []

        _run_pipeline(
            compare_fn=lambda *_args: events.append("compare"),
            run_side=lambda request: events.append(f"run:{request.side}"),
            release_side=lambda request: events.append(f"release:{request.side}"),
        )

        assert events == ["run:baseline", "release:baseline", "run:target", "release:target", "compare"]

    def test_a_failed_side_is_released_without_starting_the_next_side(self, pipeline_dump_dir: Path) -> None:
        """A red verdict still releases its GPUs, but it must not continue into target or compare."""
        events: list[str] = []

        def fail_side(request: RunSideRequest) -> None:
            events.append(f"run:{request.side}")
            raise RuntimeError("baseline failed")

        with pytest.raises(RuntimeError, match="baseline failed"):
            _run_pipeline(
                compare_fn=lambda *_args: events.append("compare"),
                run_side=fail_side,
                release_side=lambda request: events.append(f"release:{request.side}"),
            )

        assert events == ["run:baseline", "release:baseline"]

    def test_a_failed_release_blocks_the_next_side(self, pipeline_dump_dir: Path) -> None:
        """A target cannot start while the baseline's resource ownership remains unresolved."""
        events: list[str] = []

        def fail_release(request: RunSideRequest) -> None:
            events.append(f"release:{request.side}")
            raise RuntimeError("release failed")

        with pytest.raises(RuntimeError, match="release failed"):
            _run_pipeline(
                compare_fn=lambda *_args: events.append("compare"),
                run_side=lambda request: events.append(f"run:{request.side}"),
                release_side=fail_release,
            )

        assert events == ["run:baseline", "release:baseline"]


class TestReleaseComparisonSide:
    def test_a_kubernetes_side_removes_the_release_of_its_own_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Removing another release would leave this side's GPUs reserved when the next side starts."""
        removed: list[dict[str, str]] = []
        monkeypatch.setattr(app_module, "remove_release_and_wait", lambda **kwargs: removed.append(kwargs))
        config = command_utils.ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace="ci", run_id="run-baseline"
        )

        app_module._release_comparison_side(_request(config))

        assert removed == [{"release": "miles-run-run-baseline-all", "namespace": "ci"}]

    def test_a_kubernetes_side_without_a_namespace_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Helm would fall back to the default namespace and remove a release this run never installed."""
        monkeypatch.setattr(
            app_module,
            "remove_release_and_wait",
            lambda **_kwargs: pytest.fail("removed a release without a namespace"),
        )
        config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, run_id="run-baseline")

        with pytest.raises(AssertionError, match="needs a namespace"):
            app_module._release_comparison_side(_request(config))

    def test_a_ray_side_never_calls_kubernetes_release_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ray owns no Helm release, so the comparison handoff has nothing cluster-side to remove."""
        monkeypatch.setattr(
            app_module, "remove_release_and_wait", lambda **_kwargs: pytest.fail("a ray side touched helm")
        )

        app_module._release_comparison_side(
            _request(command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY))
        )


class TestGenerateData:
    def test_a_fixed_topology_generate_data_runs_without_a_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scenario whose topology is fixed asks its user for no --mode on any of its subcommands."""
        launched: list[dict[str, Any]] = []
        mode = dataclasses.replace(_mode_fixture(), rollout_num_engines=1, rollout_gpus_per_engine=1)
        monkeypatch.setattr(app_module, "prepare", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(app_module, "get_common_train_args", lambda *_args, **_kwargs: "--num-rollout 1")
        monkeypatch.setattr(app_module, "resolve_dump_dir", lambda test_name, *, run_id: f"/dumps/{test_name}")
        monkeypatch.setattr(app_module, "run_training", lambda **kwargs: launched.append(kwargs))
        app, _ = app_module.create_comparison_app_and_run_ci(
            test_name="scenario_x",
            build_baseline_args=lambda *_args: "",
            build_target_args=lambda *_args: "",
            compare_fn=lambda *_args: None,
            resolve_mode_fn=lambda _mode: mode,
        )

        result = CliRunner().invoke(app, ["generate-data", "--num-steps", "1"])

        assert result.exit_code == 0, result.output
        assert [(one["train_args"], one["mode"]) for one in launched] == [("--num-rollout 1", mode)]


def _run_pipeline(**overrides: Any) -> None:
    run_pipeline(
        **{
            "test_name": "scenario_x",
            "build_baseline_args": lambda *_args: "",
            "build_target_args": lambda *_args: "",
            "compare_fn": lambda *_args: None,
            "phases": None,
            "mode": None,
            "run_side": lambda request: None,
            "release_side": lambda request: None,
            "resolve_mode_fn": lambda _mode: _mode_fixture(),
            **overrides,
        }
    )


def _mode_fixture() -> FTTestMode:
    return FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )


def _request(config: command_utils.ExecuteTrainConfig) -> RunSideRequest:
    return RunSideRequest(
        side="baseline",
        mode=_mode_fixture(),
        train_args="",
        dump_dir="/dumps/baseline",
        config=config,
        enable_dumper=True,
    )
