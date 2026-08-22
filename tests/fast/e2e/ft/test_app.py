import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft import app as app_module
from tests.e2e.ft.conftest_ft.app import _DUMPS_ROOT_ENV, RunSideRequest, resolve_dump_dir, run_pipeline
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.workers.k8s_types import Pod, PodMetadata
from miles.utils.workers.types import ClusterBackend


def test_dump_dir_hangs_off_the_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster says where dumps go through the environment its infra file sets."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")

    assert resolve_dump_dir("scenario_x") == str(tmp_path / "dumps" / "run-a" / "scenario_x")


def test_an_empty_configured_root_is_not_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable and one set to nothing both mean the cluster configured no root."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, "")
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok: None)

    assert resolve_dump_dir("scenario_x") == "/node_public/dumps/run-a/scenario_x"


def test_two_runs_of_one_test_do_not_share_a_dump_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run id in the path is what stops one run's rmtree deleting another's dumps."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")
    first = resolve_dump_dir("scenario_x")
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-b")

    assert resolve_dump_dir("scenario_x") != first


def test_the_dump_directory_exists_when_it_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers write into the returned path without creating it themselves."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")

    assert Path(resolve_dump_dir("scenario_x")).is_dir()


def test_each_comparison_side_can_transform_its_config_before_the_context_and_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A side-specific release has to be shared by its target context and the launch it drives."""
    requests: list[RunSideRequest] = []
    contexts: list[command_utils.ExecuteTrainConfig] = []
    mode = FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )

    @contextlib.contextmanager
    def target_context(_mode: FTTestMode, _dump_dir: str, config: command_utils.ExecuteTrainConfig) -> Iterator[None]:
        contexts.append(config)
        yield

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)
    monkeypatch.setattr(
        command_utils, "default_config", lambda: command_utils.ExecuteTrainConfig(run_id="shared-release")
    )

    run_pipeline(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: None,
        phases=None,
        mode=None,
        target_side_context=target_context,
        config_for_side=lambda side, config: dataclasses.replace(config, run_id=f"{config.run_id}-{side}"),
        run_side=requests.append,
        resolve_mode_fn=lambda _mode: mode,
    )

    assert [request.config.run_id for request in requests] == ["shared-release-baseline", "shared-release-target"]
    assert len(contexts) == 1
    assert contexts[0] is requests[1].config


def test_each_comparison_side_releases_its_resources_before_the_next_side_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished baseline cannot overlap the target's GPU requests while its release tears down."""
    events: list[str] = []
    mode = FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    run_pipeline(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: events.append("compare"),
        phases=None,
        mode=None,
        run_side=lambda request: events.append(f"run:{request.side}"),
        release_side=lambda request: events.append(f"release:{request.side}"),
        resolve_mode_fn=lambda _mode: mode,
    )

    assert events == ["run:baseline", "release:baseline", "run:target", "release:target", "compare"]


def test_a_failed_comparison_side_is_released_without_starting_the_next_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A red verdict still releases its GPUs, but it must not continue into target or compare."""
    events: list[str] = []

    def fail_side(request: RunSideRequest) -> None:
        events.append(f"run:{request.side}")
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    with pytest.raises(RuntimeError, match="baseline failed"):
        run_pipeline(
            test_name="scenario_x",
            build_baseline_args=lambda *_args: "",
            build_target_args=lambda *_args: "",
            compare_fn=lambda *_args: events.append("compare"),
            phases=None,
            mode=None,
            run_side=fail_side,
            release_side=lambda request: events.append(f"release:{request.side}"),
            resolve_mode_fn=lambda _mode: _mode_fixture(),
        )

    assert events == ["run:baseline", "release:baseline"]


def test_a_failed_release_blocks_the_next_comparison_side(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A target cannot start while the baseline's resource ownership remains unresolved."""
    events: list[str] = []

    def fail_release(request: RunSideRequest) -> None:
        events.append(f"release:{request.side}")
        raise RuntimeError("release failed")

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    with pytest.raises(RuntimeError, match="release failed"):
        run_pipeline(
            test_name="scenario_x",
            build_baseline_args=lambda *_args: "",
            build_target_args=lambda *_args: "",
            compare_fn=lambda *_args: events.append("compare"),
            phases=None,
            mode=None,
            run_side=lambda request: events.append(f"run:{request.side}"),
            release_side=fail_release,
            resolve_mode_fn=lambda _mode: _mode_fixture(),
        )

    assert events == ["run:baseline", "release:baseline"]


def test_kubernetes_side_waits_for_its_pods_after_helm_no_longer_lists_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helm can forget a release before its terminating GPU pods stop reserving the node."""
    events: list[str] = []
    pod = Pod(metadata=PodMetadata(name="gpu-pod", uid="gpu-pod-uid"))
    pod_answers = iter([[pod], []])
    config = command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace="ci",
        run_id="run-baseline",
    )

    monkeypatch.setattr(
        app_module.Helm,
        "uninstall_if_present",
        lambda *, release, namespace: events.append(f"uninstall:{namespace}/{release}"),
    )
    monkeypatch.setattr(app_module.Helm, "get_manifest", lambda release, namespace: None)
    monkeypatch.setattr(
        app_module,
        "selected_pods",
        lambda namespace, selector: events.append(f"pods:{namespace}/{selector}") or next(pod_answers),
    )
    monkeypatch.setattr(app_module.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))

    app_module._release_comparison_side(_request(config))

    assert events == [
        "uninstall:ci/miles-run-run-baseline-all",
        "pods:ci/app.kubernetes.io/instance=miles-run-run-baseline-all",
        "sleep:1.0",
        "pods:ci/app.kubernetes.io/instance=miles-run-run-baseline-all",
    ]


def test_kubernetes_side_fails_after_a_bounded_wait_for_terminating_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck terminating pod fails the handoff instead of polling forever or starting the target."""
    pod = Pod(metadata=PodMetadata(name="stuck-gpu-pod", uid="stuck-gpu-pod-uid"))
    config = command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace="ci",
        run_id="run-baseline",
    )

    monkeypatch.setattr(app_module, "_RELEASE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(app_module.Helm, "uninstall_if_present", lambda **_kwargs: None)
    monkeypatch.setattr(app_module.Helm, "get_manifest", lambda release, namespace: None)
    monkeypatch.setattr(app_module, "selected_pods", lambda namespace, selector: [pod])

    with pytest.raises(TimeoutError, match="stuck-gpu-pod"):
        app_module._release_comparison_side(_request(config))


def test_ray_side_never_calls_kubernetes_release_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray owns no Helm release, so the comparison handoff has nothing cluster-side to remove."""
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    monkeypatch.setattr(
        app_module.Helm,
        "uninstall_if_present",
        lambda **_kwargs: pytest.fail("Ray comparison side touched Helm"),
    )

    app_module._release_comparison_side(_request(config))


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
