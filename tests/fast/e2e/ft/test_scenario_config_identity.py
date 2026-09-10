import dataclasses
import itertools
import shlex
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.e2e.ft.conftest_ft import scenario_random_crash, scenario_realistic_gsm8k
from tests.utils.soak import state
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.recipes import gsm8k, gsm8k_launcher

from miles.utils.external_utils import command_utils


@dataclasses.dataclass
class _Seen:
    created: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    prepared: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    asked_for_host: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    trained: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)


class _RecordingBackend:
    def __init__(self, config: command_utils.ExecuteTrainConfig, seen: _Seen) -> None:
        self.config = config
        self._seen = seen

    def api_server_host(self, config: command_utils.ExecuteTrainConfig) -> str:
        self._seen.asked_for_host.append(config)
        return f"orchestrator-of-{config.run_id}"

    def execute_train(self, **kwargs: object) -> None:
        self._seen.trained.append(self.config)


class _StubInjector:
    def __init__(self) -> None:
        self.event_log = state.EventLog()

    def stop_and_join(self, *, teardown: Callable[[], None]) -> None:
        pass


def _install(monkeypatch, seen: _Seen) -> None:
    run_ids = itertools.count()

    def fake_default_config() -> command_utils.ExecuteTrainConfig:
        config = command_utils.ExecuteTrainConfig(run_id=f"sentinel-{next(run_ids)}", namespace="miles-e2e")
        seen.created.append(config)
        return config

    monkeypatch.setattr(command_utils, "default_config", fake_default_config)
    monkeypatch.setattr(
        command_utils.ExecuteTrainConfig, "create_backend", lambda self: _RecordingBackend(self, seen), raising=True
    )


class TestOneConfigPerSoak:
    @pytest.mark.parametrize("scenario", ["random", "all_gather", "all_targets"])
    def test_the_random_soak_builds_one_config_and_aims_every_step_at_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
    ) -> None:
        """Regression: a second default_config gave the injector a run_id no release was ever installed under."""
        seen = _Seen()
        _install(monkeypatch, seen)
        dump_run_ids: list[str] = []
        monkeypatch.setattr(
            scenario_random_crash,
            "resolve_dump_dir",
            lambda test_name, *, run_id: dump_run_ids.append(run_id) or str(tmp_path / "dump"),
        )
        monkeypatch.setattr(scenario_random_crash, "prepare", lambda mode, *, config: seen.prepared.append(config))
        monkeypatch.setattr(
            scenario_random_crash, "materialize_cyclic_debug_rollout_data", lambda count: str(tmp_path / "rollout")
        )
        monkeypatch.setattr(scenario_random_crash, "get_common_train_args", lambda mode, **kwargs: "")
        spawns: list[dict] = []
        launches: list[dict] = []
        hook_checks: list[str] = []
        monkeypatch.setattr(
            scenario_random_crash, "spawn_fault_injector", lambda **kwargs: spawns.append(kwargs) or _StubInjector()
        )
        monkeypatch.setattr(scenario_random_crash, "assert_healing", lambda ft_components, **kwargs: None)
        monkeypatch.setattr(scenario_random_crash, "assert_tail_complete", lambda events: None)
        monkeypatch.setattr(
            scenario_random_crash,
            "run_training",
            lambda **kwargs: launches.append(kwargs) or seen.trained.append(kwargs["config"]),
        )
        monkeypatch.setattr(scenario_random_crash, "read_events", lambda path: [])
        monkeypatch.setattr(
            scenario_random_crash, "assert_hook_effects", lambda events, **kwargs: hook_checks.append("effects")
        )
        monkeypatch.setattr(
            scenario_random_crash, "assert_hook_survivors", lambda events, **kwargs: hook_checks.append("survivors")
        )
        monkeypatch.setattr(
            scenario_random_crash, "assert_remote_p2p_failures", lambda events, **kwargs: hook_checks.append("p2p")
        )
        monkeypatch.setattr(
            scenario_random_crash,
            "assert_batch_trainers_recovered",
            lambda events, **kwargs: hook_checks.append("batch_recovery"),
        )

        scenario_random_crash.run_ci(
            {
                "random": "kill_train__dp4_cp2__fake_rollout__moe_5layer",
                "all_gather": "kill_train__dp2_tp2",
                "all_targets": "kill_rollout__dp2_tp2",
            }[scenario],
            num_steps=60,
            precise_all_gather=scenario == "all_gather",
            precise_p2p=scenario == "all_targets",
            all_p2p_targets=scenario == "all_targets",
            min_survivors=2 if scenario == "all_targets" else 1,
        )

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert dump_run_ids == ["sentinel-0"]
        run_config = seen.prepared[0]
        assert run_config.run_id == seen.created[0].run_id
        assert run_config.ray_submission_id
        assert [config is run_config for config in seen.prepared] == [True]
        assert [config is run_config for config in seen.asked_for_host] == [True]
        assert [config is run_config for config in seen.trained] == [True]
        assert len(spawns) == len(launches) == 1
        assert spawns[0]["config"] is run_config
        assert (
            hook_checks
            == {
                "random": [],
                "all_gather": ["effects", "survivors"],
                "all_targets": ["effects", "p2p", "batch_recovery"],
            }[scenario]
        )
        if scenario != "random":
            mode = launches[0]["mode"]
            assert mode.has_real_rollout and not mode.colocate
            assert mode.num_cells == 2
            parallel = shlex.split(mode.parallel_args)
            assert parallel[parallel.index("--tensor-model-parallel-size") + 1] == "2"
            argv = shlex.split(launches[0]["train_args"])
            for flag, value in {
                "--update-weight-transfer-mode": "p2p",
                "--train-step-timeout": "600",
                "--update-weights-timeout": "600",
            }.items():
                assert argv.count(flag) == 1
                assert argv[argv.index(flag) + 1] == value
            forms = spawns[0]["cell_fault_forms"]
            if scenario == "all_targets":
                assert set(forms) == {"rollout"}
                assert all(isinstance(form, HookFaultForm) and form.all_targets for form in forms["rollout"])
                assert argv[argv.index("--ft-components") + 1 : argv.index("--ft-components") + 3] == [
                    "train",
                    "rollout",
                ]
                assert set(spawns[0]["mean_interval_seconds_of_cell_type"]) == {"rollout"}
            else:
                assert set(forms) == {"actor"}
                assert all(isinstance(form, HookFaultForm) for form in forms["actor"])
                assert {form.name for form in forms["actor"]} == {
                    "hook:trainer_before_all_gather:sigkill:0ms",
                    "hook:trainer_before_all_gather:deadlock:0ms",
                    "hook:trainer_before_all_gather:thread_deadlock:0ms",
                }

    def test_the_gsm8k_soak_builds_one_config_and_aims_every_step_at_it(self, monkeypatch, tmp_path: Path) -> None:
        """The same bug here would point the injector at one release while training ran under another."""
        seen = _Seen()
        _install(monkeypatch, seen)
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.setenv(proxy_var, "http://unused")
        monkeypatch.setattr(
            gsm8k,
            "create_backend_for_run",
            lambda config: seen.prepared.append(config) or _RecordingBackend(config, seen),
        )
        monkeypatch.setattr(gsm8k, "prepare_gsm8k", lambda U: None)
        monkeypatch.setattr(gsm8k, "get_dumps_root", lambda: tmp_path)
        monkeypatch.setattr(
            gsm8k, "validate_dump_storage", lambda path: SimpleNamespace(model_dump_json=lambda **kwargs: "{}")
        )
        monkeypatch.setattr(gsm8k, "validate_training_storage", lambda args: None)
        monkeypatch.setattr(gsm8k, "assert_tail_complete", lambda events: None)
        monkeypatch.setattr(gsm8k, "assert_tail_quality", lambda events, **kwargs: None)
        monkeypatch.setattr(
            gsm8k_launcher, "execute_session", lambda *, run, injector, fully_async: run.launch(run.config)
        )
        dump_run_ids: list[str] = []
        monkeypatch.setattr(
            gsm8k,
            "resolve_dump_dir",
            lambda test_name, *, run_id: dump_run_ids.append(run_id) or str(tmp_path / "gsm8k"),
        )
        monkeypatch.setattr(gsm8k, "spawn_fault_injector", lambda **kwargs: _StubInjector())
        monkeypatch.setattr(scenario_realistic_gsm8k, "assert_healing", lambda ft_components, **kwargs: None)

        scenario_realistic_gsm8k.run_ci(num_rollout=100)

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert dump_run_ids == ["sentinel-0"]
        run_config = seen.prepared[0]
        assert run_config.run_id == seen.created[0].run_id
        assert run_config.ray_submission_id
        assert [config is run_config for config in seen.prepared] == [True, True]
        assert [config is run_config for config in seen.asked_for_host] == [True]
        assert [config is run_config for config in seen.trained] == [True]


class TestRelaunchingASoak:
    def test_a_relaunch_is_installed_under_the_config_it_was_handed(self, monkeypatch) -> None:
        """A hot restart relaunches the soak with --hot-restart added, and the backend reads that off its config."""
        seen = _Seen()
        _install(monkeypatch, seen)
        monkeypatch.setattr(gsm8k, "create_backend_for_run", lambda config: _RecordingBackend(config, seen))
        monkeypatch.setattr(gsm8k, "validate_training_storage", lambda args: None)
        relaunch = dataclasses.replace(command_utils.default_config(), hot_restart="orchestration")

        gsm8k.launch_gsm8k(relaunch, train_args="", fully_async=False)

        assert [config.hot_restart for config in seen.trained] == ["orchestration"]
