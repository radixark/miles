import dataclasses
import itertools
from pathlib import Path

from tests.e2e.ft.conftest_ft import scenario_random_crash, scenario_realistic_gsm8k

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

    def api_server_host(self) -> str:
        self._seen.asked_for_host.append(self.config)
        return f"orchestrator-of-{self.config.run_id}"

    def execute_train(self, **kwargs: object) -> None:
        self._seen.trained.append(self.config)


class _StubInjector:
    def __init__(self) -> None:
        self.num_successful_injections = 0
        self.recovery_witness = None

    def stop_and_join(self, *, timeout_seconds: float) -> None:
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
    def test_the_random_soak_builds_one_config_and_aims_every_step_at_it(self, monkeypatch, tmp_path: Path) -> None:
        """Regression: a second default_config gave the injector a run_id no release was ever installed under."""
        seen = _Seen()
        _install(monkeypatch, seen)
        monkeypatch.setattr(scenario_random_crash, "resolve_dump_dir", lambda test_name: str(tmp_path / "dump"))
        monkeypatch.setattr(scenario_random_crash, "prepare", lambda mode, *, config: seen.prepared.append(config))
        monkeypatch.setattr(
            scenario_random_crash, "materialize_cyclic_debug_rollout_data", lambda count: str(tmp_path / "rollout")
        )
        monkeypatch.setattr(scenario_random_crash, "get_common_train_args", lambda mode, **kwargs: "")
        monkeypatch.setattr(scenario_random_crash, "get_ft_args", lambda mode: "")
        monkeypatch.setattr(scenario_random_crash, "spawn_fault_injector", lambda **kwargs: _StubInjector())
        monkeypatch.setattr(scenario_random_crash, "assert_healing", lambda mode, **kwargs: None)

        scenario_random_crash.run_ci("kill_train__dp4_cp2__fake_rollout__moe_5layer", num_steps=1)

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert [config is seen.created[0] for config in seen.prepared] == [True]
        assert [config is seen.created[0] for config in seen.asked_for_host] == [True]
        assert [config is seen.created[0] for config in seen.trained] == [True]

    def test_the_gsm8k_soak_builds_one_config_and_aims_every_step_at_it(self, monkeypatch, tmp_path: Path) -> None:
        """The same bug here would point the injector at one release while training ran under another."""
        seen = _Seen()
        _install(monkeypatch, seen)
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.setenv(proxy_var, "http://unused")
        monkeypatch.setattr(
            scenario_realistic_gsm8k,
            "create_backend_for_run",
            lambda config: seen.prepared.append(config) or _RecordingBackend(config, seen),
        )
        monkeypatch.setattr(scenario_realistic_gsm8k, "_prepare_gsm8k", lambda U: None)
        monkeypatch.setattr(scenario_realistic_gsm8k, "resolve_dump_dir", lambda test_name: str(tmp_path / "gsm8k"))
        monkeypatch.setattr(scenario_realistic_gsm8k, "spawn_fault_injector", lambda **kwargs: _StubInjector())
        monkeypatch.setattr(
            scenario_realistic_gsm8k, "assert_soak_reconfigure_events", lambda event_dir, **kwargs: None
        )

        scenario_realistic_gsm8k.run_ci(num_rollout=1)

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert [config is seen.created[0] for config in seen.prepared] == [True]
        assert [config is seen.created[0] for config in seen.asked_for_host] == [True]
        assert [config is seen.created[0] for config in seen.trained] == [True]
