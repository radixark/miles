from pathlib import Path

import pytest
from tests.e2e.ft import test_random_crash_fully_async__kill_train_rollout__dp2_cp2 as fully_async_entry
from tests.e2e.ft.conftest_ft import scenario_random_crash, scenario_random_crash_fully_async
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode
from tests.fast.e2e.scenario_harness import SCENARIO_RUN_ID, ScenarioHarness, parse_fault_tolerance_args
from tests.utils.ft.launch import DETERMINISTIC_ENV_VARS, MEGATRON_PATH
from tests.utils.soak.core.config import SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.events import LaunchOutcome, SoakLaunchFinishedEvent
from tests.utils.soak.core.utils import API_SERVER_PORT
from tests.utils.soak.ft.actions.factory import create_cell_fault_forms
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.workers.types import ClusterBackend

_TRAIN_ONLY_MODE = "kill_train__dp2_cp2"
_ROLLOUT_ONLY_MODE = "kill_rollout__dp4"
_MIXED_MODE = "kill_train_rollout__dp2_cp2"
_FAKE_ROLLOUT_MODE = "kill_train__dp4_cp2__fake_rollout__moe_5layer"


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ScenarioHarness:
    monkeypatch.setattr(scenario_random_crash, "prepare", scenario_harness.record_prepare)
    monkeypatch.setattr(
        scenario_random_crash, "materialize_cyclic_debug_rollout_data", lambda count: str(tmp_path / f"cyclic-{count}")
    )
    monkeypatch.setattr(scenario_random_crash, "assert_healing", scenario_harness.recorder("assert_healing"))
    return scenario_harness


class TestTheSoakARandomCrashRunSchedules:
    def test_a_trainer_only_mode_soaks_only_actor_cells_at_the_trainer_cadence(self, harness: ScenarioHarness) -> None:
        """A rollout schedule in a trainer-only mode would draw targets the mode never made fault tolerant."""
        _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        runner_config = soak["runner_config"]
        assert runner_config.seed == 7
        assert runner_config.target_configs == {
            ACTOR_CELL_TYPE: SoakTargetConfig(expected_count=2, mean_interval_seconds=11.0)
        }
        assert runner_config.tail == SoakTailConfig.create(num_rollout=9)
        assert set(soak["forms"]) == {ACTOR_CELL_TYPE}

    def test_a_rollout_only_mode_expects_one_target_per_engine_at_the_rollout_cadence(
        self, harness: ScenarioHarness
    ) -> None:
        """The expected count is what the end-state check compares against, so it must be the engine count."""
        _run(_ROLLOUT_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        assert soak["runner_config"].target_configs == {
            ROLLOUT_CELL_TYPE: SoakTargetConfig(expected_count=4, mean_interval_seconds=13.0)
        }
        assert set(soak["forms"]) == {ROLLOUT_CELL_TYPE}

    def test_a_mixed_mode_keeps_each_kind_on_its_own_count_and_cadence(self, harness: ScenarioHarness) -> None:
        """Swapping the two cadences or counts would soak each kind at the rate meant for the other."""
        _run(_MIXED_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        assert soak["runner_config"].target_configs == {
            ACTOR_CELL_TYPE: SoakTargetConfig(expected_count=2, mean_interval_seconds=11.0),
            ROLLOUT_CELL_TYPE: SoakTargetConfig(expected_count=4, mean_interval_seconds=13.0),
        }

    def test_the_forms_are_the_factory_forms_for_this_backend(self, harness: ScenarioHarness) -> None:
        """Forms built for another backend would inject faults the run was never configured to survive."""
        _run(_MIXED_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        expected = create_cell_fault_forms(soak["config"])
        assert {kind: [form.name for form in forms] for kind, forms in soak["forms"].items()} == {
            kind: [form.name for form in expected[kind]] for kind in (ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE)
        }

    def test_the_observer_watches_the_api_server_the_run_is_launched_with(self, harness: ScenarioHarness) -> None:
        """An observer polling another port would see no cells, and the soak would never draw a target."""
        _run(_MIXED_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        (launch,) = harness.launches
        observer = soak["observer"]
        assert observer.base_url == f"http://localhost:{API_SERVER_PORT}"
        assert int(launch.value_of("--api-server-port")) == API_SERVER_PORT
        assert {ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE} <= observer.cell_types


class TestOneRunIdentityAcrossTheScenario:
    def test_prepare_the_launch_and_the_soak_share_one_ray_submission(self, harness: ScenarioHarness) -> None:
        """A second default config would mint another submission id, and teardown would stop a job nobody ran."""
        _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        (launch,) = harness.launches
        (prepared,) = harness.prepared
        assert launch.config is soak["config"] is prepared["config"]
        assert launch.config.cluster_backend is ClusterBackend.RAY
        assert launch.config.ray_submission_id.startswith("miles-soak-")

    def test_the_dumps_and_evidence_of_the_run_sit_under_its_run_id(self, harness: ScenarioHarness) -> None:
        """Evidence written beside another run's dumps would be archived as that run's proof."""
        _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        (launch,) = harness.launches
        dump_dir = harness.dumps_root / SCENARIO_RUN_ID / f"random_crash_{_TRAIN_ONLY_MODE}"
        assert soak["dump_dir"] == dump_dir
        assert soak["evidence_dir"].parent == dump_dir.with_name(f"{dump_dir.name}-soak")
        assert soak["event_log"].path == soak["evidence_dir"] / "events.jsonl"
        assert Path(launch.value_of("--save-debug-event-data")).parent == dump_dir


class TestTheLaunchedTrainArguments:
    @pytest.mark.parametrize("mode_name", sorted(MODES))
    def test_every_mode_launches_arguments_the_fault_tolerance_parser_gate_accepts(
        self, harness: ScenarioHarness, mode_name: str
    ) -> None:
        """A mode whose generated args fail the real partial-target gate cannot start, whatever the soak checks."""
        mode = MODES[mode_name]

        _run(mode_name, seed=7, num_steps=9)

        (launch,) = harness.launches
        parsed = parse_fault_tolerance_args(launch.request.train_args)
        assert parsed.ft_components == list(mode.ft_components)
        assert parsed.mini_ft_controller_enable
        assert not parsed.namespace.colocate
        assert "rollout" not in parsed.ft_components or parsed.partial_target_weight_update
        _assert_the_gpu_layout_is_the_modes(launch.request.num_gpus_per_node, parsed.namespace, mode=mode)

    def test_a_real_rollout_run_carries_the_p2p_update(self, harness: ScenarioHarness) -> None:
        """A rollout soak without the disaggregated P2P update would fail the partial-target gate at launch."""
        _run(_MIXED_MODE, seed=7, num_steps=9)

        (launch,) = harness.launches
        assert launch.value_of("--update-weight-transfer-mode") == "p2p"
        assert launch.value_of("--num-rollout") == "9"
        assert launch.request.train_script.endswith("/train.py")

    def test_the_launch_carries_the_shared_eager_deterministic_environment(self, harness: ScenarioHarness) -> None:
        """A respawned cell recompiling under torch.compile can OOM, so every soak launch must run eager."""
        _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (launch,) = harness.launches
        env = launch.request.extra_env_vars
        assert DETERMINISTIC_ENV_VARS.items() <= env.items()
        assert (env["TORCHDYNAMO_DISABLE"], env["RAY_DEDUP_LOGS"]) == ("1", "0")
        assert launch.request.megatron_path == MEGATRON_PATH
        assert launch.request.megatron_model_type == MODES[_TRAIN_ONLY_MODE].megatron_model_type

    def test_a_fake_rollout_run_trains_off_the_materialized_cyclic_data_without_hooks(
        self, harness: ScenarioHarness, tmp_path: Path
    ) -> None:
        """Without engines there is no weight update to send, and the data must cover every one of the steps."""
        _run(_FAKE_ROLLOUT_MODE, seed=7, num_steps=9)

        (launch,) = harness.launches
        (soak,) = harness.soaks
        assert launch.value_of("--load-debug-rollout-data") == f"{tmp_path / 'cyclic-9'}/{{rollout_id}}.pt"
        assert "--update-weight-transfer-mode" not in launch.argv
        expected = create_cell_fault_forms(soak["config"])
        assert [form.name for form in soak["forms"][ACTOR_CELL_TYPE]] == [
            form.name for form in expected[ACTOR_CELL_TYPE]
        ]

    def test_a_fully_async_run_launches_the_async_driver_under_a_name_of_its_own(
        self, harness: ScenarioHarness
    ) -> None:
        """The sync driver ignores --fully-async, so the soak would test a mode that never ran."""
        _run(_MIXED_MODE, seed=7, num_steps=9, fully_async=True)

        (launch,) = harness.launches
        (soak,) = harness.soaks
        assert launch.request.train_script.endswith("/train_async.py")
        assert "--fully-async" in launch.argv
        assert launch.value_of("--pause-generation-mode") == "in_place"
        assert soak["dump_dir"].name == f"random_crash_fully_async_{_MIXED_MODE}"

    def test_the_fully_async_entry_soaks_trainers_and_real_engines_through_the_async_driver(
        self, harness: ScenarioHarness
    ) -> None:
        """Killing an engine that keeps generating across updates is the fault only this entry can produce."""
        scenario_random_crash_fully_async.run_ci(fully_async_entry._MODE)

        (launch,) = harness.launches
        (soak,) = harness.soaks
        assert launch.request.train_script.endswith("/train_async.py")
        assert set(soak["runner_config"].target_configs) == {ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE}
        assert MODES[fully_async_entry._MODE].has_real_rollout

    def test_a_fully_async_fake_rollout_run_is_refused_before_anything_is_prepared(
        self, harness: ScenarioHarness
    ) -> None:
        """Training off recorded data proves nothing about generating while training, so it must not start."""
        with pytest.raises(AssertionError, match="fully-async soak"):
            _run(_FAKE_ROLLOUT_MODE, seed=7, num_steps=9, fully_async=True)

        assert harness.prepared == []
        assert harness.launches == []


class TestWhatTheSoakIsJudgedBy:
    def test_the_checkers_read_the_events_and_forms_of_this_soak(self, harness: ScenarioHarness) -> None:
        """Checking another event log or form set would pass a soak on evidence it never produced."""
        _run(_MIXED_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        events = soak["event_log"].events
        assert harness.checker_names == ["assert_healing"]
        ((healing_args, healing_kwargs),) = harness.calls_of("assert_healing")
        assert healing_args == (("train", "rollout"),)
        assert healing_kwargs["events"] == events
        assert healing_kwargs["forms"] is soak["forms"]
        assert healing_kwargs["context"] == f"random_crash {_MIXED_MODE}"

    def test_the_launch_outcome_is_recorded_in_the_soak_event_log(self, harness: ScenarioHarness) -> None:
        """The runner decides the soak ended from this event, so a launch that finished must be written there."""
        _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        (event,) = soak["event_log"].events
        assert isinstance(event, SoakLaunchFinishedEvent)
        assert (event.request_id, event.outcome, event.error) == (None, LaunchOutcome.FINISHED, None)

    def test_a_failed_launch_is_raised_recorded_and_never_graded(self, harness: ScenarioHarness) -> None:
        """A failed run must fail the soak rather than be graded as though it had trained."""
        harness.launch_error = RuntimeError("the job exited 1")

        with pytest.raises(RuntimeError, match="the job exited 1"):
            _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)

        (soak,) = harness.soaks
        (event,) = soak["event_log"].events
        assert event.outcome is LaunchOutcome.FAILED
        assert harness.checks == []

    def test_a_soak_that_injected_nothing_fails_the_real_healing_check(
        self, scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run nothing was injected into proves no recovery, so the scenario must fail and not pass."""
        monkeypatch.setattr(scenario_random_crash, "prepare", scenario_harness.record_prepare)

        with pytest.raises(AssertionError, match="Soak proved too little"):
            _run(_TRAIN_ONLY_MODE, seed=7, num_steps=9)


def _run(
    mode: str,
    *,
    seed: int,
    num_steps: int,
    fully_async: bool = False,
) -> None:
    scenario_random_crash.run_ci(
        mode=mode,
        seed=seed,
        num_steps=num_steps,
        trainer_crash_interval_seconds=11.0,
        rollout_crash_interval_seconds=13.0,
        fully_async=fully_async,
    )


def _assert_the_gpu_layout_is_the_modes(num_gpus_per_node: int, namespace: object, *, mode: FTTestMode) -> None:
    assert num_gpus_per_node == mode.train_gpus_per_node + mode.rollout_num_engines * mode.rollout_gpus_per_engine
    assert namespace.actor_num_gpus_per_node == mode.train_gpus_per_node
    assert namespace.actor_num_nodes == mode.train_num_nodes
    if mode.has_real_rollout:
        assert namespace.rollout_num_gpus == mode.total_rollout_gpus
        assert namespace.rollout_num_gpus_per_engine == mode.rollout_gpus_per_engine
