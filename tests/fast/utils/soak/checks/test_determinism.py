import pytest
from tests.utils.soak.checks.determinism import assert_deterministic_environment

from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent, EnvReportEvent, WeightUpdateResultEvent


class TestDeterministicEnvironment:
    @pytest.mark.parametrize("replacement_reported", [False, True])
    def test_replacement_requires_its_own_numerical_report(
        self, deterministic_environment: list[EnvReportEvent | EngineEnvReportEvent], replacement_reported: bool
    ) -> None:
        """An old incarnation's report cannot certify a replacement that receives training weights."""
        _, engine = deterministic_environment
        result = WeightUpdateResultEvent(
            timestamp=engine.timestamp,
            source=engine.source,
            update_id="replacement-update",
            rollout_id=2,
            candidate_version=3,
            published_version=3,
            target_incarnations={engine.cell_id: "replacement"},
            updated_cell_ids=[engine.cell_id],
            failed_cell_ids=[],
        )
        events = [*deterministic_environment, result]
        if replacement_reported:
            events.append(engine.model_copy(update={"workers_hash": "replacement"}))
        kwargs = dict(
            trainer_ranks={(0, 0)},
            engine_count=1,
            trainer_env={"NCCL_ALGO": "Ring"},
            engine_env={"SGLANG_ENABLE_JIT_DEEPGEMM": "false"},
        )
        if replacement_reported:
            assert_deterministic_environment(events, **kwargs)
        else:
            with pytest.raises(AssertionError, match="incarnations lack numerical reports"):
                assert_deterministic_environment(events, **kwargs)

    @pytest.mark.parametrize(
        "case",
        [
            "valid",
            "missing_trainer",
            "missing_engine",
            "trainer_env",
            "trainer_ft",
            "engine_env",
            "engine_args",
            "empty_workers",
            "identity",
        ],
    )
    def test_numeric_comparison_requires_actual_worker_configuration(
        self, deterministic_environment: list[EnvReportEvent | EngineEnvReportEvent], case: str
    ) -> None:
        """Missing reports and runtime settings that disagree with the recipe block numerical comparison."""
        trainer, engine = deterministic_environment
        if case == "missing_trainer":
            deterministic_environment.remove(trainer)
        elif case == "missing_engine":
            deterministic_environment.remove(engine)
        elif case == "trainer_env":
            trainer.report.process.env_vars["NCCL_ALGO"] = "Tree"
        elif case == "trainer_ft":
            trainer.report.process.args.values["ft_components"] = ["train", "rollout"]
        elif case == "engine_env":
            engine.server_info["internal_states"][0]["env_vars"] = {}
        elif case == "engine_args":
            engine.server_info["enable_deterministic_inference"] = False
        elif case == "empty_workers":
            engine.server_info["internal_states"] = []
        elif case == "identity":
            deterministic_environment[1] = engine.model_copy(update={"workers_hash": None})
        kwargs = dict(
            trainer_ranks={(0, 0)},
            engine_count=1,
            trainer_env={"NCCL_ALGO": "Ring"},
            engine_env={"SGLANG_ENABLE_JIT_DEEPGEMM": "false"},
        )
        if case == "valid":
            assert_deterministic_environment(deterministic_environment, **kwargs)
        else:
            with pytest.raises(AssertionError):
                assert_deterministic_environment(deterministic_environment, **kwargs)
