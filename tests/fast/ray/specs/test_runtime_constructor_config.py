import pytest

from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_inference_controller_config, make_rollout_config

from miles.ray.specs.inference import InferenceControllerSpec, InferenceRegistrationReporterSpec
from miles.ray.specs.rollout import RolloutExecutorSpec
from miles.ray.specs.train import TrainerControllerSpec
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.worker_spec import WorkerCtorContext


class TestRuntimeConstructorConfig:
    def test_inference_controller_uses_the_context_deployment_identity(self) -> None:
        """A controller rebuilt with a runtime context observes that context's registration hub."""
        original = make_inference_controller_config(make_args())
        runtime = original.model_copy(update={"deploy_component": "primary", "run_uuid": "runtime-run"})
        spec = InferenceControllerSpec.create(original)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())
        context = WorkerCtorContext(
            args=runtime,
            cell_index=0,
            worker_in_cell_index=0,
            num_workers_per_cell=1,
            gpu_ids=[],
            capability=capability,
        )

        kwargs = spec.ctor_kwargs(context)

        assert kwargs["args"] is runtime
        assert isinstance(kwargs["engine_provider"], RegistrationHub)
        assert kwargs["engine_provider"].run_uuid == "runtime-run"

    def test_rollout_session_lookup_follows_the_context_configuration(self) -> None:
        """A runtime session setting controls provider lookup instead of the specification's captured setting."""
        original = make_rollout_config(use_session_server=False)
        runtime = original.model_copy(update={"use_session_server": True})
        spec = RolloutExecutorSpec.create(original)
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(
            args=runtime,
            cell_index=0,
            worker_in_cell_index=0,
            num_workers_per_cell=1,
            gpu_ids=[],
            capability=capability,
        )

        kwargs = spec.ctor_kwargs(context)

        assert kwargs["args"] is runtime
        assert kwargs["session_server_provider"] is capability.static_provider
        assert "session-server" in capability.requested_static_pool_ids

    def test_trainer_controller_targets_the_context_trainer_pool(self) -> None:
        """A controller's runtime trainer identity chooses its cell provider and deployment identity."""
        config = parse_megatron_test_config()
        original = compute_trainer_config(config, config.raw_megatron.trainers[0])
        runtime = original.model_copy(update={"trainer_id": "runtime-actor", "run_uuid": "runtime-run"})
        spec = TrainerControllerSpec.create(original)
        capability = FakeBackendCapability(cells_provider=object(), cell_operations=object())
        context = WorkerCtorContext(
            args=runtime,
            cell_index=0,
            worker_in_cell_index=0,
            num_workers_per_cell=1,
            gpu_ids=[],
            capability=capability,
        )

        kwargs = spec.ctor_kwargs(context)

        assert kwargs["args"] is runtime
        assert kwargs["deployment_identity"].trainer_id == "runtime-actor"
        assert kwargs["deployment_identity"].run_uuid == "runtime-run"
        assert capability.requested_pool_ids == [["trainer-engine-runtime-actor"]]

    @pytest.mark.parametrize("pin_to_head", [False, True])
    def test_static_control_workers_follow_head_pinning(self, pin_to_head: bool) -> None:
        """Every static control worker respects both settings of the rollout head placement flag."""
        inference = make_inference_controller_config(make_args(pin_rollout_manager_to_head=pin_to_head))
        rollout = make_rollout_config(pin_rollout_manager_to_head=pin_to_head)
        specs = [
            InferenceControllerSpec.create(inference),
            InferenceRegistrationReporterSpec.create(inference),
            RolloutExecutorSpec.create(rollout),
        ]
        scaling = ScalingConfig.model_construct()

        assert [spec.scheduling(scaling).pin_to_head for spec in specs] == [pin_to_head] * 3
