import pytest
from tests.fast.backends.megatron_utils.test_model import TrainOneStepEnv, train_one_step_env

from miles.backends.megatron_utils import model as model_module
from miles.backends.training_utils.types import TrainStepOutcome

__all__ = ["train_one_step_env"]


def _reject_global_config() -> None:
    raise AssertionError("Training read the global Megatron configuration")


class TestExplicitTrainConfig:
    def test_training_geometry_validation_uses_the_supplied_config_without_sdk_globals(
        self, monkeypatch: pytest.MonkeyPatch, train_one_step_env: TrainOneStepEnv
    ) -> None:
        """Mismatched step geometry is rejected independently of Megatron's global arguments."""
        monkeypatch.setattr(model_module, "get_args", _reject_global_config, raising=False)
        env = train_one_step_env

        with pytest.raises(AssertionError, match="must have the same length"):
            model_module.train(
                args=env.args,
                rollout_id=2,
                model=env.model,
                optimizer=None,
                opt_param_scheduler=None,
                data_iterator=env.data_iterator,
                num_microbatches=[1],
                num_rollouts=[],
                witness_info=None,
                attempt=0,
            )

    def test_a_training_step_runs_with_the_supplied_config_without_sdk_globals(
        self, monkeypatch: pytest.MonkeyPatch, train_one_step_env: TrainOneStepEnv
    ) -> None:
        """The real step reaches the forward-backward engine without reading Megatron globals."""
        monkeypatch.setattr(model_module, "get_args", _reject_global_config, raising=False)
        env = train_one_step_env

        _, grad_norm, outcome = model_module.train_one_step(
            args=env.args,
            rollout_id=2,
            step_id=0,
            model=env.model,
            optimizer=None,
            opt_param_scheduler=None,
            data_iterator=env.data_iterator,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=0,
        )

        assert outcome is TrainStepOutcome.NORMAL
        assert grad_norm == 0.0
        assert len(env.forward_backward_engine.calls) == 1
        assert env.forward_backward_engine.calls[0]["forward_only"] is False
