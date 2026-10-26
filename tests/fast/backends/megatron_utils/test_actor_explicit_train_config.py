from types import ModuleType

import pytest
from tests.fast.backends.megatron_utils.test_explicit_train_config import _reject_global_config
from tests.fast.backends.megatron_utils.test_model import TrainOneStepEnv, train_one_step_env
from tests.fast.backends.megatron_utils.test_shared_ppo_lifecycle import (
    _actor_worker,
    _critic_worker,
    _FakeRay,
    _noop_timer,
    _patch_shared_train_helpers,
    actor_module,
)

from miles.backends.megatron_utils import model as model_module

__all__ = ["actor_module", "train_one_step_env"]


class TestActorExplicitTrainConfig:
    @pytest.mark.parametrize("role", ["actor", "critic"])
    def test_both_roles_reach_real_training_validation_with_their_explicit_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        actor_module: ModuleType,
        train_one_step_env: TrainOneStepEnv,
        role: str,
    ) -> None:
        """Actor and critic forward their trainer config into real training geometry validation."""
        _patch_shared_train_helpers(actor_module, monkeypatch, _FakeRay())
        monkeypatch.setattr(actor_module, "train", model_module.train)
        monkeypatch.setattr(model_module, "get_args", _reject_global_config, raising=False)
        monkeypatch.setattr(actor_module, "forward_only", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)
        monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(actor_module, "log_train_advantage_computation_event", lambda *_args, **_kwargs: None)
        worker = _critic_worker(actor_module) if role == "critic" else _actor_worker(actor_module)
        data = {"tokens": [], "num_rollouts": []}

        with pytest.raises(AssertionError, match="must have the same length"):
            if role == "critic":
                worker._train_critic(rollout_id=7, rollout_data=data)
            else:
                worker._train_actor(rollout_id=7, rollout_data=data, external_data=None, witness_info=None, attempt=0)
