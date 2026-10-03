from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils import checkpoint, model
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig
from miles.ray.specs.train import TrainerControllerSpec
from miles.ray.train.init_request import TrainerControllerInitRequest
from miles.utils.args.runtime import OrchestratorConfig, TrainerConfig
from miles.utils.multi_policy import utils as multi_policy_utils
from miles.utils.workers.backend_capability.base import BackendCapability


@dataclass
class _LoadingTrainer:
    args: TrainerConfig

    async def is_initialized(self) -> bool:
        return False

    async def init(self, request: TrainerControllerInitRequest) -> list[int]:
        self.args.num_rollout = request.num_rollout
        self.args.wandb_run_id = request.wandb_run_id
        self.args.mlflow_run_id = request.mlflow_run_id
        assert request.checkpoint_load is not None
        with request.checkpoint_load.apply(self.args.backend):
            result = model.load_model_state(
                self.args,
                model=[],
                optimizer=None,
                opt_param_scheduler=None,
                role="actor",
                checkpointing_context=None,
            )
        return [result.start_rollout_id]

    async def get_train_parallel_config(self) -> None:
        return None


@dataclass
class _Rollout:
    restored_rollouts: list[int] = field(default_factory=list)

    async def set_train_parallel_config(self, config: Any, *, trainer_model_id: str) -> None:
        pass

    async def load(self, rollout_id: int, *, load: str | None) -> None:
        self.restored_rollouts.append(rollout_id)


@dataclass
class _FreshPolicyStartup:
    args: OrchestratorConfig
    handles: dict[str, _LoadingTrainer]
    rollout: _Rollout


@pytest.fixture
def fresh_policy_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FreshPolicyStartup:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    all_args = parse_megatron_test_config(
        "--megatron-config",
        encode_megatron_config("solver", "verifier"),
        "--megatron-to-hf-mode",
        "bridge",
        "--ref-load",
        str(weights),
        "--save",
        str(tmp_path / "checkpoints"),
        "--save-interval",
        "2",
    )
    handles = {args.trainer_id: _LoadingTrainer(args=args) for args in TrainerControllerSpec.slice_configs(all_args)}

    def create_handles(
        args: OrchestratorConfig,
        *,
        trainer_configs: list[MegatronTrainerConfig],
        capability: BackendCapability,
    ) -> dict[str, _LoadingTrainer]:
        return {config.trainer_id: handles[config.trainer_id] for config in trainer_configs}

    monkeypatch.setattr(multi_policy_utils, "create_trainer_handles", create_handles)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **kwargs: (0, 0))
    monkeypatch.setattr(model, "clear_memory", lambda: None)
    monkeypatch.setattr(model, "check_peak_gpu_memory_after_load", lambda args: None)
    monkeypatch.setattr(model, "check_model_hashes", lambda *args: None)

    return _FreshPolicyStartup(args=OrchestratorConfig.slice_from(all_args), handles=handles, rollout=_Rollout())
