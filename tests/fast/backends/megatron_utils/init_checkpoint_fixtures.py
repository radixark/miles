from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.megatron_utils.init_system_fixtures import _install_process_environment, _TorchftGroup
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from typing_extensions import Self

from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.init_once import InitOnce


@dataclass
class _InitSdk:
    args: TrainerConfig
    iteration: int
    failure: str | None
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    peer_sources: list[int] = field(default_factory=list)
    checkpointing_context: dict[str, Any] | None = None

    def get_model(self, provider: object, model_type: object) -> list[torch.nn.Module]:
        self.observations["build"] = self._runtime_fields()
        self.args.backend.model_type = model_type
        if self.failure == "build":
            raise RuntimeError("model SDK failed")
        return [torch.nn.Linear(1, 1)]

    def load_checkpoint(self, **kwargs: Any) -> tuple[int, int]:
        self.observations["load"] = self._runtime_fields()
        self.checkpointing_context = kwargs["checkpointing_context"]
        if self.peer_sources:
            assert self.args.backend.pretrained_checkpoint is None
        self.args.backend.consumed_train_samples = 64
        if self.failure == "load":
            raise RuntimeError("checkpoint SDK failed")
        return self.iteration, 0

    def create_transport(self, *, pg: _TorchftGroup, timeout: timedelta, device: torch.device) -> Self:
        assert pg.configuration["rank"] == 1
        assert pg.configuration["world_size"] == 2
        return self

    def metadata(self) -> dict[str, Any]:
        return {}

    def recv_checkpoint(self, *, src_rank: int, metadata: object, step: int, timeout: timedelta) -> dict[str, object]:
        from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

        from miles.backends.megatron_utils.ft.checkpoint_transfer import _TransportCodec

        self.peer_sources.append(src_rank)
        state = MCoreTensorAwareStateDict(common={"iteration": self.iteration}, sharded_state_dict={})
        return _TransportCodec.encode(state_dict=state, iteration=self.iteration)

    def _runtime_fields(self) -> dict[str, Any]:
        return {name: vars(self.args.backend)[name] for name in CHECKPOINT_LOAD_FIELDS}


def _init_worker(
    *,
    actor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    role: str,
    iteration: int,
    failure: str | None,
    peer: bool,
) -> tuple[Any, _InitSdk]:
    from miles.backends.megatron_utils import checkpoint, model
    from miles.backends.megatron_utils.ft import checkpoint_transfer, in_memory_checkpoint

    path.mkdir()
    (path / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    flags = ["--debug-train-only", "--debug-disable-optimizer", "--megatron-to-hf-mode", "raw"]
    if role == "critic":
        flags.extend(["--advantage-estimator", "ppo"])
    if peer:
        flags.extend(
            [
                "--pretrained-checkpoint",
                str(path / "startup-fallback"),
                "--indep-dp",
                "--use-dynamic-batch-size",
                "--max-tokens-per-gpu",
                "1024",
                "--non-persistent-ckpt-type",
                "local",
                "--non-persistent-local-ckpt-dir",
                str(path / "local"),
                "--non-persistent-local-ckpt-algo",
                "atomic",
            ]
        )
    config = parse_megatron_test_config(*flags)
    trainer = next(item for item in config.raw_megatron.trainers if item.role == role)
    args = compute_trainer_config(config, trainer).model_copy(
        update={"clear_quantized_weight_workspaces_on_offload": False}
    )
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker._init_once = InitOnce("MegatronTrainRayActor")
    worker._heartbeat = Mock()
    worker._rank = 0
    sdk = _InitSdk(args=args, iteration=iteration, failure=failure)
    _install_process_environment(actor_module=actor_module, args=args, monkeypatch=monkeypatch)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "get_model", sdk.get_model)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", sdk.load_checkpoint)
    monkeypatch.setattr(checkpoint_transfer, "PGTransport", sdk.create_transport)
    monkeypatch.setattr(in_memory_checkpoint, "get_args", lambda: args.backend)
    return worker, sdk
