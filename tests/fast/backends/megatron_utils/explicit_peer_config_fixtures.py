import importlib
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from tests.fast.backends.megatron_utils.test_checkpoint_transfer import _FakeTransport


@dataclass
class PeerCheckpointBoundary:
    actor_module: ModuleType
    model_module: ModuleType
    checkpoint_module: ModuleType
    transfer_module: ModuleType
    transport: _FakeTransport
    save_calls: list[dict[str, Any]]
    load_calls: list[dict[str, Any]]
    checkpoint_calls: list[dict[str, Any]]


@pytest.fixture
def peer_checkpoint_boundary(
    actor_module: ModuleType, state_dict: MCoreTensorAwareStateDict, monkeypatch: pytest.MonkeyPatch
) -> PeerCheckpointBoundary:
    model_module = importlib.import_module("miles.backends.megatron_utils.model")
    checkpoint_module = importlib.import_module("miles.backends.megatron_utils.checkpoint")
    transfer_module = importlib.import_module("miles.backends.megatron_utils.ft.checkpoint_transfer")
    memory_module = importlib.import_module("miles.backends.megatron_utils.ft.in_memory_checkpoint")
    boundary = PeerCheckpointBoundary(
        actor_module=actor_module,
        model_module=model_module,
        checkpoint_module=checkpoint_module,
        transfer_module=transfer_module,
        transport=_FakeTransport(),
        save_calls=[],
        load_calls=[],
        checkpoint_calls=[],
    )

    def save_checkpoint(
        iteration: int, model: list[Any], optimizer: object, opt_param_scheduler: object, **kwargs: Any
    ) -> None:
        boundary.save_calls.append(
            dict(
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                **kwargs
            )
        )
        kwargs["checkpointing_context"]["local_checkpoint_manager"].save(state_dict=state_dict, iteration=iteration)

    def load_checkpoint(**kwargs: Any) -> tuple[int, int]:
        boundary.load_calls.append(kwargs)
        context = kwargs["checkpointing_context"]
        if "local_checkpoint_manager" in context:
            manager = context["local_checkpoint_manager"]
            return manager.find_latest(), 0
        return 0, 0

    def observe_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
        trainer = kwargs["args"]
        boundary.checkpoint_calls.append(
            dict(
                args=trainer,
                load=trainer.backend.load,
                ckpt_step=trainer.backend.ckpt_step,
                local_checkpoint_manager="local_checkpoint_manager" in (kwargs["checkpointing_context"] or {}),
                backend=dict(vars(trainer.backend)),
            )
        )
        return checkpoint_module.load_checkpoint(*args, **kwargs)

    monkeypatch.setattr(model_module, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", load_checkpoint)
    monkeypatch.setattr(model_module, "load_checkpoint", observe_checkpoint_load)
    monkeypatch.setattr(actor_module, "load_checkpoint", observe_checkpoint_load)
    monkeypatch.setattr(
        memory_module,
        "get_args",
        lambda: SimpleNamespace(non_persistent_ckpt_type="local", non_persistent_local_ckpt_algo="fully_parallel"),
    )
    monkeypatch.setattr(transfer_module, "_create_transport", lambda indep_dp, timeout: boundary.transport)
    monkeypatch.setattr(model_module, "clear_memory", lambda: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda args: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda args, model, iteration: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda: None)
    return boundary


def _persistent_trainer_args(**values: Any) -> Any:
    from tests.fast.fixtures.args_fixtures import make_trainer_args

    from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS

    args = make_trainer_args(**{"spec": None, **values})
    args.backend = type(args.backend)(
        **{
            name: value
            for name, value in vars(args.backend).items()
            if name not in CHECKPOINT_LOAD_FIELDS | {"requested_load"}
        }
    )
    return args
