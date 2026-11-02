from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from tests.fast.backends.megatron_utils.backend_scope_fixtures import _parsed_trainer
from tests.fast.backends.megatron_utils.test_actor_reload import _actor, _watch_load, _write_checkpoint


def _auxiliary_worker(*, actor_module: ModuleType) -> Any:
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = _parsed_trainer("--ref-ckpt-step", "3", "--opd-teacher-ckpt-step", "9")
    worker.model = []
    worker.weights_backuper = Mock()
    return worker


def _record_auxiliary_load(*, worker: Any, calls: list[dict[str, Any]], fails: bool) -> Callable[..., None]:
    def load(*args: Any, **kwargs: Any) -> None:
        calls.append(vars(worker.args.backend).copy())
        if fails:
            raise RuntimeError("auxiliary SDK failed")

    return load


def _core_worker(
    *, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, directory: Path
) -> tuple[Any, dict[str, Any]]:
    _write_checkpoint(directory / "pretrain", iteration=7)
    args = _parsed_trainer("--load", str(directory / "pretrain"))
    with args.backend.mutable():
        args.backend.ckpt_format = "torch"
    worker = _actor(actor_module, role="critic", args=args)
    seen = _watch_load(actor_module, monkeypatch, args=args, iteration=7)
    return worker, seen


def _vocabulary_worker(*, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, vocab_size: int | None) -> Any:
    args = _parsed_trainer("--debug-train-only", "--megatron-to-hf-mode", "raw")
    with args.backend.mutable():
        args.backend.vocab_size = vocab_size
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = args
    worker.role = "actor"
    worker._rank = 0
    worker.tokenizer = SimpleNamespace(vocab_size=128)
    parallel = SimpleNamespace(cp=SimpleNamespace(size=1), train_parallel_config=lambda **kwargs: None)

    monkeypatch.setattr(actor_module.TrainRayActor, "_init_common", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "monkey_patch_torch_dist", lambda: None)
    monkeypatch.setattr(actor_module, "all_replay_managers", [])
    monkeypatch.setattr(actor_module, "routing_replay_manager", SimpleNamespace())
    monkeypatch.setattr(actor_module, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "fault_hook_controller", SimpleNamespace(configure=lambda **kwargs: None))
    monkeypatch.setattr(actor_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(actor_module, "dashboard_hooks", SimpleNamespace(register_train_actor=lambda args: None))
    monkeypatch.setattr(actor_module, "TrainProfiler", lambda args: SimpleNamespace())
    monkeypatch.setattr(worker, "load_hf_assets", lambda: None)
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(actor_module, "build_model_and_optimizer", lambda *args, **kwargs: ([], None, None))
    monkeypatch.setattr(actor_module, "RandomState", SimpleNamespace(capture=lambda: None))
    monkeypatch.setattr(actor_module, "verify_megatron_parallel_state", lambda model: None)
    monkeypatch.setattr(actor_module, "TensorBackuper", SimpleNamespace(create=lambda **kwargs: SimpleNamespace()))
    monkeypatch.setattr(
        worker, "_load_state_core", Mock(side_effect=RuntimeError("stopped after vocabulary resolution"))
    )
    return worker
