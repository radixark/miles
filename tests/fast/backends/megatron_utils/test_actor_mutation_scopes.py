import importlib
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest
from tests.fast.backends.megatron_utils.actor_scope_fixtures import (
    _auxiliary_worker,
    _core_worker,
    _record_auxiliary_load,
    _vocabulary_worker,
)
from tests.fast.backends.megatron_utils.test_shared_ppo_lifecycle import actor_module

from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo

__all__ = ["actor_module"]


class TestAuxiliaryCheckpointMutationScope:
    @pytest.mark.parametrize("model_tag,expected_step", [("ref", 3), ("teacher", 9), ("old_actor", 7)])
    def test_auxiliary_weights_use_private_load_settings_then_restore_the_actor_configuration(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, model_tag: str, expected_step: int
    ) -> None:
        """Auxiliary weight loading restores actor flags and preserves each model's checkpoint step."""
        worker = _auxiliary_worker(actor_module=actor_module)
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            actor_module, "load_checkpoint", _record_auxiliary_load(worker=worker, calls=calls, fails=False)
        )

        request = MegatronCkptLoadConfig.for_peer_recovery().model_copy(update={"load": "/actor", "ckpt_step": 7})
        with request.apply(worker.args.backend):
            original = vars(worker.args.backend).copy()
            worker.load_other_checkpoint(model_tag, "/auxiliary")
            assert vars(worker.args.backend) == original

        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(worker.args.backend))
        assert len(calls) == 1
        assert calls[0]["load"] == "/auxiliary"
        assert calls[0]["no_load_optim"] is calls[0]["no_load_rng"] is calls[0]["finetune"] is True
        assert calls[0]["ckpt_step"] == expected_step
        worker.weights_backuper.backup.assert_called_once_with(model_tag)
        with pytest.raises(AttributeError, match="immutable"):
            worker.args.backend.ckpt_step = 99

    def test_auxiliary_sdk_failure_revokes_mutation_without_backing_up_failed_weights(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed auxiliary load cannot retain write authority or publish a backup."""
        worker = _auxiliary_worker(actor_module=actor_module)
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            actor_module, "load_checkpoint", _record_auxiliary_load(worker=worker, calls=calls, fails=True)
        )

        request = MegatronCkptLoadConfig.for_peer_recovery().model_copy(update={"load": "/actor", "ckpt_step": 7})
        with request.apply(worker.args.backend):
            with pytest.raises(RuntimeError, match="auxiliary SDK failed"):
                worker.load_other_checkpoint("ref", "/auxiliary")

        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(worker.args.backend))

        assert len(calls) == 1
        worker.weights_backuper.backup.assert_not_called()
        with pytest.raises(AttributeError, match="immutable"):
            worker.args.backend.load = "/leaked"


class TestCoreCheckpointMutationScope:
    @pytest.mark.parametrize("fails", [False, True])
    def test_temporary_checkpoint_format_is_restored_before_the_backend_lease_ends(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fails: bool
    ) -> None:
        """Temporary checkpoint overrides restore on both normal and exceptional weight-loader exits."""
        worker, seen = _core_worker(actor_module=actor_module, monkeypatch=monkeypatch, directory=tmp_path)
        if fails:
            model = importlib.import_module("miles.backends.megatron_utils.model")
            monkeypatch.setattr(model, "load_checkpoint", Mock(side_effect=RuntimeError("core SDK failed")))

        request = MegatronCkptLoadConfig.for_peer_recovery().model_copy(update={"load": str(tmp_path / "pretrain")})
        with request.apply(worker.args.backend):
            if fails:
                with pytest.raises(RuntimeError, match="core SDK failed"):
                    worker._load_state_core(
                        checkpointing_context=None,
                        overrider_for_loading={"ckpt_format": "torch_dist"},
                        resume_from_ckpt=True,
                    )
            else:
                result = worker._load_state_core(
                    checkpointing_context=None,
                    overrider_for_loading={"ckpt_format": "torch_dist"},
                    resume_from_ckpt=True,
                )
                assert (result.loaded_rollout_id, result.start_rollout_id) == (7, 8)
                assert seen["args_during_load"]["ckpt_format"] == "torch_dist"

        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(worker.args.backend))
        assert worker.args.backend.ckpt_format == "torch"
        with pytest.raises(AttributeError, match="immutable"):
            worker.args.backend.ckpt_format = "leaked"


class TestActorVocabularyMutationScope:
    @pytest.mark.parametrize("declared,expected", [(None, 128), (77, 77)])
    def test_actor_startup_fills_only_an_unspecified_vocabulary_and_remains_immutable(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, declared: int | None, expected: int
    ) -> None:
        """Actor startup derives a missing vocabulary from the tokenizer without replacing an explicit value."""
        worker = _vocabulary_worker(actor_module=actor_module, monkeypatch=monkeypatch, vocab_size=declared)
        body = actor_module.MegatronTrainRayActor.init.__wrapped__.__wrapped__

        with pytest.raises(RuntimeError, match="stopped after vocabulary resolution"):
            body(
                worker,
                args=worker.args,
                role="actor",
                ckpt_load_config=MegatronCkptLoadConfig.for_peer_recovery(),
                indep_dp_info=IndepDPInfo.create_trivial(),
                indep_dp_store_addr=None,
            )

        assert worker.args.backend.vocab_size == expected
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(worker.args.backend))
        with pytest.raises(AttributeError, match="immutable"):
            worker.args.backend.vocab_size = 99
