from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from tests.fast.backends.megatron_utils.explicit_peer_config_fixtures import (
    PeerCheckpointBoundary,
    _persistent_trainer_args,
    peer_checkpoint_boundary,
)
from tests.fast.backends.megatron_utils.test_actor import actor_module
from tests.fast.backends.megatron_utils.test_checkpoint_transfer import state_dict

from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
from miles.utils.ft_utils.process_group_utils import GroupInfo

__all__ = ["actor_module", "state_dict", "peer_checkpoint_boundary"]


class TestExplicitPeerCheckpointConfig:
    def test_a_donor_transfer_and_peer_load_use_each_actors_own_trainer_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        peer_checkpoint_boundary: PeerCheckpointBoundary,
        state_dict: MCoreTensorAwareStateDict,
    ) -> None:
        """A donor sends its trained state while the recovering peer loads with its own configuration."""
        boundary = peer_checkpoint_boundary
        model = torch.nn.Linear(2, 2)
        model.lora_attached = False
        donor = object.__new__(boundary.actor_module.MegatronTrainRayActor)
        donor.args = _persistent_trainer_args(
            keep_old_actor=False, ci_test=False, use_distributed_optimizer=False, overlap_param_gather=False
        )
        donor.model = [model]
        donor.optimizer = object()
        donor.opt_param_scheduler = object()
        donor._last_rollout_id = 17
        group = GroupInfo(rank=0, size=2, group=None)
        monkeypatch.setattr(boundary.actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=group))
        expected_tensors = [tensor.clone() for tensor in state_dict.tensors]

        donor.send_ckpt(dst_rank=1)
        boundary.transport.recv_payload = boundary.transport.sent_payloads[0]
        manager = boundary.transfer_module.recv_ckpt(indep_dp=group, src_rank=0)
        peer = object.__new__(boundary.actor_module.MegatronTrainRayActor)
        peer.args = _persistent_trainer_args(pretrained_checkpoint=None)
        peer.role = "critic"
        peer.model = [model]
        peer.optimizer = object()
        peer.opt_param_scheduler = None
        context = {"local_checkpoint_manager": manager}
        request = MegatronCkptLoadConfig.for_peer_recovery()

        with request.apply(peer.args.backend):
            output = peer._load_state_core(
                checkpointing_context=context, overrider_for_loading={}, resume_from_ckpt=request.resume_from_ckpt
            )

        [saved] = boundary.save_calls
        assert saved["iteration"] == 17
        assert saved["model"] is donor.model
        assert saved["optimizer"] is donor.optimizer
        assert saved["opt_param_scheduler"] is donor.opt_param_scheduler
        assert saved["non_persistent_ckpt"] is True
        assert boundary.transport.sent_dst_ranks == [[1]]
        assert boundary.transport.disallow_calls == 1
        assert manager.find_latest() == 17
        restored, _ = manager.load()
        for expected, actual in zip(expected_tensors, restored.tensors, strict=True):
            torch.testing.assert_close(actual, expected)
        [load] = boundary.checkpoint_calls
        assert load["args"] is peer.args
        assert load["args"] is not donor.args
        assert load["ckpt_step"] is None
        assert load["local_checkpoint_manager"] is True
        assert boundary.load_calls[0]["checkpointing_context"] is context
        assert boundary.load_calls[0]["ddp_model"] is peer.model
        assert output.loaded_rollout_id == 17
        assert output.start_rollout_id == 18
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(peer.args.backend))
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(donor.args.backend))

    @pytest.mark.parametrize("model_tag,expected_step", [("ref", 7), ("teacher", 11)])
    def test_an_auxiliary_load_passes_the_live_trainer_config_to_the_real_checkpoint_loader(
        self, tmp_path: Path, peer_checkpoint_boundary: PeerCheckpointBoundary, model_tag: str, expected_step: int
    ) -> None:
        """Reference and teacher loading use temporary options on the actor payload and then restore them."""
        boundary = peer_checkpoint_boundary
        path = tmp_path / model_tag
        path.mkdir()
        (path / "latest_checkpointed_iteration.txt").write_text("7\n")
        actor = object.__new__(boundary.actor_module.MegatronTrainRayActor)
        actor.args = _persistent_trainer_args(ref_ckpt_step=7, opd_teacher_ckpt_step=11)
        actor.model = []
        actor.weights_backuper = Mock()
        request = MegatronCkptLoadConfig(
            load="/original/trainer",
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            ckpt_step=3,
            resume_from_ckpt=True,
        )

        with request.apply(actor.args.backend):
            original = dict(vars(actor.args.backend))
            actor.load_other_checkpoint(model_tag=model_tag, path=str(path))
            assert vars(actor.args.backend) == original

        [load] = boundary.checkpoint_calls
        assert load["args"] is actor.args
        assert load["load"] == str(path)
        assert load["ckpt_step"] == expected_step
        assert load["backend"]["no_load_optim"] is True
        assert load["backend"]["no_load_rng"] is True
        assert load["backend"]["finetune"] is True
        assert boundary.load_calls[0]["ddp_model"] is actor.model
        assert boundary.load_calls[0]["skip_load_to_model_and_opt"] is False
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(actor.args.backend))
        actor.weights_backuper.backup.assert_called_once_with(model_tag)
        assert actor._active_model_tag == model_tag
