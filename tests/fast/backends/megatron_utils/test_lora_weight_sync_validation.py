"""Compatibility tests for current LoRA weight-sync behavior."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import torch

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"

SAMPLE_MIXED_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
]


def _make_args() -> Namespace:
    return Namespace(
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["linear_qkv", "linear_proj"],
        megatron_to_hf_mode="bridge",
        rollout_num_gpus_per_engine=1,
        hf_checkpoint="/fake/path",
        update_weight_buffer_size=1 << 20,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )


class TestSendHfParamsCompatibility:
    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_lora_sync_filters_out_base_weights(self, mock_iter_base, mock_dist, mock_send_to_colocated_engine):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        updater = UpdateWeightFromTensor(
            args=_make_args(),
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        updater._ipc_engine = MagicMock()
        updater._ipc_gather_src = 0
        updater._ipc_gather_group = MagicMock()
        updater.use_distribute = False

        updater._send_hf_params(SAMPLE_MIXED_WEIGHTS)

        sent_weights = mock_send_to_colocated_engine.call_args.kwargs["hf_named_tensors"]
        assert [name for name, _ in sent_weights] == [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        assert updater._lora_loaded is True

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_base_sync_keeps_all_weights_and_version(self, mock_iter_base, mock_dist, mock_send_to_colocated_engine):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        updater = UpdateWeightFromTensor(
            args=_make_args(),
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=False,
        )
        updater._ipc_engine = MagicMock()
        updater._ipc_gather_src = 0
        updater._ipc_gather_group = MagicMock()
        updater.use_distribute = False
        updater.weight_version = 7

        updater._send_hf_params(SAMPLE_MIXED_WEIGHTS)

        send_kwargs = mock_send_to_colocated_engine.call_args.kwargs
        assert send_kwargs["hf_named_tensors"] == SAMPLE_MIXED_WEIGHTS
        assert send_kwargs["weight_version"] == 7
