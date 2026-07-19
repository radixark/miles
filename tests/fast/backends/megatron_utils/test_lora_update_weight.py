"""Mock-based tests for LoRA weight-sync logic in update_weight_from_tensor.py.

Validates that _send_hf_params correctly separates LoRA vs base weights
and that UpdateWeightFromTensor initialises _lora_config only when LoRA is active.
"""

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name

# ---------------------------------------------------------------------------
# LoRA / base weight separation (pure logic, no distributed deps)
# ---------------------------------------------------------------------------


class TestLoraWeightSeparation:
    """Test the filtering logic that _send_hf_params relies on."""

    SAMPLE_WEIGHTS = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
        ("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 4)),
        ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(8, 2)),
        ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(2, 8)),
    ]

    def test_separation_when_lora(self):
        base = [(n, t) for n, t in self.SAMPLE_WEIGHTS if not is_lora_weight_name(n)]
        lora = [(n, t) for n, t in self.SAMPLE_WEIGHTS if is_lora_weight_name(n)]
        assert len(base) == 2
        assert len(lora) == 4

    def test_no_separation_when_not_lora(self):
        base = self.SAMPLE_WEIGHTS
        lora = []
        assert len(base) == 6
        assert len(lora) == 0

    def test_lora_names_contain_lora_A_or_B(self):
        lora = [(n, t) for n, t in self.SAMPLE_WEIGHTS if is_lora_weight_name(n)]
        for name, _ in lora:
            assert ".lora_A." in name or ".lora_B." in name

    def test_base_names_do_not_contain_lora(self):
        base = [(n, t) for n, t in self.SAMPLE_WEIGHTS if not is_lora_weight_name(n)]
        for name, _ in base:
            assert ".lora_A." not in name
            assert ".lora_B." not in name


def test_cpu_backup_is_zero_copy(monkeypatch):
    from miles.backends.megatron_utils.update_weight import common

    tensor = torch.randn(4)
    backup = torch.randn(4)
    get_cpu_backup = MagicMock(return_value=backup)
    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        SimpleNamespace(torch_memory_saver=SimpleNamespace(get_cpu_backup=get_cpu_backup)),
    )

    assert common._maybe_get_cpu_backup(tensor) is backup
    get_cpu_backup.assert_called_once_with(tensor, zero_copy=True)


# ---------------------------------------------------------------------------
# UpdateWeightFromTensor._lora_config initialisation
# ---------------------------------------------------------------------------

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


class TestUpdateWeightFromTensorLoraConfig:
    """Verify _lora_config is set only when is_lora=True."""

    def _make_args(self):
        return Namespace(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["linear_qkv", "linear_proj"],
            megatron_to_hf_mode="bridge",
            rollout_num_gpus_per_engine=2,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_lora_true_sets_config(self, mock_iter_base, mock_dist):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        args = self._make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        assert updater._lora_config is not None
        assert updater._lora_config["peft_type"] == "LORA"
        assert updater._lora_config["r"] == 32

    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_lora_false_no_config(self, mock_iter_base, mock_dist):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        args = self._make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=False,
        )
        assert not hasattr(updater, "_lora_config")


@patch(f"{_UW_MODULE}.end_weight_update")
@patch(f"{_UW_MODULE}.begin_weight_update")
@patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
@patch(f"{_UW_MODULE}.ray.get")
@patch(f"{_UW_MODULE}.dist")
def test_update_weights_can_defer_generation_resume(
    mock_dist,
    _mock_ray_get,
    _mock_get_gloo_group,
    _mock_begin_weight_update,
    _mock_end_weight_update,
):
    from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    engine = MagicMock()
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(pause_generation_mode="retract")
    updater.is_lora = False
    updater.use_distribute = True
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.weights_getter = MagicMock(return_value={})
    updater._hf_weight_iterator = MagicMock()
    updater._hf_weight_iterator.get_hf_weight_chunks.return_value = []
    mock_dist.get_rank.return_value = 0

    updater.update_weights(resume_generation=False)

    engine.pause_generation.remote.assert_called_once_with(mode="retract")
    engine.continue_generation.remote.assert_not_called()
