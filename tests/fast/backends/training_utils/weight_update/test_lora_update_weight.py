"""Mock-based tests for LoRA weight-sync logic.

Validates the LoRA vs base weight-name separation and that WeightUpdater
requires a lora_sync_config exactly when LoRA is active.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.lora.utils import is_lora_weight_name

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


# ---------------------------------------------------------------------------
# WeightUpdater lora_sync_config initialisation
# ---------------------------------------------------------------------------

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


class TestWeightUpdaterLoraConfig:
    """The updater requires a lora_sync_config exactly when LoRA is active."""

    def _make_updater(self, *, is_lora, lora_sync_config, args=None):
        protocol = MagicMock()
        protocol.supports_lora = True
        with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
            return WeightUpdater(
                args or Namespace(),
                [MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen",
                quantization_config=None,
                iterator_factory=lambda *a, **k: MagicMock(),
                parallel_state=MagicMock(),
                is_lora=is_lora,
                lora_sync_config=lora_sync_config,
            )

    def test_lora_requires_config(self):
        with pytest.raises(AssertionError):
            self._make_updater(is_lora=True, lora_sync_config=None)

    def test_lora_config_stored(self):
        updater = self._make_updater(is_lora=True, lora_sync_config={"peft_type": "LORA", "r": 32})
        assert updater._lora_sync_config["r"] == 32

    def test_no_lora_no_config(self):
        updater = self._make_updater(is_lora=False, lora_sync_config=None)
        assert updater._lora_sync_config is None

    @pytest.mark.parametrize("attention_dp", [False, True], ids=["no-attention-dp", "attention-dp"])
    def test_registration_pins_the_adapter_only_under_attention_dp(self, attention_dp):
        """SGLang's attention-DP LoRA rejects unpinned adapters; other layouts keep them evictable."""
        updater = self._make_updater(
            is_lora=True,
            lora_sync_config={"peft_type": "LORA", "r": 8},
            args=Namespace(
                sglang=make_sglang_config(attn_dp_size=8 if attention_dp else 1, enable_dp_attention=False, dp_size=1)
            ),
        )
        engines = [MagicMock()]
        with patch(f"{_UPDATER_MODULE}.register_lora_adapter") as register:
            updater._register_new_lora_adapters(engines, [("miles_lora", None)])
        register.assert_called_once_with(
            engines, lora_name="miles_lora", lora_config={"peft_type": "LORA", "r": 8}, pinned=attention_dp
        )
