from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf import linear_attn_layout
from miles.backends.megatron_utils.megatron_to_hf.qwen3_next import convert_qwen3_next_to_hf
from miles.utils.args.runtime import TrainerConfig


class TestExplicitCheckpoint:
    @pytest.mark.parametrize("mtp", [False, True])
    def test_convolution_export_uses_shared_hf_path_with_backend_only_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mtp: bool
    ) -> None:
        """Normal and MTP GDN convolutions read the shared HF path from trainer configuration."""
        checkpoint = str(tmp_path)
        loaded = []

        def load_config(path: str) -> Namespace:
            loaded.append(path)
            return Namespace(
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=1, linear_value_head_dim=1
            )

        monkeypatch.setattr(linear_attn_layout, "load_hf_config", load_config)
        backend = Namespace(kv_channels=1, hidden_size=1, num_attention_heads=2, num_query_groups=2)
        args = TrainerConfig.model_construct(backend=backend, hf_checkpoint=checkpoint)
        prefix = "mtp.layers.0.transformer_layer" if mtp else "decoder.layers.0"
        weight = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)

        converted = convert_qwen3_next_to_hf(
            args,
            f"module.module.{prefix}.self_attention.linear_attn.conv1d.weight",
            weight,
        )

        expected_prefix = "mtp.layers.0" if mtp else "model.layers.0"
        assert len(converted) == 1
        assert converted[0][0] == f"{expected_prefix}.linear_attn.conv1d.weight"
        assert torch.equal(converted[0][1], weight[[0, 3, 1, 4, 2, 5]])
        assert loaded == [checkpoint]
