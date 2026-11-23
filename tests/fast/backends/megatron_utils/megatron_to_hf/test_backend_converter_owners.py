import json
from pathlib import Path

import pytest
import torch

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.megatron_utils.megatron_to_hf.inkling import convert_inkling_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.backends.training_utils import parallel
from miles.utils.args.runtime import TrainerConfig


class TestQwenBackendGeometry:
    @pytest.mark.parametrize("family", ["3.5", "3.8"])
    @pytest.mark.parametrize("kv_channels", [None, 1, 2])
    def test_weight_and_bias_exports_read_only_backend_geometry(self, family: str, kv_channels: int | None) -> None:
        """Both converter entrypoints preserve grouped QKV rows using backend-owned geometry."""
        args = TrainerConfig.model_construct(
            backend=MegatronArgsNamespace(
                hidden_size=4, num_attention_heads=4, num_query_groups=2, kv_channels=kv_channels
            )
        )
        convert = convert_qwen3_5_to_hf if family == "3.5" else convert_qwen3_8_next_to_hf
        head_dim = 2 if kv_channels == 2 else 1
        weight = torch.arange(12 * head_dim * 4, dtype=torch.float32).reshape(12 * head_dim, 4)
        bias = torch.arange(8 * head_dim, dtype=torch.float32)

        weights = convert(args, "module.module.decoder.layers.3.self_attention.linear_qkv.weight", weight)
        biases = convert(args, "module.module.decoder.layers.3.self_attention.linear_qkv.bias", bias)

        assert [name for name, _ in weights] == [
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "model.language_model.layers.3.self_attn.v_proj.weight",
        ]
        assert [name for name, _ in biases] == [name.replace("weight", "bias") for name, _ in weights]
        q_rows = (
            [0, 2, 1, 3, 6, 8, 7, 9] if head_dim == 1 else [0, 1, 4, 5, 2, 3, 6, 7, 12, 13, 16, 17, 14, 15, 18, 19]
        )
        k_rows = [4, 10] if head_dim == 1 else [8, 9, 20, 21]
        v_rows = [5, 11] if head_dim == 1 else [10, 11, 22, 23]
        for (_, actual), rows in zip(weights, [q_rows, k_rows, v_rows], strict=True):
            assert torch.equal(actual, weight[rows])
        bias_rows = (
            [[0, 1, 4, 5], [2, 6], [3, 7]]
            if head_dim == 1
            else [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]]
        )
        for (_, actual), rows in zip(biases, bias_rows, strict=True):
            assert torch.equal(actual, bias[rows])


class TestInklingBackendFallback:
    @pytest.mark.parametrize("tp_size", [None, 1, 2])
    def test_uninitialized_parallel_state_uses_backend_tp_for_actual_qkvr_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tp_size: int | None
    ) -> None:
        """Offline QKVR export deinterleaves gathered ranks using the backend TP fallback."""
        (tmp_path / "config.json").write_text(
            json.dumps(dict(local_layer_ids=[], num_attention_heads=2, num_key_value_heads=2, head_dim=1, d_rel=1))
        )
        args = TrainerConfig.model_construct(
            hf_checkpoint=str(tmp_path), backend=MegatronArgsNamespace(tensor_model_parallel_size=tp_size)
        )
        monkeypatch.setattr(parallel, "_parallel_state", None)
        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)

        converted = convert_inkling_to_hf(
            args, "module.module.decoder.layers.0.self_attention.linear_qkv.weight", weight
        )

        assert [name for name, _ in converted] == [
            "model.llm.layers.0.attn.wq_du.weight",
            "model.llm.layers.0.attn.wk_dv.weight",
            "model.llm.layers.0.attn.wv_dv.weight",
            "model.llm.layers.0.attn.wr_du.weight",
        ]
        expected = [[0, 4], [1, 5], [2, 6], [3, 7]] if tp_size == 2 else [[0, 1], [2, 3], [4, 5], [6, 7]]
        for (_, actual), rows in zip(converted, expected, strict=True):
            assert torch.equal(actual, weight[rows])
