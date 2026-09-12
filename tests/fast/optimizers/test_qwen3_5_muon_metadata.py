import pytest
import torch

import miles_plugins.models.qwen3_5 as qwen3_5


class _FakeConfig:
    hidden_size = 256
    linear_num_value_heads = 4
    linear_num_key_heads = 2
    linear_key_head_dim = 64
    linear_value_head_dim = 64
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = torch.float32


def test_qwen3_5_gdn_marks_packed_qkv_for_muon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_5, "ShortConvolution", torch.nn.Identity, raising=False)
    monkeypatch.setattr(qwen3_5, "FusedRMSNormGated", torch.nn.Identity, raising=False)
    monkeypatch.setattr(qwen3_5, "get_chunk_gated_delta_rule", lambda _backend: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    module = qwen3_5.Qwen3_5GatedDeltaNet(_FakeConfig, layer_idx=0)
    qkv_weight = module.in_proj_qkv.weight

    assert qkv_weight.is_qkv is True
    assert qkv_weight.qkv_split_shapes == (module.key_dim, module.key_dim, module.value_dim)
    assert sum(qkv_weight.qkv_split_shapes) == qkv_weight.shape[0]
