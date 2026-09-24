"""Every Megatron parameter of Kimi-K3's MLA and KDA attention layers has an HF name in both the mbridge
map (HF -> Megatron load) and the megatron_to_hf map (weight updates and export). The two layer kinds
share leaf names such as ``g_proj`` and ``o_proj``, so a rename made for one kind can silently drop the
other; conversion then fails only on a real checkpoint."""

import os

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"])

pytest.importorskip("fla")
pytest.importorskip("mbridge")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import MLATransformerConfig  # noqa: E402

from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import _LAYER_NAMES  # noqa: E402
from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge  # noqa: E402
from miles_plugins.models.kimi_k3.layers import KimiK3Attention, KimiK3KDAAttention  # noqa: E402


@pytest.fixture(scope="module")
def config():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        params_dtype=torch.bfloat16,
        bf16=True,
        layernorm_epsilon=1e-5,
    )
    config.kimi_kda_layers = (2,)
    config.kimi_linear_num_heads = 4
    config.kimi_linear_head_dim = 64
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    yield config
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize("layer_number", [1, 2], ids=["mla", "kda"])
def test_every_attention_parameter_has_an_hf_name(config, layer_number):
    layer_cls = KimiK3KDAAttention if layer_number in config.kimi_kda_layers else KimiK3Attention
    layer = layer_cls(config, layer_number=layer_number)
    names = [f"self_attention.{name}" for name, _ in layer.named_parameters()]
    assert names
    bridge_names = set(KimiK3Bridge._ATTENTION_MAPPING)
    assert [name for name in names if name not in bridge_names] == []
    assert [name for name in names if name not in _LAYER_NAMES] == []
    disagreeing = [
        name
        for name in names
        if [hf.format(layer_number=0) for hf in KimiK3Bridge._ATTENTION_MAPPING[name]]
        != [f"language_model.model.layers.0.{_LAYER_NAMES[name]}"]
    ]
    assert disagreeing == []
