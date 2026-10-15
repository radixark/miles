"""Step5's full-attention layers: which layers the spec builds sparse, and how their tensors are named and kept.

Built on CPU with Megatron's local linears; the GPU tests (``tests/fast-gpu/test_step5_sparse_attention.py``)
run the Transformer Engine layers.
"""

import json

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")
pytest.importorskip("triton")  # the sparse attention kernel's module imports it
from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear  # noqa: E402
from megatron.core.transformer.attention import SelfAttentionSubmodules  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from megatron.core.transformer.spec_utils import ModuleSpec, build_module  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from tests.fast.launch_scripts.model_args_harness import expand_model_args  # noqa: E402

from miles.backends.megatron_utils.megatron_to_hf.step5 import convert_step5_to_hf  # noqa: E402
from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES  # noqa: E402
from miles_plugins.models.step5.layer_math import FULL_ATTENTION_ROPE, SLIDING_WINDOW_ROPE  # noqa: E402
from miles_plugins.models.step5.layers import Step5Attention, Step5RMSNorm  # noqa: E402
from miles_plugins.models.step5.model import build_step5_spec  # noqa: E402
from miles_plugins.models.step5.sparse_attention import Step5SparseAttention, Step5SparseCoreAttention  # noqa: E402

# The original layer indices each model flag set builds (scripts/models/step5*.py).
_ORIGINAL_LAYERS = {"step5": range(92), "step5-5layer": (0, 3, 4, 88, 91)}


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


@pytest.fixture(scope="module")
def model_parallel(tmp_path_factory):
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        rendezvous = tmp_path_factory.mktemp("step5-sparse") / "process-group"
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        if owns_process_group:
            dist.destroy_process_group()


def _flags_config(model_type: str) -> TransformerConfig:
    argv = expand_model_args(model_type)
    window = json.loads(_flag(argv, "--window-attn-skip-freq"))
    return TransformerConfig(
        num_layers=len(window),
        hidden_size=64,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=192,
        window_size=(511, 0),
        window_attn_skip_freq=window,
        num_moe_experts=352,
        moe_layer_freq=json.loads(_flag(argv, "--moe-layer-freq")),
        moe_ffn_hidden_size=1536,
        moe_router_topk=8,
        moe_grouped_gemm=True,
    )


@pytest.mark.parametrize("model_type", sorted(_ORIGINAL_LAYERS))
def test_spec_builds_every_fourth_layer_sparse(model_parallel, model_type):
    spec = build_step5_spec(_flags_config(model_type))

    attention = [layer.submodules.self_attention for layer in spec.layer_specs]
    assert len(attention) == len(_ORIGINAL_LAYERS[model_type])
    for index, layer in zip(_ORIGINAL_LAYERS[model_type], attention, strict=True):
        full = index % 4 == 3
        assert layer.module is (Step5SparseAttention if full else Step5Attention), index
        assert (layer.submodules.core_attention is Step5SparseCoreAttention) == full, index
        assert layer.params["attn_mask_type"] == AttnMaskType.causal


def _attention_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=192,
        layernorm_epsilon=1e-5,
        qk_layernorm=True,
        head_wise_attn_gate=True,
        add_bias_linear=False,
        window_size=(511, 0),
        window_attn_skip_freq=[1, 1, 1, 0],
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )


def _build_attention(module, core_attention, layer_number: int):
    spec = ModuleSpec(
        module=module,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=RowParallelLinear,
            q_layernorm=Step5RMSNorm,
            k_layernorm=Step5RMSNorm,
        ),
    )
    return build_module(spec, config=_attention_config(), layer_number=layer_number)


def _bridge():
    # mbridge is installed in the training image but not in CPU CI, so only the bridge tests skip there
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.step5 import Step5Bridge

    bridge = object.__new__(Step5Bridge)
    bridge.make_vocab_size_divisible_by = None
    return bridge


def test_full_attention_parameters_are_the_bridges_attention_tensors(model_parallel):
    attention = _build_attention(Step5SparseAttention, Step5SparseCoreAttention, layer_number=4)
    names = {f"self_attention.{name}" for name, _ in attention.named_parameters()}

    bridge = _bridge()
    assert names == set(bridge._ATTENTION_MAPPING)
    for name in sorted(names):
        hf_names = bridge._weight_name_mapping_attention(f"decoder.layers.3.{name}")
        assert all(hf_name.startswith("model.layers.3.self_attn.") for hf_name in hf_names), (name, hf_names)
    assert {
        name: bridge._weight_name_mapping_attention(f"decoder.layers.3.self_attention.indexer.{name}")
        for name in CHECKPOINT_NAMES
    } == {name: [f"model.layers.3.self_attn.{checkpoint}"] for name, checkpoint in CHECKPOINT_NAMES.items()}


def test_sliding_window_parameters_are_the_bridges_without_the_indexer(model_parallel):
    attention = _build_attention(Step5Attention, _DenseCoreAttentionStub, layer_number=1)
    names = {f"self_attention.{name}" for name, _ in attention.named_parameters()}
    assert names == {name for name in _bridge()._ATTENTION_MAPPING if ".indexer." not in name}


def test_each_attention_type_keeps_its_rope(model_parallel):
    sparse = _build_attention(Step5SparseAttention, Step5SparseCoreAttention, layer_number=4)
    sliding = _build_attention(Step5Attention, _DenseCoreAttentionStub, layer_number=1)
    for attention, rope, rotated in ((sparse, FULL_ATTENTION_ROPE, 64), (sliding, SLIDING_WINDOW_ROPE, 192)):
        expected = 1.0 / rope.theta ** (torch.arange(0, rotated, 2, dtype=torch.float32) / rotated)
        torch.testing.assert_close(attention.rotary_pos_emb.inv_freq, expected)


def test_only_the_indexer_is_frozen(model_parallel):
    attention = _build_attention(Step5SparseAttention, Step5SparseCoreAttention, layer_number=4)

    frozen = {name for name, param in attention.named_parameters() if not param.requires_grad}
    assert frozen == {f"indexer.{name}" for name in CHECKPOINT_NAMES}


# A full-attention layer's ``model.layers.N.self_attn.*`` tensors, with the dtypes the checkpoint stores.
_CHECKPOINT_FULL_ATTENTION = {
    "q_proj.weight": torch.bfloat16,
    "k_proj.weight": torch.bfloat16,
    "v_proj.weight": torch.bfloat16,
    "g_proj.weight": torch.bfloat16,
    "o_proj.weight": torch.bfloat16,
    "q_norm.weight": torch.float32,
    "k_norm.weight": torch.float32,
    "sparse_indexer_q.weight": torch.bfloat16,
    "sparse_indexer_k.weight": torch.bfloat16,
    "sparse_indexer_z.weight": torch.bfloat16,
    "sparse_indexer_w.weight": torch.float32,
    "sparse_indexer_q_norm.weight": torch.float32,
    "sparse_indexer_k_norm.weight": torch.float32,
    "sparse_indexer_k_norm.bias": torch.float32,
    "ssmax_s": torch.float32,
}


def test_full_attention_exports_every_checkpoint_tensor(model_parallel):
    attention = _build_attention(Step5SparseAttention, Step5SparseCoreAttention, layer_number=4)

    exported = {}
    for name, param in attention.named_parameters():
        megatron_name = f"module.module.decoder.layers.3.self_attention.{name}"
        exported |= dict(convert_step5_to_hf(_SyncArgs(), megatron_name, param))
    prefix = "model.layers.3.self_attn."
    assert {name.removeprefix(prefix): tensor.dtype for name, tensor in exported.items()} == _CHECKPOINT_FULL_ATTENTION


class _DenseCoreAttentionStub(torch.nn.Module):
    """Stands in for TE's parameter-free dense attention, which needs a GPU to build."""

    def __init__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None,
        softmax_scale: float | None,
        pg_collection,
    ) -> None:
        super().__init__()


class _SyncArgs:
    num_query_groups = 4
    num_attention_heads = 64
    kv_channels = 192
    tensor_model_parallel_size = 1
