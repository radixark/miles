"""Step5's sparse layer specs, parameter names and Indexer selection reuse.

Built on CPU with Megatron's local linears; the GPU tests (``tests/fast-gpu/test_step5_sparse_attention.py``)
run the Transformer Engine layers and real Indexer kernels.
"""

import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

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
from miles_plugins.models.step5 import get_step5_spec  # noqa: E402
from miles_plugins.models.step5 import model as step5_model  # noqa: E402
from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES  # noqa: E402
from miles_plugins.models.step5.layers import (  # noqa: E402
    FULL_ATTENTION_ROPE,
    SLIDING_WINDOW_ROPE,
    Step5Attention,
    Step5RMSNorm,
)
from miles_plugins.models.step5.region_selection import RegionSelection  # noqa: E402
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
@pytest.mark.parametrize(
    "settings,reuse",
    [
        ({}, False),
        ({"reuse_indexer_on_recompute": False}, False),
        ({"reuse_indexer_on_recompute": True}, True),
    ],
)
def test_spec_builds_every_fourth_layer_sparse(model_parallel, monkeypatch, model_type, settings, reuse):
    monkeypatch.setattr(step5_model, "load_hf_config", lambda _: SimpleNamespace(text_config=None))
    monkeypatch.setattr(step5_model, "check_matches_checkpoint", lambda *_: None)
    args = SimpleNamespace(hf_checkpoint="unused", **settings)
    spec = get_step5_spec(args, _flags_config(model_type), vp_stage=None)

    attention = [layer.submodules.self_attention for layer in spec.layer_specs]
    assert len(attention) == len(_ORIGINAL_LAYERS[model_type])
    for index, layer in zip(_ORIGINAL_LAYERS[model_type], attention, strict=True):
        full = index % 4 == 3
        assert layer.module is (Step5SparseAttention if full else Step5Attention), index
        assert (layer.submodules.core_attention is Step5SparseCoreAttention) == full, index
        assert layer.params["attn_mask_type"] == AttnMaskType.causal
        if full:
            assert layer.params["reuse_indexer_on_recompute"] is reuse
        else:
            assert "reuse_indexer_on_recompute" not in layer.params


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


def _build_attention(module, core_attention, layer_number: int, **params):
    spec = ModuleSpec(
        module=module,
        params={"attn_mask_type": AttnMaskType.causal, **params},
        submodules=SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=RowParallelLinear,
            q_layernorm=Step5RMSNorm,
            k_layernorm=Step5RMSNorm,
        ),
    )
    return build_module(spec, config=_attention_config(), layer_number=layer_number)


@pytest.mark.parametrize(
    "overrides,reused",
    [
        pytest.param({}, True, id="scored-pack"),
        pytest.param({"reuse_indexer_on_recompute": False}, False, id="reuse-disabled"),
        pytest.param({"length": 4104}, True, id="short-pack"),
        pytest.param({"training": False, "checkpointing": False}, False, id="eval-forward-only"),
        pytest.param({"checkpointing": False}, False, id="no-checkpoint"),
    ],
)
def test_attention_reuses_selections_only_during_training_recompute(model_parallel, monkeypatch, overrides, reused):
    case = {
        "length": 4105,
        "training": True,
        "reuse_indexer_on_recompute": True,
        "checkpointing": True,
    } | overrides
    attention = _build_attention(
        Step5SparseAttention,
        Step5SparseCoreAttention,
        layer_number=4,
        reuse_indexer_on_recompute=case["reuse_indexer_on_recompute"],
    )
    attention.train(case["training"])
    assert attention.reuse_indexer_on_recompute is case["reuse_indexer_on_recompute"]
    packed = SimpleNamespace(max_seqlen_q=case["length"])
    calls = []

    def select(*_):
        result = RegionSelection(torch.tensor([len(calls)], dtype=torch.int32), torch.tensor([0, case["length"]]))
        calls.append(result)
        return result

    monkeypatch.setattr(attention.indexer, "select_regions", select)
    monkeypatch.setattr("miles_plugins.models.step5.sparse_attention.is_checkpointing", lambda: case["checkpointing"])
    with torch.set_grad_enabled(case["training"] and not case["checkpointing"]):
        first = attention._select_regions(torch.empty(0), packed)
    assert hasattr(packed, "_step5_indexer_selections") is reused
    if not case["training"] or not case["checkpointing"]:
        assert len(calls) == 1
        return
    with torch.enable_grad():
        second = attention._select_regions(torch.empty(0), packed)
    assert torch.equal(first.region_ids, second.region_ids) is reused
    assert len(calls) == (1 if reused else 2)
    assert not getattr(packed, "_step5_indexer_selections", {})


@pytest.fixture
def selection_attention(model_parallel, monkeypatch):
    def build(*, reuse=True, layer_number=4):
        attention = _build_attention(
            Step5SparseAttention,
            Step5SparseCoreAttention,
            layer_number=layer_number,
            reuse_indexer_on_recompute=reuse,
        )
        selections = []

        def select(hidden_states, packed_seq_params):
            selection = RegionSelection(
                region_ids=hidden_states.sum(-1).argsort(), cu_seqlens=packed_seq_params.cu_seqlens_q
            )
            selections.append(selection)
            return selection

        monkeypatch.setattr(attention.indexer, "select_regions", select)
        return attention, selections

    return build


def _packed():
    return SimpleNamespace(cu_seqlens_q=torch.tensor([0, 5]))


def _selection_forward(attention, hidden, weight, packed, *, checkpointing=True):
    # PyTorch supplies the no_grad/grad-enabled pair; Megatron also sets this flag.
    with patch("miles_plugins.models.step5.sparse_attention.is_checkpointing", return_value=checkpointing):
        selection = attention._select_regions(hidden, packed)
    return (hidden @ weight).sin().index_select(0, selection.region_ids)


def _inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(5, 4, generator=generator, requires_grad=True),
        torch.randn(4, 3, generator=generator, requires_grad=True),
    )


@pytest.mark.parametrize("backward_order", [(0, 1), (1, 0)])
def test_two_outstanding_microbatches_keep_their_own_selections(selection_attention, backward_order):
    results = []
    for reuse in (False, True):
        attention, selections = selection_attention(reuse=reuse)
        packs = [_packed(), _packed()]
        inputs = [_inputs(1), _inputs(2)]
        outputs = []
        for packed, (hidden, weight) in zip(packs, inputs, strict=True):
            forward = partial(_selection_forward, attention, weight=weight, packed=packed)
            outputs.append(checkpoint(forward, hidden, use_reentrant=True))
        for position, batch in enumerate(backward_order):
            outputs[batch].square().sum().backward()
            assert not getattr(packs[batch], "_step5_indexer_selections", {})
            if reuse and position == 0:
                assert packs[1 - batch]._step5_indexer_selections
        assert len(selections) == (2 if reuse else 4)
        results.append([*outputs, *(tensor.grad for pair in inputs for tensor in pair)])
    for baseline, cached in zip(*results, strict=True):
        torch.testing.assert_close(cached, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_groups", [((0, 1), (1, 2), (2, 3)), ((0, 2), (2, 3)), ((0, 1),)])
def test_recompute_layer_groups_preserve_outputs_gradients_and_cache_lifetime(selection_attention, checkpoint_groups):
    def run_layers(x, *, attentions, weights, packed, start, end, checkpointing):
        for layer in range(start, end):
            x = _selection_forward(attentions[layer], x, weights[layer], packed, checkpointing=checkpointing)
        return x

    results = []
    checkpointed_layers = {layer for start, end in checkpoint_groups for layer in range(start, end)}
    for reuse in (False, True):
        packed = _packed()
        attentions, selections = zip(
            *(selection_attention(reuse=reuse, layer_number=layer + 1) for layer in range(3)), strict=True
        )
        generator = torch.Generator().manual_seed(17)
        hidden = torch.randn(5, 4, generator=generator, requires_grad=True)
        weights = [torch.randn(4, 4, generator=generator, requires_grad=True) for _ in attentions]

        run = partial(run_layers, attentions=attentions, weights=weights, packed=packed)
        output = hidden
        for start, end in checkpoint_groups:
            chunk = partial(run, start=start, end=end, checkpointing=True)
            output = checkpoint(chunk, output, use_reentrant=True)
        output = run(output, start=checkpoint_groups[-1][1], end=len(attentions), checkpointing=False)
        expected_keys = {attentions[layer].layer_number for layer in checkpointed_layers} if reuse else set()
        assert set(getattr(packed, "_step5_indexer_selections", {})) == expected_keys
        output.square().sum().backward()
        assert not getattr(packed, "_step5_indexer_selections", {})
        assert [len(calls) for calls in selections] == [
            2 if layer in checkpointed_layers and not reuse else 1 for layer in range(len(attentions))
        ]
        results.append((output, hidden.grad, *(weight.grad for weight in weights)))
    for baseline, cached in zip(*results, strict=True):
        torch.testing.assert_close(cached, baseline, rtol=0, atol=0)


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
    assert not attention.reuse_indexer_on_recompute

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
