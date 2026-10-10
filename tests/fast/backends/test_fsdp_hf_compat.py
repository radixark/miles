"""Unit tests for the experimental-FSDP HF-compat fixes (CPU-only, no GPU/sglang).

Covers:
  * F9 weight-sync: batched MoE expert params are unfused through transformers' own
    reverse conversion (``revert_weight_conversion``) into the per-expert names
    SGLang expects, with the correct gate/up row split, contiguous tensors, and
    only for the right model types. These tests pin the HF revert output to the
    on-disk dialect; any upstream drift in the qwen2_moe conversion family or in
    per-tensor revert semantics turns them red.
"""

import itertools
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils import hf_weight_iterator
from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.specs import _ADAPTERS, resolve_arch_adapter
from miles.backends.fsdp_utils.adaptations.weight_bridge import unfuse_batched_experts


def _adapter(model_type):
    return resolve_arch_adapter(SimpleNamespace(model_type=model_type))


def _packed_boundaries(*lengths):
    """The boundary fields `get_batch` records for documents of these lengths."""
    cu_seqlens_host = tuple(itertools.accumulate(lengths, initial=0))
    return dict(
        cu_seqlens=torch.tensor(cu_seqlens_host, dtype=torch.int32),
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max(lengths),
    )


@pytest.fixture(scope="module")
def tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        decoder_sparse_step=1,
        head_dim=4,
    )
    return Qwen3MoeForCausalLM(cfg)


def test_unfuse_gate_up_proj_rows_and_names(tiny_qwen3_moe):
    # [E=2, 2*inter=6, H=4]: fused rows are [gate(:3) | up(3:)]
    E, inter, H = 2, 3, 4
    full = torch.arange(E * 2 * inter * H, dtype=torch.float32).reshape(E, 2 * inter, H)
    out = dict(unfuse_batched_experts("model.layers.0.mlp.experts.gate_up_proj", full, tiny_qwen3_moe))

    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    for e in range(E):
        g = out[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        u = out[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
        assert g.shape == (inter, H) and u.shape == (inter, H)
        torch.testing.assert_close(g, full[e, :inter, :])
        torch.testing.assert_close(u, full[e, inter:, :])
        assert g.is_contiguous() and u.is_contiguous()


def test_unfuse_down_proj(tiny_qwen3_moe):
    E, H, inter = 2, 4, 3
    full = torch.arange(E * H * inter, dtype=torch.float32).reshape(E, H, inter)
    out = dict(unfuse_batched_experts("model.layers.5.mlp.experts.down_proj", full, tiny_qwen3_moe))
    assert set(out) == {
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.1.down_proj.weight",
    }
    for e in range(E):
        d = out[f"model.layers.5.mlp.experts.{e}.down_proj.weight"]
        assert d.shape == (H, inter)
        torch.testing.assert_close(d, full[e])
        assert d.is_contiguous()


def test_unfuse_glm4_moe_lite_same_family():
    # glm4_moe_lite shares qwen3_moe's conversion family (qwen2_moe); its real batched
    # params must unfuse to the same per-expert dialect.
    from transformers import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

    cfg = Glm4MoeLiteConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        first_k_dense_replace=1,
        n_shared_experts=1,
        head_dim=4,
    )
    model = Glm4MoeLiteForCausalLM(cfg)
    name = "model.layers.1.mlp.experts.gate_up_proj"
    full = model.state_dict()[name]
    E, two_inter = full.shape[0], full.shape[1]
    out = dict(unfuse_batched_experts(name, full, model))
    assert set(out) == {
        f"model.layers.1.mlp.experts.{e}.{proj}.weight" for e in range(E) for proj in ("gate_proj", "up_proj")
    }
    for e in range(E):
        torch.testing.assert_close(
            out[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"], full[e, : two_inter // 2, :]
        )
        torch.testing.assert_close(out[f"model.layers.1.mlp.experts.{e}.up_proj.weight"], full[e, two_inter // 2 :, :])


def test_param_transform_gating():
    def applies(name, param, model_type):
        return _adapter(model_type).param_transform(name, param) is not None

    gate_up = torch.zeros(2, 6, 4)
    name = "model.layers.0.mlp.experts.gate_up_proj"
    # only for model types whose SGLang loader expects per-expert weights
    assert applies(name, gate_up, "qwen3_moe")
    assert not applies(name, gate_up, "qwen3_5_moe")  # consumes batched directly
    assert not applies(name, gate_up, "qwen3")  # dense
    # non-expert params are never split
    assert not applies("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4), "qwen3_moe")
    # 2D tensor named like an expert param is not the batched layout
    assert not applies(name, torch.zeros(6, 4), "qwen3_moe")


def test_the_iterator_streams_params_without_a_transform_unchanged():
    from types import SimpleNamespace
    from unittest.mock import patch

    embed = torch.zeros(4, 4)
    experts = torch.zeros(2, 6, 4)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        state_dict=lambda: {"model.embed_tokens.weight": embed, "model.layers.0.mlp.experts.gate_up_proj": experts},
    )
    iterator = object.__new__(hf_weight_iterator.FSDPHfWeightIterator)
    iterator.model = model
    iterator.args = SimpleNamespace(update_weight_buffer_size=1 << 30)
    iterator._sync_dtypes = {}
    iterator._arch_adapter = _adapter("qwen3_5_moe")
    with patch.object(hf_weight_iterator, "gather_full_param", lambda t, async_op=False: t):
        units = list(iterator._iter_hf_param_units(None, materialize=True))
    assert [[name for name, _ in unit] for unit in units] == [
        ["model.embed_tokens.weight"],
        ["model.layers.0.mlp.experts.gate_up_proj"],
    ]
    assert units[0][0][1] is embed and units[1][0][1] is experts


def test_every_model_type_has_one_adapter():
    model_types = [model_type for adapter in _ADAPTERS for model_type in adapter.model_types]
    assert len(model_types) == len(set(model_types))


def test_unknown_model_type_runs_the_stock_hf_path():
    adapter = _adapter("some_remote_lm")
    assert type(adapter) is ArchAdapter
    assert not adapter.verified and adapter.routing_replay is None


def test_qwen3_moe_class_patch_is_inert_outside_true_on_policy():
    # Batched experts need no off-mode patch under the pinned transformers version.
    _adapter("qwen3_moe").patch_classes(SimpleNamespace(true_on_policy_mode=False))


def test_validate_hf_config_rejects_fp8_checkpoints():
    from miles.backends.fsdp_utils.adaptations.config_checks import validate_hf_config

    cfg = SimpleNamespace(model_type="qwen3", quantization_config={"quant_method": "fp8"})
    with pytest.raises(ValueError, match="fp8-quantized checkpoint"):
        validate_hf_config(cfg, verified=True, rank=0)


@pytest.mark.parametrize("model_type", ["glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"])
def test_verified_model_types_validate_silently(model_type, caplog):
    from miles.backends.fsdp_utils.adaptations.config_checks import validate_hf_config

    cfg = SimpleNamespace(model_type=model_type)
    validate_hf_config(cfg, verified=_adapter(model_type).verified, rank=0)

    assert not caplog.records


def test_verified_model_types_match_recorded_validation():
    verified = {model_type for adapter in _ADAPTERS if adapter.verified for model_type in adapter.model_types}
    assert verified == {"glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"}


def test_unverified_model_type_warns_once_on_rank_zero(caplog):
    from miles.backends.fsdp_utils.adaptations.config_checks import validate_hf_config

    validate_hf_config(SimpleNamespace(model_type="qwen3_5_moe"), verified=False, rank=0)

    assert len(caplog.records) == 1
    assert "model_type='qwen3_5_moe' has no recorded FSDP validation" in caplog.text


def test_unverified_model_type_is_silent_on_nonzero_rank(caplog):
    from miles.backends.fsdp_utils.adaptations.config_checks import validate_hf_config

    validate_hf_config(SimpleNamespace(model_type="qwen3_5_moe"), verified=False, rank=1)

    assert not caplog.records


def test_packed_seq_context_boundaries():
    # The shared boundary derivation (formerly duplicated verbatim in nemotron_h.py + qwen3_5_moe.py).
    from miles.backends.fsdp_utils.adaptations.packing import packed_seq_context

    # single document / non-packed / wrong shape -> None (packing is a no-op)
    assert packed_seq_context(None) is None
    assert packed_seq_context(torch.arange(8).view(1, 8)) is None  # one doc, never resets to 0
    assert packed_seq_context(torch.arange(8)) is None  # not [1, T]
    assert packed_seq_context(torch.zeros(2, 4, dtype=torch.long)) is None  # batch > 1

    # three packed docs of length 3, 2, 4 -> position_ids reset to 0 at each start
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    ctx = packed_seq_context(pos)
    assert ctx is not None
    assert ctx.cu_seqlens.tolist() == [0, 3, 5, 9]
    assert ctx.cu_seqlens.dtype == torch.int32
    assert ctx.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert ctx.seq_idx.dtype == torch.int32
    assert ctx.seq_idx.shape == (1, 9)
    assert ctx.max_seqlen == 4


def test_nemotron_attention_reuses_precomputed_max_seqlen(monkeypatch):
    from types import ModuleType, SimpleNamespace

    from miles.backends.fsdp_utils.models import nemotron_h

    flash_calls = {}
    flash_attn = ModuleType("flash_attn")

    def flash_attn_varlen_func(q, k, v, **kwargs):
        flash_calls.update(kwargs)
        return q

    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    class UnreadableCuSeqlens:
        def __getitem__(self, key):
            raise AssertionError("attention must not recompute max_seqlen from cu_seqlens")

    cu_seqlens = UnreadableCuSeqlens()
    ctx = SimpleNamespace(cu_seqlens=cu_seqlens, seq_idx=None, max_seqlen=3)
    monkeypatch.setattr(nemotron_h, "packed_seq_context", lambda position_ids: ctx)

    class DummyMixer(torch.nn.Module):
        pass

    class DummyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.q_proj = torch.nn.Identity()
            self.k_proj = torch.nn.Identity()
            self.v_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden_states, *args, **kwargs):
            raise AssertionError("packed attention should use flash_attn_varlen_func")

    class DummyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyAttention()

        def forward(self, hidden_states, position_ids=None):
            return self.attn(hidden_states)

    nemotron_h._patch_attn_forward(DummyAttention)
    nemotron_h._patch_causallm_forward(DummyCausalLM, DummyMixer, DummyAttention)

    model = DummyCausalLM()
    output, _ = model(torch.ones(1, 3, 2), position_ids=torch.zeros(1, 3, dtype=torch.long))

    assert output.shape == (1, 3, 2)
    assert flash_calls["cu_seqlens_q"] is cu_seqlens
    assert flash_calls["max_seqlen_q"] == 3
    assert flash_calls["max_seqlen_k"] == 3


def test_nemotron_pattern_to_list_repair():
    # sglang's import monkeypatches NemotronHConfig._pattern_to_list to drop unmapped chars, deleting
    # every '-' (MLP) layer; the config-time repair must restore the full mapping, and must leave a
    # healthy implementation untouched.
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    from miles.backends.fsdp_utils.adaptations.specs.nemotron_h import _repair_pattern_to_list

    original = NemotronHConfig.__dict__["_pattern_to_list"]
    healthy = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}[c] for c in pattern]
    )
    broken = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention"}[c] for c in pattern if c in "ME*"]
    )
    try:
        NemotronHConfig._pattern_to_list = healthy
        _repair_pattern_to_list()
        assert NemotronHConfig.__dict__["_pattern_to_list"] is healthy

        NemotronHConfig._pattern_to_list = broken
        _repair_pattern_to_list()
        assert NemotronHConfig._pattern_to_list("M-*E") == ["mamba", "mlp", "attention", "moe"]
    finally:
        NemotronHConfig._pattern_to_list = original


def test_hf_packing_kwargs_match_the_padding_free_collator():
    from transformers import DataCollatorWithFlattening

    from miles.backends.fsdp_utils.adaptations.packing import HF_PACKING_KWARG_NAMES, hf_packing_kwargs

    collator = DataCollatorWithFlattening(return_tensors="pt", return_flash_attn_kwargs=True, return_seq_idx=True)
    flattened = collator([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}, {"input_ids": [6, 7, 8, 9]}])
    kwargs = hf_packing_kwargs(**_packed_boundaries(3, 2, 4))

    assert set(kwargs) == HF_PACKING_KWARG_NAMES
    for name, value in kwargs.items():
        expected = flattened[name]
        if isinstance(expected, torch.Tensor):
            assert value.dtype == expected.dtype and torch.equal(value, expected), name
        else:
            assert value == expected, name


def test_only_gated_deltanet_archs_pass_packing_kwargs():
    boundaries = _packed_boundaries(2, 3)
    for model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"):
        assert "seq_idx" in _adapter(model_type).packing_kwargs(**boundaries), model_type
    # NemotronH resets through its own patch; other archs (including remote code) keep their stock inputs.
    for model_type in ("nemotron_h", "glm4_moe_lite", "qwen3", "qwen3_vl", "some_remote_lm"):
        assert _adapter(model_type).packing_kwargs(**boundaries) == {}, model_type


_TINY_QWEN3_5_TEXT = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
)


def test_gated_deltanet_kernels_receive_packed_boundaries():
    # HF's Qwen3.5 forwards the padding-free kwargs to its GatedDeltaNet kernels; no patch in between.
    import torch.nn.functional as F
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig(**_TINY_QWEN3_5_TEXT)
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).eval()
    _adapter("qwen3_5_text").patch_model(model, None)  # text-only: no vision tower to guard
    gdn = model.model.layers[0].linear_attn
    seen = {}

    def conv(x, weight, bias, activation, seq_idx=None):
        seen["seq_idx"] = seq_idx
        return F.silu(F.conv1d(F.pad(x, (weight.shape[-1] - 1, 0)), weight.unsqueeze(1), bias, groups=x.shape[1]))

    chunk = gdn.chunk_gated_delta_rule

    def chunk_rule(*args, cu_seqlens=None, **kwargs):
        seen["cu_seqlens"] = cu_seqlens
        return chunk(*args, **kwargs)

    gdn.causal_conv1d_fn = conv
    gdn.chunk_gated_delta_rule = chunk_rule

    packing_kwargs = _adapter("qwen3_5_text").packing_kwargs(**_packed_boundaries(3, 2))
    with torch.no_grad():
        model(input_ids=torch.arange(5).view(1, 5), position_ids=torch.tensor([[0, 1, 2, 0, 1]]), **packing_kwargs)

    assert seen["cu_seqlens"].tolist() == [0, 3, 5]
    assert seen["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]


def test_qwen3_5_vision_tower_never_sees_language_model_packing_kwargs(monkeypatch):
    """HF hands the language model's kwargs to the vision tower, whose flash attention sets its own
    `cu_seq_lens_q`; without the guard a packed batch with an image raises."""
    import torch.nn.functional as F
    from transformers import AttentionInterface, Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    vision_cu_seqlens = []

    def flash_stand_in(module, query, key, value, attention_mask, scaling=None, cu_seq_lens_q=None, **kwargs):
        vision_cu_seqlens.append(cu_seq_lens_q.tolist())
        return F.scaled_dot_product_attention(query, key, value, scale=scaling).transpose(1, 2), None

    # a "flash" name sends the vision tower down its varlen branch without needing flash-attn on CPU
    monkeypatch.setitem(AttentionInterface._global_mapping, "flash_stand_in", flash_stand_in)
    cfg = Qwen3_5Config(
        text_config=_TINY_QWEN3_5_TEXT,
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=16,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    model.config.text_config._attn_implementation = "eager"
    model.config.vision_config._attn_implementation = "flash_stand_in"

    # doc 0 is text; doc 1 holds one image whose 2x2 patch grid merges into one token
    adapter = _adapter("qwen3_5")
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3, 62, 60, 63, 4]]),
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
        pixel_values=torch.randn(4, 3 * 2 * 2 * 2),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        **adapter.packing_kwargs(**_packed_boundaries(3, 4)),
    )
    with torch.no_grad():
        with pytest.raises(TypeError, match="multiple values for keyword argument 'cu_seq_lens_q'"):
            model(**inputs)

        adapter.patch_model(model, None)
        wrapped = model.model.visual.forward
        adapter.patch_model(model, None)
        assert model.model.visual.forward is wrapped  # idempotent across the actor and ref model
        model(**inputs)

    assert vision_cu_seqlens == [[0, 4]]  # the image's own patch boundaries, not the packed text rows


import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


def _model_cls(**config_fields):
    actor = object.__new__(FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(**config_fields)
    return actor._get_model_cls()


def test_native_vlm_routes_to_image_text_to_text():
    # Qwen3-VL: multimodal, no auto_map, resolves through the transformers registry.
    assert _model_cls(model_type="qwen3_vl", vision_config={"depth": 24}) is AutoModelForImageTextToText


def test_remote_code_multimodal_without_i2t_in_auto_map_routes_to_causal_lm():
    # Kimi-K2.5 ships a vision_config but its remote code only maps AutoModelForCausalLM,
    # so asking for AutoModelForImageTextToText raises "Unrecognized configuration".
    cls = _model_cls(
        model_type="kimi_k25",
        vision_config={"init_pos_emb_height": 64},
        auto_map={
            "AutoConfig": "configuration_kimi.KimiK25Config",
            "AutoModel": "modeling_kimi.KimiK25Model",
            "AutoModelForCausalLM": "modeling_kimi.KimiK25ForCausalLM",
        },
    )
    assert cls is AutoModelForCausalLM


def test_remote_code_multimodal_declaring_i2t_routes_to_image_text_to_text():
    cls = _model_cls(
        model_type="some_remote_vlm",
        vision_config={"depth": 8},
        auto_map={"AutoModelForImageTextToText": "modeling_x.XForConditionalGeneration"},
    )
    assert cls is AutoModelForImageTextToText


def test_text_only_remote_code_routes_to_causal_lm():
    cls = _model_cls(model_type="some_remote_lm", auto_map={"AutoModelForCausalLM": "modeling_x.XForCausalLM"})
    assert cls is AutoModelForCausalLM


def test_native_causal_lm_resolves_concrete_class():
    assert _model_cls(model_type="qwen3") is transformers.Qwen3ForCausalLM
