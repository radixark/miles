"""DeepSeek-V4.1 TOP family composition: what the block config and the export refuse instead of running another
program or sending a tensor under a name serving does not hold."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.top.dsv41_export import export_dsv41, reference_grads
from miles_plugins.top.dsv41_spec import validate_dsv41_config


def _config(**overrides):
    fields = dict(
        enable_hyper_connections=True,
        num_residual_streams=4,
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        moe_grouped_gemm=False,
        moe_n_hash_layers=0,
        mtp_num_layers=None,
        recompute_granularity=None,
    )
    return SimpleNamespace(**{**fields, **overrides})


def test_the_pinned_block_config_is_admitted():
    validate_dsv41_config(_config())


def test_activation_recompute_is_refused():
    """A recomputed V4.1 layer would read the next layers' pre and selections, which live outside its hidden state."""
    with pytest.raises(NotImplementedError, match="not recomputed"):
        validate_dsv41_config(_config(recompute_granularity="full"))


def test_draft_layers_in_the_trainer_are_refused():
    """DSpark is serving-only; trained MTP layers would be a program serving does not run."""
    with pytest.raises(NotImplementedError, match="trains no draft layers"):
        validate_dsv41_config(_config(mtp_num_layers=1))


@pytest.mark.parametrize(
    "field, value",
    [("moe_grouped_gemm", True), ("moe_n_hash_layers", 3), ("moe_flex_dispatcher_backend", "hybridep")],
)
def test_a_config_off_the_pinned_program_is_refused(field, value):
    with pytest.raises(NotImplementedError, match=f"requires {field}="):
        validate_dsv41_config(_config(**{field: value}))


def test_an_unnamed_parameter_is_refused_by_the_export():
    """Every V4.1 name is owned by the export table; an unknown parameter is not sent under a guessed name."""
    with pytest.raises(NotImplementedError, match="no export"):
        export_dsv41(
            None, "deepseekv41", "module.module.decoder.layers.0.self_attention.new_proj.weight", None, None, None
        )
    with pytest.raises(NotImplementedError, match="no export"):
        reference_grads("module.module.mtp.layers.0.weight", torch.zeros(1))


def test_reference_gradients_take_the_names_weight_sync_sends():
    """The VJP's reference names a gradient as weight sync names the parameter's bytes (scales aside)."""
    layer = "module.module.decoder.layers.3."
    params = {
        layer + "self_attention.wq_a_linear.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        layer + "mlp.shared_experts.linear_fc1.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        layer + "input_layernorm.weight": torch.randn(64, dtype=torch.bfloat16),
        layer + "self_attention_hyper_connection.mapping_proj.weight": torch.randn(24, 256),
    }
    for name, param in params.items():
        synced = {n for n, _ in export_dsv41(None, "deepseekv41", name, param, None, None) if not n.endswith(".scale")}
        assert {n for n, _ in reference_grads(name, param)} == synced


def _fp8_b32(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    payload = (torch.randn(n, k, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload, torch.full((-(-n // 32), -(-k // 32)), 2.0**-10)


def test_frozen_indexers_travel_in_the_sync_form_with_the_served_values():
    """A colocated engine gets back only what is sent: the untrained indexer goes out on every sync, wq_b in the
    sync quantizer's form (no raw checkpoint scale) with the values serving loaded."""
    from miles.backends.megatron_utils.megatron_to_hf.export_hooks import export_hooks
    from miles_plugins.top import dsv41_export
    from miles_plugins.top.fp8_b32 import dequantize_b32

    payload, scale = _fp8_b32(64, 96)
    norm = torch.randn(128, dtype=torch.bfloat16)
    p = "layers.2.attn.indexer."
    sent = dict(
        dsv41_export.convert_frozen_unit([(p + "wq_b.weight", payload), (p + "wq_b.scale", scale)])
        + dsv41_export.convert_frozen_unit([(p + "k_norm.weight", norm)])
    )
    assert set(sent) == {p + "wq_b.weight", p + "wq_b.scale", p + "k_norm.weight"}
    assert torch.equal(dequantize_b32(sent[p + "wq_b.weight"], sent[p + "wq_b.scale"]), dequantize_b32(payload, scale))
    assert torch.equal(sent[p + "k_norm.weight"], norm)
    hooks = export_hooks()
    saved = (hooks.converter, hooks.checkpoint_units, hooks.frozen_indexer)
    try:
        dsv41_export.register()
        assert hooks.checkpoint_units is dsv41_export.checkpoint_units
    finally:
        hooks.converter, hooks.checkpoint_units, hooks.frozen_indexer = saved


def test_reference_weights_are_served_or_masters():
    from tools.convert_dsv41_to_bf16_reference import master, served

    from miles_plugins.top.fp8_b32 import dequantize_b32

    payload, scale = _fp8_b32(64, 128)
    value = served(payload, scale)
    assert torch.equal(value, dequantize_b32(payload, scale))
    # The masters sit off the served values; the sync quantizer still returns the release's codes for them.
    assert not torch.equal(master(payload, scale), value)


def test_kv_quantizers_round_only_inside_kv_quantization():
    from miles.utils.debug_utils import top_reference_dsv41 as ref

    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 128, generator=g)
    kept = x.clone()
    ref._act_quant(kept, 64, "ue8m0", torch.float32, True)
    assert torch.equal(kept, x)
    leaf = x.clone().requires_grad_()
    rows = leaf * 1.0
    with ref.kv_quantization(None):
        ref._act_quant(rows, 64, "ue8m0", torch.float32, True)
    assert not torch.equal(rows, x)
    rows.backward(torch.full_like(rows, 2.0))
    assert torch.equal(leaf.grad, torch.full_like(x, 2.0))
