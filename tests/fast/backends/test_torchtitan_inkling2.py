import re

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")

from tests.fast.backends.inkling2_tiny import TEXT_CONFIG, model_config, random_model  # noqa: E402

_PROJECTIONS = [
    f"attn.{p}.weight" for p in ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj", "wo_ud")
]
_EVERY_LAYER = _PROJECTIONS + [
    "attn.q_l2_clip.scale",
    "attn.rel_l2_clip.scale",
    "attn.rel_logits_proj.proj",
    "attn_sconv.weight",
    "mlp_norm.weight",
    "mlp_sconv.weight",
    "mlp.gate.weight",
    "mlp.gate.bias",
    "mlp.experts.w13_weight",
    "mlp.experts.w2_weight",
]
_LOCAL_LAYER = ["attn_norm.weight", "attn.k_sconv.weight", "attn.v_sconv.weight"]
_GLOBAL_LAYER = [
    "q_input_norm.weight",
    "k_input_norm.weight",
    "v_input_norm.weight",
    *(f"attn.{kv}_sconv.{p}" for kv in "kv" for p in ("static_w", "weight_proj")),
]
_CORE_LAYER = [f"controllers.{m}.{w}_weight" for m in ("attn", "k", "v", "mlp") for w in ("down", "gain", "forget")]
_ROOT = [
    "embed.weight",
    "embed_norm.weight",
    "norm.weight",
    "unembed.weight",
    "attn_res.output_query",
    "middle_cycle_cross_pass_queries",
    "middle_cycle_cross_pass_kv_queries",
]


def _checkpoint_text_keys() -> set[str]:
    """The released BF16 checkpoint's ``model.llm.*`` keys, laid out by layer kind."""
    hf = TEXT_CONFIG
    loop = hf["loop_config"]
    keys = set(_ROOT)
    keys |= {f"attn_res.attn_queries.{i}" for i in range(hf["num_hidden_layers"] - 1)}
    keys |= {f"attn_res.mlp_queries.{i}" for i in range(hf["num_hidden_layers"])}
    for i in range(hf["num_hidden_layers"]):
        suffixes = _EVERY_LAYER + (_LOCAL_LAYER if i in hf["local_layer_ids"] else _GLOBAL_LAYER)
        if loop["core_start"] <= i < loop["core_end"]:
            suffixes = suffixes + _CORE_LAYER
        keys |= {f"layers.{i}.{s}" for s in suffixes}
    return {f"model.llm.{key}" for key in keys}


def _trained_state(model) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "tokens_per_expert" not in k}


def test_the_model_is_sized_from_the_checkpoint_config(tmp_path):
    hf = TEXT_CONFIG
    loop = hf["loop_config"]
    config = model_config(tmp_path)
    assert (config.dim, config.vocab_size, config.unpadded_vocab_size, len(config.layers)) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["unpadded_vocab_size"],
        hf["num_hidden_layers"],
    )
    assert (config.core_start, config.core_end, config.num_passes) == (
        loop["core_start"],
        loop["core_end"],
        loop["num_passes"],
    )
    assert [layer.is_local for layer in config.layers] == [
        i in hf["local_layer_ids"] for i in range(hf["num_hidden_layers"])
    ]
    assert [layer.controller is not None for layer in config.layers] == [
        loop["core_start"] <= i < loop["core_end"] for i in range(hf["num_hidden_layers"])
    ]
    assert config.layers[0].attn_query is None and all(layer.attn_query for layer in config.layers[1:])
    moe = config.layers[0].feed_forward
    assert (moe.num_experts, moe.router.top_k, moe.routed_experts.w2.in_features, moe.shared_experts) == (
        hf["n_routed_experts"],
        hf["num_experts_per_tok"],
        hf["intermediate_size"],
        None,
    )


def test_both_released_model_types_build_the_same_model(tmp_path):
    shapes = {}
    for model_type in ("inkling2_loop", "inkling2"):
        with torch.device("meta"):
            model = model_config(tmp_path / model_type, model_type=model_type).build()
        shapes[model_type] = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    assert shapes["inkling2_loop"] == shapes["inkling2"]


def test_every_trained_tensor_maps_onto_exactly_the_checkpoint_text_keys(tmp_path):
    config = model_config(tmp_path)
    with torch.device("meta"):
        model = config.build()
    adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    assert set(adapter.to_hf(_trained_state(model))) == _checkpoint_text_keys()


def test_the_adapter_round_trips_every_tensor_and_interleaves_gate_and_up_rows(tmp_path):
    config, model = random_model(tmp_path)
    adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    state = _trained_state(model)
    hf = adapter.to_hf(state)
    back = adapter.from_hf(hf)
    assert back.keys() == state.keys()
    assert all(torch.equal(back[key], value) for key, value in state.items())
    w13_titan = state["layers.0.feed_forward.routed_experts.w13.weight"]
    w13_hf = hf["model.llm.layers.0.mlp.experts.w13_weight"]
    assert torch.equal(w13_hf[:, 0::2], w13_titan[:, 0]) and torch.equal(w13_hf[:, 1::2], w13_titan[:, 1])


@pytest.mark.parametrize(
    "text_overrides, match",
    [
        (dict(loop_config={**TEXT_CONFIG["loop_config"], "is_varloop": True}), "is_varloop"),
        (dict(n_shared_experts=1), "n_shared_experts"),
        (dict(model_max_length=128), "exceeds model_max_length"),
    ],
)
def test_a_checkpoint_the_package_cannot_train_as_released_is_refused(tmp_path, text_overrides, match):
    with pytest.raises(ValueError, match=match):
        model_config(tmp_path, **text_overrides)


def test_muon_orthogonalizes_exactly_the_parameters_it_lays_out(tmp_path):
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.models.inkling2 import dist_muon_layouts

    config = model_config(tmp_path)
    pattern, layouts, buckets = dist_muon_layouts(config, ParallelismConfig())
    with torch.device("meta"):
        model = config.build()
    matched = {name for name, _ in model.named_parameters() if re.search(pattern, name)}
    assert matched == set(layouts)
    assert {name for bucket in buckets for name in bucket.patterns} == set(layouts)
    assert "layers.1.controllers.k.down_weight" in matched and "layers.0.controllers.k.down_weight" not in layouts


def test_row_norm_projection_restores_the_norms_recorded_at_the_first_call(tmp_path):
    _, model = random_model(tmp_path)
    model.project_row_norms()
    params = dict(model.named_parameters())
    projected = params["layers.2.attn.wq_du.weight"]
    experts = params["layers.2.feed_forward.routed_experts.w13.weight"]
    untouched = params["layers.2.mlp_norm.weight"]
    before = {"wq": projected.norm(dim=-1), "experts": experts.norm(dim=-1)}
    with torch.no_grad():
        for param in (projected, experts, untouched):
            param.mul_(torch.rand_like(param) + 0.5)
    untouched_after_step = untouched.detach().clone()
    model.project_row_norms()
    torch.testing.assert_close(projected.norm(dim=-1), before["wq"])
    torch.testing.assert_close(experts.norm(dim=-1), before["experts"])
    assert torch.equal(untouched, untouched_after_step)
