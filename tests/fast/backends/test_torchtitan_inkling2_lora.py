import json

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")

from tests.fast.backends.inkling2_tiny import TEXT_CONFIG, as_trained, model_config, random_model  # noqa: E402

from miles.backends.torchtitan_utils.lora import convert_model_config  # noqa: E402
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors  # noqa: E402
from miles.backends.torchtitan_utils.models.inkling2 import layers  # noqa: E402
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets  # noqa: E402

_RANK = 8
_ADAPTED = {
    "attn.wq_du",
    "attn.wk_dv",
    "attn.wv_dv",
    "attn.wr_du",
    "attn.wo_ud",
    "feed_forward.routed_experts.w13",
    "feed_forward.routed_experts.w2",
}
_NUM_LAYERS = TEXT_CONFIG["num_hidden_layers"]
_LOOP = TEXT_CONFIG["loop_config"]


def _eager_relative_attention(q_THD, k_TKD, v_TKD, rel_THE, docs, *, window, scale):
    """``layers.relative_attention`` without flex attention, whose log-sum-exp the model reads only on GPU."""
    tokens, heads, _ = q_THD.shape
    k = k_TKD.float().repeat_interleave(heads // k_TKD.shape[1], dim=1)
    v = v_TKD.float().repeat_interleave(heads // v_TKD.shape[1], dim=1)
    scores_HTS = torch.einsum("thd,shd->hts", q_THD.float(), k) * scale
    distance = torch.arange(tokens)[:, None] - torch.arange(tokens)[None, :]
    extent = rel_THE.shape[-1]
    bias = torch.gather(rel_THE.float().permute(1, 0, 2), 2, distance.clamp(0, extent - 1).expand(heads, -1, -1))
    scores_HTS = torch.where((distance >= 0) & (distance < extent), scores_HTS + bias, scores_HTS)
    keep = (distance >= 0) & (docs.doc_ids[:, None] == docs.doc_ids[None, :])
    if window is not None:
        keep = keep & (distance <= window)
    scores_HTS = scores_HTS.masked_fill(~keep, float("-inf"))
    out = torch.einsum("hts,shd->thd", torch.softmax(scores_HTS, dim=-1), v)
    return out, torch.logsumexp(scores_HTS, dim=-1).permute(1, 0)


@pytest.fixture
def eager_attention(monkeypatch):
    monkeypatch.setattr(layers, "relative_attention", _eager_relative_attention)


def _targets(directory) -> list[str]:
    return resolve_hf_lora_targets(json.loads((directory / "config.json").read_text()))


def _lora_config(directory):
    config = model_config(directory)
    return convert_model_config(config, hf_targets=_targets(directory), rank=_RANK, alpha=16)


def _base_and_lora(directory):
    """The tiny model and its LoRA twin on the same base weights, as trained (bf16), returning the final hidden."""
    _, base = random_model(directory)
    lora = _lora_config(directory).build()
    lora.init_states()
    missing, unexpected = lora.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(".lora_" in key for key in missing)
    base, lora = as_trained(base), as_trained(lora.eval())
    # the fp32-output lm_head GEMM has no CPU kernel; the hidden state already went through every layer
    base._skip_lm_head = lora._skip_lm_head = True
    return base, lora


def _packed_batch():
    torch.manual_seed(1)
    positions = torch.cat([torch.arange(12), torch.arange(9)])
    return torch.randint(0, TEXT_CONFIG["unpadded_vocab_size"], (positions.numel(),)), positions


def test_attention_and_routed_expert_targets_adapt_exactly_those_projections(tmp_path):
    with torch.device("meta"):
        model = _lora_config(tmp_path).build()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    expected = {
        f"layers.{layer}.{projection}.lora_{factor}.weight"
        for layer in range(_NUM_LAYERS)
        for projection in _ADAPTED
        for factor in "ab"
    }
    assert trainable == expected
    # the loop's message controllers, the output gate and the sink logits keep no adapter
    assert not any(".controllers." in name or "gate_attn" in name or "attn_sink_proj" in name for name in trainable)


def test_routed_experts_get_their_own_factors_per_expert(tmp_path):
    with torch.device("meta"):
        model = _lora_config(tmp_path).build()
    experts, hidden, dim = (
        TEXT_CONFIG["n_routed_experts"],
        TEXT_CONFIG["intermediate_size"],
        TEXT_CONFIG["hidden_size"],
    )
    shapes = {name: tuple(param.shape) for name, param in model.named_parameters() if param.requires_grad}
    prefix = "layers.3.feed_forward.routed_experts"
    assert shapes[f"{prefix}.w13.lora_a.weight"] == (experts, _RANK, dim)
    assert shapes[f"{prefix}.w13.lora_b.weight"] == (experts, 2, hidden, _RANK)
    assert shapes[f"{prefix}.w2.lora_a.weight"] == (experts, _RANK, hidden)
    assert shapes[f"{prefix}.w2.lora_b.weight"] == (experts, dim, _RANK)


def test_at_init_the_adapted_model_matches_the_base_model(tmp_path, eager_attention):
    base, lora = _base_and_lora(tmp_path)
    tokens, positions = _packed_batch()
    with torch.no_grad():
        assert torch.equal(lora(tokens, positions), base(tokens, positions))


def test_a_step_moves_only_the_adapters(tmp_path, eager_attention):
    _, lora = _base_and_lora(tmp_path)
    lora.train()
    frozen = {name: param.detach().clone() for name, param in lora.named_parameters() if not param.requires_grad}
    tokens, positions = _packed_batch()
    lora(tokens, positions).float().square().mean().backward()
    assert {name for name, param in lora.named_parameters() if param.grad is not None} == {
        name for name, param in lora.named_parameters() if param.requires_grad
    }
    # lora_B starts at zero, so only it has a gradient at the first step; every adapted layer gets one
    assert all(param.grad.abs().sum() > 0 for name, param in lora.named_parameters() if name.endswith("lora_b.weight"))
    torch.optim.SGD([param for param in lora.parameters() if param.requires_grad], lr=1.0).step()
    assert all(torch.equal(param, frozen[name]) for name, param in lora.named_parameters() if name in frozen)
    assert any(param.abs().sum() > 0 for name, param in lora.named_parameters() if name.endswith("lora_b.weight"))


def test_every_loop_pass_runs_the_shared_adapters(tmp_path, eager_attention, monkeypatch):
    _, lora = _base_and_lora(tmp_path)
    calls: dict[int, int] = {}
    for layer_id, block in lora.layers.items():
        adapter = block.attn.wq_du.lora_a
        original = adapter._linear

        def counted(*args, layer_id=int(layer_id), original=original):
            calls[layer_id] = calls.get(layer_id, 0) + 1
            return original(*args)

        monkeypatch.setattr(adapter, "_linear", counted)
    tokens, positions = _packed_batch()
    with torch.no_grad():
        lora(tokens, positions)
    core = range(_LOOP["core_start"], _LOOP["core_end"])
    assert calls == {layer: _LOOP["num_passes"] if layer in core else 1 for layer in range(_NUM_LAYERS)}


def test_activation_checkpointing_keeps_the_adapter_gradients(tmp_path, eager_attention):
    """Each core layer runs once per pass inside its own checkpoint; the recomputed backward sums the same
    per-pass gradients into the shared adapters as the stored one."""
    _, lora = _base_and_lora(tmp_path)
    lora.train()
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if name.endswith("lora_b.weight"):
                param.normal_(std=0.05)
    tokens, positions = _packed_batch()
    grads = []
    for activation_checkpoint in (False, True):
        lora.activation_checkpoint = activation_checkpoint
        lora.zero_grad()
        lora(tokens, positions).float().square().mean().backward()
        grads.append({name: param.grad.clone() for name, param in lora.named_parameters() if param.requires_grad})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name])
    core_layer = f"layers.{_LOOP['core_start']}.attn.wq_du"
    assert grads[0][f"{core_layer}.lora_a.weight"].abs().sum() > 0


def test_adapters_export_under_the_checkpoints_hf_modules(tmp_path):
    config = model_config(tmp_path)
    sd_adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    lora = _lora_config(tmp_path).build()
    with torch.no_grad():
        for param in lora.parameters():
            param.normal_()
    state = lora.state_dict()
    exported = dict(iter_hf_adapter_tensors(sd_adapter, state))

    hf_modules = (
        "attn.wq_du",
        "attn.wk_dv",
        "attn.wv_dv",
        "attn.wr_du",
        "attn.wo_ud",
        "mlp.experts.w13",
        "mlp.experts.w2",
    )
    assert set(exported) == {
        f"model.llm.layers.{layer}.{module}.lora_{factor}.weight"
        for layer in range(_NUM_LAYERS)
        for module in hf_modules
        for factor in "AB"
    }
    assert exported["model.llm.layers.2.attn.wk_dv.lora_A.weight"] is state["layers.2.attn.wk_dv.lora_a.weight"]
    assert exported["model.llm.layers.2.attn.wk_dv.lora_B.weight"] is state["layers.2.attn.wk_dv.lora_b.weight"]
    # gate/up lora_B rows interleave as the checkpoint's w13_weight does; the shared lora_A exports as is
    titan_b = state["layers.2.feed_forward.routed_experts.w13.lora_b.weight"]
    hf_b = exported["model.llm.layers.2.mlp.experts.w13.lora_B.weight"]
    assert torch.equal(hf_b[:, 0::2], titan_b[:, 0]) and torch.equal(hf_b[:, 1::2], titan_b[:, 1])
    assert torch.equal(
        exported["model.llm.layers.2.mlp.experts.w13.lora_A.weight"],
        state["layers.2.feed_forward.routed_experts.w13.lora_a.weight"],
    )
    assert torch.equal(
        exported["model.llm.layers.2.mlp.experts.w2.lora_B.weight"],
        state["layers.2.feed_forward.routed_experts.w2.lora_b.weight"],
    )


def test_merging_the_exported_adapter_reproduces_the_adapted_forward(tmp_path, eager_attention):
    """The fallback sync while SGLang serves no Inkling 2 LoRA: W + (alpha / r) B @ A in the checkpoint layout,
    loaded into the plain model, matches the adapted model."""
    base, lora = _base_and_lora(tmp_path)
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if name.endswith("lora_b.weight"):
                param.normal_(std=0.05)
    config = model_config(tmp_path)
    sd_adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    adapter = dict(iter_hf_adapter_tensors(sd_adapter, lora.state_dict()))
    merged = sd_adapter.to_hf({k: v for k, v in base.state_dict().items() if "tokens_per_expert" not in k})
    scale = 16 / _RANK
    for name in adapter:
        if ".lora_A." not in name:
            continue
        module = name.removesuffix(".lora_A.weight")
        weight = f"{module}_weight" if ".experts." in module else f"{module}.weight"
        delta = adapter[f"{module}.lora_B.weight"].float() @ adapter[name].float()
        merged[weight] = (merged[weight].float() + scale * delta).to(merged[weight].dtype)
    tokens, positions = _packed_batch()
    with torch.no_grad():
        adapted = lora(tokens, positions)
        assert not torch.allclose(base(tokens, positions), adapted, atol=0.05, rtol=0.05)
        base.load_state_dict(sd_adapter.from_hf(merged), strict=False)
        torch.testing.assert_close(base(tokens, positions), adapted, atol=0.05, rtol=0.05)


def test_the_unembedding_is_not_adaptable(tmp_path):
    with pytest.raises(ValueError, match="name no projection"):
        convert_model_config(model_config(tmp_path), hf_targets=["model.llm.unembed"], rank=_RANK, alpha=16)
