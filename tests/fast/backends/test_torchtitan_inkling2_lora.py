import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")

from tests.fast.backends.inkling2_tiny import TEXT_CONFIG, as_trained, model_config, random_model  # noqa: E402

from miles.backends.torchtitan_utils import hf_weight_iterator  # noqa: E402
from miles.backends.torchtitan_utils.lora import convert_model_config  # noqa: E402
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors  # noqa: E402
from miles.backends.torchtitan_utils.models.inkling2 import layers  # noqa: E402
from miles.backends.torchtitan_utils.models.inkling2 import model as inkling2_model  # noqa: E402
from miles.utils import distributed_utils  # noqa: E402
from miles.utils.hf_utils.config import load_hf_config  # noqa: E402
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets  # noqa: E402

_RANK, _ALPHA = 8, 16
_ATTENTION = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud")
_ADAPTED = {
    *(f"layers.{layer}.attn.{p}" for layer in range(TEXT_CONFIG["num_hidden_layers"]) for p in _ATTENTION),
    *(
        f"layers.{layer}.feed_forward.routed_experts.{p}"
        for layer in range(TEXT_CONFIG["num_hidden_layers"])
        for p in ("w13", "w2")
    ),
    "lm_head",
}
_NUM_LAYERS = TEXT_CONFIG["num_hidden_layers"]
_LOOP = TEXT_CONFIG["loop_config"]
_E, _F, _D = TEXT_CONFIG["n_routed_experts"], TEXT_CONFIG["intermediate_size"], TEXT_CONFIG["hidden_size"]


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


class _CpuFp32OutMatmul:
    """``torch.mm(..., out_dtype=float32)`` has no CPU kernel: the same product, computed in fp32."""

    @staticmethod
    def apply(x_TD, w_VD):
        return x_TD.float() @ w_VD.float().t()


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(layers, "relative_attention", _eager_relative_attention)
    monkeypatch.setattr(inkling2_model, "_Fp32OutMatmul", _CpuFp32OutMatmul)


def _targets(directory) -> list[str]:
    return resolve_hf_lora_targets(json.loads((directory / "config.json").read_text()))


def _lora_config(directory, **overrides):
    config = model_config(directory)
    kwargs = dict(hf_targets=_targets(directory), rank=_RANK, alpha=_ALPHA, experts_shared_outer=True) | overrides
    return convert_model_config(config, **kwargs)


def _base_and_lora(directory, *, random_lora_b: bool = False):
    """The tiny model and its LoRA twin on the same base weights, as trained (bf16)."""
    _, base = random_model(directory)
    lora = _lora_config(directory).build()
    lora.init_states()
    missing, unexpected = lora.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(".lora_" in key for key in missing)
    if random_lora_b:
        with torch.no_grad():
            for name, param in lora.named_parameters():
                if name.endswith("lora_b.weight"):
                    param.normal_(std=0.05)
    return as_trained(base), as_trained(lora.eval())


def _packed_batch():
    torch.manual_seed(1)
    positions = torch.cat([torch.arange(12), torch.arange(9)])
    return torch.randint(0, TEXT_CONFIG["unpadded_vocab_size"], (positions.numel(),)), positions


def test_the_default_targets_adapt_the_engine_schema_and_nothing_else(tmp_path):
    with torch.device("meta"):
        model = _lora_config(tmp_path).build()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {f"{projection}.lora_{factor}.weight" for projection in _ADAPTED for factor in "ab"}
    assert not any(".controllers." in name or "attn_sink_proj" in name for name in trainable)


def test_routed_experts_are_shared_outer(tmp_path):
    with torch.device("meta"):
        model = _lora_config(tmp_path).build()
    shapes = {name: tuple(param.shape) for name, param in model.named_parameters() if param.requires_grad}
    prefix = "layers.3.feed_forward.routed_experts"
    # gate/up: one lora_A per branch for every expert; down: one lora_B for every expert
    assert shapes[f"{prefix}.w13.lora_a.weight"] == (2, _RANK, _D)
    assert shapes[f"{prefix}.w13.lora_b.weight"] == (_E, 2, _F, _RANK)
    assert shapes[f"{prefix}.w2.lora_a.weight"] == (_E, _RANK, _F)
    assert shapes[f"{prefix}.w2.lora_b.weight"] == (1, _D, _RANK)


def test_expert_lora_needs_the_shared_outer_flag_and_a_rank_of_eight(tmp_path):
    with pytest.raises(ValueError, match="--experts-shared-outer-loras"):
        _lora_config(tmp_path, experts_shared_outer=False)
    with pytest.raises(ValueError, match="divisible by 8"):
        _lora_config(tmp_path, rank=4)
    attention = resolve_hf_lora_targets(json.loads((tmp_path / "config.json").read_text()), target_modules=["attn"])
    # attention alone needs neither
    _lora_config(tmp_path, hf_targets=attention, rank=4, experts_shared_outer=False)


def test_the_frozen_base_is_stored_in_bf16_and_the_adapters_in_fp32(tmp_path):
    with torch.device("meta"):
        model = _lora_config(tmp_path).build()
    dtypes = {name: param.dtype for name, param in model.named_parameters()}
    assert all(dtype == torch.float32 for name, dtype in dtypes.items() if ".lora_" in name)
    for name in (
        "tok_embeddings.weight",
        "lm_head.weight",
        "layers.2.attn.wq_du.weight",
        "layers.2.attn.attn_sink_proj.weight",
        "layers.2.feed_forward.router.gate.weight",
        "layers.2.feed_forward.routed_experts.w13.weight",
    ):
        assert dtypes[name] == torch.bfloat16, name
    # norms, convolutions, the fp32-policy residual queries and the loop controllers keep their fp32 storage
    for name in ("layers.2.mlp_norm.weight", "layers.2.mlp_query.query", "loop_queries.queries"):
        assert dtypes[name] == torch.float32, name
    with torch.device("meta"):
        base = model_config(tmp_path).build()
    assert all(param.dtype == torch.float32 for param in base.parameters())


def test_at_init_the_adapted_model_matches_the_base_model(tmp_path, cpu_kernels):
    base, lora = _base_and_lora(tmp_path)
    tokens, positions = _packed_batch()
    with torch.no_grad():
        assert torch.equal(lora(tokens, positions), base(tokens, positions))


def test_a_step_moves_only_the_adapters(tmp_path, cpu_kernels):
    _, lora = _base_and_lora(tmp_path)
    lora.train()
    frozen = {name: param.detach().clone() for name, param in lora.named_parameters() if not param.requires_grad}
    tokens, positions = _packed_batch()
    lora(tokens, positions).float().square().mean().backward()
    assert {name for name, param in lora.named_parameters() if param.grad is not None} == {
        name for name, param in lora.named_parameters() if param.requires_grad
    }
    # lora_B starts at zero, so only it has a gradient at the first step; every adapted projection gets one
    assert all(param.grad.abs().sum() > 0 for name, param in lora.named_parameters() if name.endswith("lora_b.weight"))
    torch.optim.SGD([param for param in lora.parameters() if param.requires_grad], lr=1.0).step()
    assert all(torch.equal(param, frozen[name]) for name, param in lora.named_parameters() if name in frozen)


def test_every_loop_pass_runs_the_shared_adapters(tmp_path, cpu_kernels, monkeypatch):
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


def test_activation_checkpointing_keeps_the_adapter_gradients(tmp_path, cpu_kernels):
    """Each core layer runs once per pass inside its own checkpoint; the recomputed backward sums the same
    per-pass gradients into the shared adapters as the stored one."""
    _, lora = _base_and_lora(tmp_path, random_lora_b=True)
    lora.train()
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
    assert grads[0][f"layers.{_LOOP['core_start']}.attn.wq_du.lora_a.weight"].abs().sum() > 0


def test_adapters_export_in_the_engine_schema(tmp_path):
    config = model_config(tmp_path)
    sd_adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    lora = _lora_config(tmp_path).build()
    with torch.no_grad():
        for param in lora.parameters():
            param.normal_()
    state = lora.state_dict()
    exported = dict(iter_hf_adapter_tensors(sd_adapter, state))

    modules = [f"attn.{p}" for p in _ATTENTION] + [f"mlp.experts.{leaf}" for leaf in ("w1", "w3", "w2")]
    assert set(exported) == {
        f"model.llm.layers.{layer}.{module}.lora_{factor}.weight"
        for layer in range(_NUM_LAYERS)
        for module in modules
        for factor in "AB"
    } | {"model.llm.unembed.lora_A.weight", "model.llm.unembed.lora_B.weight"}
    experts = "model.llm.layers.2.mlp.experts"
    titan = "layers.2.feed_forward.routed_experts"
    for index, leaf in enumerate(("w1", "w3")):
        assert torch.equal(
            exported[f"{experts}.{leaf}.lora_A.weight"], state[f"{titan}.w13.lora_a.weight"][index : index + 1]
        )
        assert torch.equal(exported[f"{experts}.{leaf}.lora_B.weight"], state[f"{titan}.w13.lora_b.weight"][:, index])
    assert exported[f"{experts}.w2.lora_A.weight"].shape == (_E, _RANK, _F)
    assert exported[f"{experts}.w2.lora_B.weight"].shape == (1, _D, _RANK)
    # every padded-vocab row of the unembed
    assert exported["model.llm.unembed.lora_B.weight"].shape == (TEXT_CONFIG["vocab_size"], _RANK)


@pytest.fixture
def one_rank_group(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_weight_iterator.current_platform, "current_device", lambda: torch.device("cpu"))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.new_group(backend="gloo"))
    yield
    dist.destroy_process_group()


def test_the_weight_sync_carries_the_adapters_merged_into_the_base(tmp_path, cpu_kernels, one_rank_group):
    """The sync of a family whose engine serves no adapters (Inkling 2 Mini): the HF tensors the engine loads give
    the adapted model's logits, and the trainer's stored base is left as it was."""
    base, lora = _base_and_lora(tmp_path, random_lora_b=True)
    sd_adapter = model_config(tmp_path)._owner.state_dict_adapter_cls(model_config(tmp_path), str(tmp_path))
    trainer = SimpleNamespace(
        model_parts=[lora],
        engine=SimpleNamespace(state_dict_adapter=sd_adapter),
        parallelism=SimpleNamespace(get_optional_mesh=lambda name: None),
        device=torch.device("cpu"),
    )
    stored = {k: v.clone() for k, v in lora.state_dict().items()}
    synced = dict(
        hf_weight_iterator._hf_weights_on_device(
            trainer, complete_across_pp=True, local_experts=False, merge_lora_scale=_ALPHA / _RANK
        )
    )
    assert not any(".lora_" in name for name in synced)
    assert all(torch.equal(v, stored[k]) for k, v in lora.state_dict().items())
    tokens, positions = _packed_batch()
    with torch.no_grad():
        adapted = lora(tokens, positions)
        assert not torch.allclose(base(tokens, positions), adapted, atol=0.05, rtol=0.05)
        base.load_state_dict(sd_adapter.from_hf(synced), strict=False)
        torch.testing.assert_close(base(tokens, positions), adapted, atol=0.05, rtol=0.05)


def test_only_mini_with_engines_off_adapters_syncs_merged_weights(tmp_path):
    from tests.fast.backends.inkling2_tiny import write_checkpoint_config

    mini = load_hf_config(write_checkpoint_config(tmp_path / "mini"))
    small = load_hf_config(write_checkpoint_config(tmp_path / "small", model_type="inkling2"))
    args = dict(lora_rank=8, lora_alpha=16.0, lora_adapter_path=None, lora_train_only=True)
    assert hf_weight_iterator.merge_lora_scale(Namespace(**args), mini) == 2.0
    assert hf_weight_iterator.merge_lora_scale(Namespace(**args), small) is None
    assert hf_weight_iterator.merge_lora_scale(Namespace(**args | {"lora_train_only": False}), mini) is None
    assert hf_weight_iterator.merge_lora_scale(Namespace(**args | {"lora_rank": 0}), mini) is None


def test_fsdp_trains_the_adapters_over_a_bf16_frozen_base(tmp_path, cpu_kernels, one_rank_group):
    """FSDP2 with the model's mixed-precision policies (bf16, the residual queries fp32) takes groups that mix the
    frozen bf16 base with fp32 adapters and norms, and computes what the plain bf16 model does."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    _, reference = _base_and_lora(tmp_path, random_lora_b=True)
    lora = _lora_config(tmp_path).build()
    lora.load_state_dict(reference.state_dict())
    assert {p.dtype for p in lora.parameters() if not p.requires_grad} == {torch.bfloat16, torch.float32}
    mesh = init_device_mesh("cpu", (1,))
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
    queries = [lora.output_query, lora.loop_queries]
    queries += [q for block in lora.layers.values() for q in (block.mlp_query, block.attn_query) if q is not None]
    for module in queries:
        fully_shard(module, mesh=mesh, mp_policy=fp32)
    for block in lora.layers.values():
        fully_shard(block, mesh=mesh, mp_policy=bf16)
    fully_shard(lora, mesh=mesh, mp_policy=bf16)

    tokens, positions = _packed_batch()
    logits = lora(tokens, positions)
    logits.float().square().mean().backward()
    assert all(param.grad is not None for param in lora.parameters() if param.requires_grad)
    with torch.no_grad():
        torch.testing.assert_close(logits, reference(tokens, positions), atol=0.05, rtol=0.05)


def test_the_adapter_checkpoint_layout_probes_the_engine_schema(tmp_path, one_rank_group):
    """--save and --lora-adapter-path name each adapter element through the same export."""
    from miles.backends.torchtitan_utils.lora_checkpoint import probe_torchtitan_row_mapping
    from miles.backends.torchtitan_utils.lora_names import is_adapter_key

    config = model_config(tmp_path)
    sd_adapter = config._owner.state_dict_adapter_cls(config, str(tmp_path))
    lora = _lora_config(tmp_path).build()
    params = {key: param for key, param in lora.named_parameters() if is_adapter_key(key)}
    mapping = probe_torchtitan_row_mapping(params, sd_adapter)
    shapes = {piece.key: piece.global_shape for piece in mapping.pieces}
    assert shapes["model.llm.layers.2.mlp.experts.w1.lora_A"] == (1, _RANK, _D)
    assert shapes["model.llm.layers.2.mlp.experts.w3.lora_B"] == (_E, _F, _RANK)
    assert shapes["model.llm.layers.2.mlp.experts.w2.lora_B"] == (1, _D, _RANK)
    assert shapes["model.llm.unembed.lora_B"] == (TEXT_CONFIG["vocab_size"], _RANK)
    assert sum(piece.local_elements.numel() for piece in mapping.pieces) == sum(p.numel() for p in params.values())
