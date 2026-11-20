from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import copy
import socket
import sys

import pytest
import torch
import torch.distributed as dist

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3 import build_model_config

from miles.backends.torchtitan_utils.lora import convert_model_config
from miles.backends.torchtitan_utils.lora_checkpoint import TorchtitanAdapterSlots
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.backends.torchtitan_utils.multi_lora import slot_adapter_parameters
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    map_named_tensors_to_hf_keys,
    map_peft_keys_to_hf_keys,
)
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets, resolve_hf_lora_targets

_QWEN3_ALL_LINEAR = resolve_hf_lora_targets({"model_type": "qwen3"})
_QWEN3_MOE_HF_CONFIG = {"model_type": "qwen3_moe", "num_experts": 128, "num_hidden_layers": 48}
_TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 8})
_PROJECTIONS = ("attention.qkv_linear.wqkv", "feed_forward.w13", "attention.wo")
_MAX_RANK = 8


def _qwen3(flavor: str = "debugmodel"):
    return build_model_config(flavor, parallelism=ParallelismConfig(), seq_len=256, attn_backend="flex")


def _layer(config, projection: str):
    module = config.layers[0]
    for step in projection.split("."):
        module = getattr(module, step)
    return module


def _build(projection: str, *, num_slots=None, lora_type="lora", rank=_MAX_RANK):
    converted = convert_model_config(
        _qwen3(), hf_targets=_QWEN3_ALL_LINEAR, rank=rank, alpha=2.0 * rank, num_slots=num_slots, lora_type=lora_type
    )
    torch.manual_seed(0)
    return _layer(converted, projection).build().double()


def _claim(module, slot: int, rank: int, alpha: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, param in module.slot_named_parameters(slot):
            param.copy_(torch.randn(param.shape, generator=g, dtype=param.dtype))
    module.init_slot(slot, rank, alpha)


def test_every_adapted_projection_holds_its_slots_over_a_frozen_base():
    converted = convert_model_config(
        _qwen3(), hf_targets=_QWEN3_ALL_LINEAR, rank=_MAX_RANK, alpha=16, num_slots=3, lora_type="lora"
    )
    for projection in _PROJECTIONS:
        module = _layer(converted, projection).build()
        trainable = {name for name, p in module.named_parameters() if p.requires_grad}
        assert trainable == {f"lora_slots.{s}.lora_{f}.0.weight" for s in range(3) for f in "ab"}, projection
        assert not module.weight.requires_grad
    # the shared A of a fused projection, as LoRATransform's
    w13 = _layer(converted, "feed_forward.w13").build()
    assert w13.lora_slots[0].lora_b[0].weight.shape == (2, w13.out_features, _MAX_RANK)


def test_canonical_gives_each_hf_projection_of_a_fused_one_its_own_adapter():
    converted = convert_model_config(
        _qwen3(), hf_targets=_QWEN3_ALL_LINEAR, rank=_MAX_RANK, alpha=16, num_slots=2, lora_type="canonical_lora"
    )
    qkv, w13, wo = (_layer(converted, p).build() for p in _PROJECTIONS)
    attention = _qwen3().layers[0].attention
    head_dim, n_heads, n_kv = attention.head_dim, attention.n_heads, attention.n_kv_heads
    assert [b.weight.shape[0] for b in qkv.lora_slots[0].lora_b] == [
        n_heads * head_dim,
        n_kv * head_dim,
        n_kv * head_dim,
    ]
    assert [b.weight.shape for b in w13.lora_slots[1].lora_b] == [(w13.out_features, _MAX_RANK)] * 2
    assert len(qkv.lora_slots[0].lora_a) == 3 and len(w13.lora_slots[0].lora_a) == 2
    # an unfused projection has one adapter either way
    assert wo.packing is None and len(wo.lora_slots[0].lora_a) == 1


def test_moe_routed_experts_are_refused_with_slots():
    with pytest.raises(ValueError, match="name no Linear projection"):
        convert_model_config(
            _qwen3("30B-A3B"),
            hf_targets=resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG),
            rank=8,
            alpha=16,
            num_slots=2,
        )


def test_single_lora_cannot_be_canonical():
    with pytest.raises(AssertionError, match="shares one lora_A"):
        convert_model_config(_qwen3(), hf_targets=_QWEN3_ALL_LINEAR, rank=8, alpha=16, lora_type="canonical_lora")


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_one_routed_slot_at_the_max_rank_equals_single_lora_bitwise(projection):
    single = _build(projection).float()
    multi = _build(projection, num_slots=3).float()
    with torch.no_grad():
        multi.weight.copy_(single.weight)
        multi.lora_slots[1].lora_a[0].weight.copy_(torch.randn_like(single.lora_a.weight))
        multi.lora_slots[1].lora_b[0].weight.copy_(torch.randn_like(single.lora_b.weight))
        single.lora_a.weight.copy_(multi.lora_slots[1].lora_a[0].weight)
        single.lora_b.weight.copy_(multi.lora_slots[1].lora_b[0].weight)
    for slot in range(3):
        multi.init_slot(slot, _MAX_RANK, 2.0 * _MAX_RANK)
    x = torch.randn(11, single.in_features)
    multi.token_counts_by_slot = (0, 11, 0)
    x_single, x_multi = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_single, out_multi = single(x_single), multi(x_multi)
    assert torch.equal(out_single, out_multi)
    grad = torch.randn_like(out_single)
    out_single.backward(grad)
    out_multi.backward(grad)
    assert torch.equal(x_single.grad, x_multi.grad)
    assert torch.equal(single.lora_a.weight.grad, multi.lora_slots[1].lora_a[0].weight.grad)
    assert torch.equal(single.lora_b.weight.grad, multi.lora_slots[1].lora_b[0].weight.grad)
    # every slot gets a gradient, so FSDP reduces the same set on every rank
    assert all(p.grad is not None and not p.grad.any() for s in (0, 2) for p in multi.lora_slots[s].parameters())


@pytest.mark.parametrize("lora_type", ["lora", "canonical_lora"])
@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_each_slot_runs_on_its_token_range(projection, lora_type):
    multi = _build(projection, num_slots=3, lora_type=lora_type)
    _claim(multi, 0, 4, 8.0, seed=1)
    _claim(multi, 2, 8, 4.0, seed=2)
    x = torch.randn(9, multi.in_features, dtype=torch.float64)
    multi.token_counts_by_slot = (4, 0, 5)
    mixed = multi(x)
    multi.token_counts_by_slot = (4, 0, 0)
    torch.testing.assert_close(mixed[:4], multi(x[:4]), rtol=0, atol=1e-12)
    multi.token_counts_by_slot = (0, 0, 5)
    torch.testing.assert_close(mixed[4:], multi(x[4:]), rtol=0, atol=1e-12)
    # a free slot adds nothing
    multi.token_counts_by_slot = (0, 9, 0)
    with torch.no_grad():
        base = torch.nn.functional.linear(x, multi.weight.flatten(0, -2)).view(mixed.shape)
    torch.testing.assert_close(multi(x), base, rtol=0, atol=0)


@pytest.mark.parametrize("projection", ["attention.qkv_linear.wqkv", "feed_forward.w13"])
def test_a_canonical_slot_equals_its_hf_projections_fused_by_the_model(projection):
    """Each component's delta ``B_i @ A_i``, under the HF name the slot exports it as, fused into the base weight's
    layout by the model's own ``from_hf``."""
    config = _qwen3()
    sd_adapter = config._owner.state_dict_adapter_cls(config, None)
    multi = _build(projection, num_slots=2, lora_type="canonical_lora")
    rank, alpha = 6, 3.0
    _claim(multi, 1, rank, alpha, seed=3)
    fqn = f"layers.0.{projection}"
    params = {f"layers.0.{projection.rsplit('.', 1)[0]}.{name}": p for name, p in _slot_state(multi, 1, projection)}
    exported = dict(iter_hf_adapter_tensors(sd_adapter, params, {fqn: multi.packing}))
    hf_deltas = {
        name.replace(".lora_B.weight", ".weight"): b @ exported[name.replace("lora_B", "lora_A")]
        for name, b in exported.items()
        if name.endswith(".lora_B.weight")
    }
    assert len(hf_deltas) == multi.packing.num_components
    # from_hf ties lm_head to the embedding of this tied model
    fused_delta = sd_adapter.from_hf({**hf_deltas, "model.embed_tokens.weight": torch.zeros(1)})
    (delta,) = (v for k, v in fused_delta.items() if k.startswith(fqn))

    x = torch.randn(7, multi.in_features, dtype=torch.float64)
    multi.token_counts_by_slot = (0, 7)
    with torch.no_grad():
        expected = torch.nn.functional.linear(
            x, (multi.weight + alpha / rank * delta.view(multi.weight.shape)).flatten(0, -2)
        )
    torch.testing.assert_close(multi(x).detach().flatten(1), expected, rtol=0, atol=1e-12)


def _slot_state(module, slot: int, projection: str):
    leaf = projection.rsplit(".", 1)[1]
    for name, param in module.slot_named_parameters(slot):
        yield f"{leaf}.{name}.weight", param.detach()


@pytest.fixture
def single_rank_gloo():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    init_gloo_group()
    try:
        yield
    finally:
        dist.destroy_process_group()


def _model_part(lora_type: str, num_slots: int = 3) -> tuple[torch.nn.Module, object]:
    config = _qwen3()
    sd_adapter = config._owner.state_dict_adapter_cls(config, None)
    converted = convert_model_config(
        copy.deepcopy(config),
        hf_targets=_QWEN3_ALL_LINEAR,
        rank=_MAX_RANK,
        alpha=16,
        num_slots=num_slots,
        lora_type=lora_type,
    )
    part = torch.nn.Module()
    part.layers = torch.nn.ModuleDict({"0": converted.layers[0].build()})
    return part, sd_adapter


@pytest.mark.parametrize("lora_type", ["lora", "canonical_lora"])
def test_slots_are_claimed_masked_routed_and_cleared(single_rank_gloo, lora_type):
    part, sd_adapter = _model_part(lora_type)
    slots = TorchtitanAdapterSlots([part], None, sd_adapter, _TARGETS, multi_lora=True)
    assert slots.num_slots == 3

    slots.init(1, 4, 8.0, seed=11)
    params = slot_adapter_parameters([part], 1)[0]
    for key, param in params.items():
        past_rank = param[4:] if ".lora_a" in key else param[..., 4:]
        assert not past_rank.any(), key
    assert any(param.any() for key, param in params.items() if ".lora_a" in key)
    assert slots.slot_spec(1).rank == 4 and slots.slot_spec(1).alpha == 8.0

    with pytest.raises(AssertionError, match="already claimed"):
        slots.init(1, 4, 8.0, seed=None)
    with pytest.raises(AssertionError, match="unclaimed slots \\[0\\]"):
        slots.route_tokens([3, 2, 0])
    slots.route_tokens([0, 5, 0])

    # the same seed draws the same adapter on another slot
    slots.init(2, 4, 8.0, seed=11)
    other = slot_adapter_parameters([part], 2)[0]
    assert all(torch.equal(p, params[k]) for k, p in other.items())

    a_before = {k: p.clone() for k, p in params.items() if ".lora_a" in k}
    slots.clear(1)
    assert all(not p.any() for k, p in params.items() if ".lora_b" in k)
    assert all(not torch.equal(p, a_before[k]) for k, p in params.items() if ".lora_a" in k)
    with pytest.raises(AssertionError, match="unclaimed slots \\[1\\]"):
        slots.route_tokens([0, 5, 0])


@pytest.mark.parametrize("lora_type", ["lora", "canonical_lora"])
def test_a_slots_pieces_are_its_hf_slices_at_its_rank(single_rank_gloo, lora_type):
    part, sd_adapter = _model_part(lora_type)
    slots = TorchtitanAdapterSlots([part], None, sd_adapter, _TARGETS, multi_lora=True)
    slots.init(2, 5, 10.0, seed=7)
    params, packings = slot_adapter_parameters([part], 2)
    expected, _ = map_named_tensors_to_hf_keys(
        iter_hf_adapter_tensors(sd_adapter, {k: p.detach() for k, p in params.items()}, packings),
        map_peft_keys_to_hf_keys,
        _TARGETS,
    )
    specs = slots.get_local_shard_specs(2)
    pieces = slots.local_pieces(2, slots.slot_parameters(2))
    # a shared lora_A is stored once, under its group's first member
    assert {spec.key for spec in specs} <= set(expected)
    assert {key for key in expected if key.endswith(".lora_B")} <= {spec.key for spec in specs}
    for spec, piece in zip(specs, pieces, strict=True):
        rank_axis = 0 if spec.key.endswith(".lora_A") else len(spec.global_shape) - 1
        assert spec.global_shape[rank_axis] == 5
        box = tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))
        assert torch.equal(piece.reshape(spec.local_shape), expected[spec.key][box]), spec.key
    # canonical components have their own A; the fused adapter shares one per fused projection
    assert bool(slots.row_mapping.shared_lora_a_groups) == (lora_type == "lora")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
