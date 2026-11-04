"""LegacyRankShardLoader reads a single-LoRA ``--save`` written before the adapter format (adapter_megatron_rank*.pt)
into the model as main's restore_lora_adapter did."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.lora.checkpoint as checkpoint
import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.checkpoint import LegacyRankShardLoader
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from miles.backends.megatron_utils.lora.utils import _adapter_shard_params
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization

RUN = AdapterParametrization("m", 1, 2.0, "lora", ("model.layers.0.self_attn.q_proj",), False)


class _Adapter(NativeLoRAAdapter):
    def __init__(self):
        super().__init__("model.layers.0.self_attn.q_proj.", (), tp_rank=0)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))

    def export_plan(self, gather):
        return []


class _Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.adapter = _Adapter()
        self.param_syncs = 0

    def start_param_sync(self, force_sync):
        self.param_syncs += 1


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )


def _save_old_format(path, chunks, **training_state):
    """What main's save_lora_checkpoint wrote on rank 0."""
    torch.save(
        {name: param.detach().cpu() for name, param in _adapter_shard_params(chunks).items()},
        path / "adapter_megatron_rank0.pt",
    )
    if training_state:
        state = {"iteration": 5, "optimizer": {"step": 7}, "opt_param_scheduler": {"lr": 0.5}} | training_state
        torch.save(state, path / "training_state_rank0.pt")


def _random_chunks(n):
    chunks = [_Chunk() for _ in range(n)]
    for chunk in chunks:
        for param in chunk.adapter.parameters():
            param.data.copy_(torch.randn_like(param))
    return chunks


def _optimizer(calls):
    return SimpleNamespace(
        load_state_dict=lambda state: calls.append(("load_state_dict", state)),
        reload_model_params=lambda: calls.append(("reload_model_params", None)),
    )


def _load(path, chunks, calls, *, load_optimizer):
    loader = LegacyRankShardLoader(chunks, _optimizer(calls), RUN, load_optimizer, precision_aware=False)
    return loader(path)


def test_virtual_pipeline_chunks_restore_their_own_adapters_bitwise(tmp_path):
    saved = _random_chunks(2)
    _save_old_format(tmp_path, saved)
    restored, calls = [_Chunk(), _Chunk()], []

    # the old save records no alpha; load_adapter_dir falls back to the run's
    assert _load(tmp_path, restored, calls, load_optimizer=False) == replace(RUN, lora_alpha=None)

    for before, after in zip(saved, restored, strict=True):
        for a, b in zip(before.adapter.parameters(), after.adapter.parameters(), strict=True):
            assert torch.equal(a, b)
    # the fp32 masters were built before the adapter load; without the refresh the first step reverts it
    assert calls == [("reload_model_params", None)]


def test_a_full_checkpoint_hands_the_saved_optimizer_and_parameter_state_over_unchanged(tmp_path, monkeypatch):
    """The old path loaded optimizer.state_dict() and each DistributedOptimizer's reshardable parameter state."""
    calls, casts = [], []
    parameter_state = {"per_bucket_numel": {}, 0: {torch.float32: [[{"param": torch.randn(2), "padding": False}]]}}
    distributed = SimpleNamespace(
        optimizer=object(), load_parameter_state_from_dp_reshardable=lambda state: calls.append(("parameter", state))
    )
    monkeypatch.setattr(checkpoint, "_distributed_optimizers", lambda _optimizer: [distributed])
    monkeypatch.setattr(
        checkpoint, "_build_cast_main_to_params_fn", lambda _optimizer, precision_aware: lambda: casts.append(1)
    )
    saved = _random_chunks(1)
    _save_old_format(tmp_path, saved, parameter_state=[parameter_state])
    restored = [_Chunk()]

    _load(tmp_path, restored, calls, load_optimizer=True)

    assert [name for name, _ in calls] == ["load_state_dict", "parameter"]
    assert calls[0][1] == {"step": 7}
    assert torch.equal(calls[1][1][0][torch.float32][0][0]["param"], parameter_state[0][torch.float32][0][0]["param"])
    # the loaded masters, not the shard's bf16 weights, become the model's parameters on every rank
    assert casts == [1] and restored[0].param_syncs == 1
    for a, b in zip(saved[0].adapter.parameters(), restored[0].adapter.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "training_state",
    [{}, {"optimizer": None, "parameter_state": None}, {"optimizer": {"step": 7}}],
    ids=["no-training-state", "saved-with-no-save-optim", "pre-parameter-state-format"],
)
def test_an_unrestorable_optimizer_fails_unless_no_load_optim(tmp_path, monkeypatch, training_state):
    monkeypatch.setattr(checkpoint, "_distributed_optimizers", lambda _optimizer: [])
    _save_old_format(tmp_path, _random_chunks(1), **training_state)

    with pytest.raises(AssertionError, match="no optimizer state to resume; pass --no-load-optim"):
        _load(tmp_path, [_Chunk()], [], load_optimizer=True)
    _load(tmp_path, [_Chunk()], [], load_optimizer=False)


def test_a_shard_that_does_not_match_the_adapter_fails(tmp_path):
    """A silently partial load resumes training on a half-initialized adapter."""
    torch.save(
        {"adapter.lora_A": torch.ones(1, 2), "stale_lora_C": torch.ones(1)}, tmp_path / "adapter_megatron_rank0.pt"
    )

    with pytest.raises(RuntimeError, match=r"missing=\['adapter.lora_B'\], unexpected=\['stale_lora_C'\]"):
        _load(tmp_path, [_Chunk()], [], load_optimizer=False)


def test_shards_saved_under_another_layout_fail(tmp_path):
    """Falling through to fresh adapter weights would hide a resharding mistake."""
    torch.save({"adapter.lora_A": torch.ones(1, 2)}, tmp_path / "adapter_megatron_rank1.pt")

    with pytest.raises(AssertionError, match="none for global rank 0"):
        _load(tmp_path, [_Chunk()], [], load_optimizer=False)


def test_a_directory_without_a_shard_for_this_rank_fails(tmp_path):
    with pytest.raises(AssertionError, match="no adapter shard for this rank"):
        _load(tmp_path, [_Chunk()], [], load_optimizer=False)
