"""CPU unit tests for FSDP TPU support, 4D packed causal masking, PaddedRows integration, and local weight export."""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import Qwen3Config, Qwen3ForCausalLM

from miles import platforms
from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.fsdp_utils import checkpoint as checkpoint_module
from miles.backends.fsdp_utils import hf_weight_iterator as iterator_module
from miles.backends.fsdp_utils import parallel as parallel_module
from miles.backends.fsdp_utils.adaptations.precision import PrecisionPolicy, precision_forward_context
from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
from miles.backends.fsdp_utils.arguments import FSDPArgs, parse_fsdp_cli
from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.training_utils.loss.hub.response_layout import PackedSlices, PaddedRows, padded_rows_enabled
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.torch_native import offload as offload_module
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

_CPU_DEVICE = torch.device("cpu")


def _fake_tpu_platform(*, device: torch.device = _CPU_DEVICE) -> SimpleNamespace:
    return SimpleNamespace(
        device_type="tpu",
        compile_backend="tpu",
        default_attn_implementation="sdpa",
        supports_pin_memory=False,
        premul_sum_reduce=False,
        dynamic_shapes=False,
        current_device=lambda: device,
        empty_cache=Mock(),
    )


def _fake_cuda_platform(*, device: torch.device = _CPU_DEVICE) -> SimpleNamespace:
    return SimpleNamespace(
        device_type="cuda",
        compile_backend="inductor",
        default_attn_implementation="flash_attention_2",
        supports_pin_memory=True,
        premul_sum_reduce=True,
        dynamic_shapes=True,
        current_device=lambda: device,
        empty_cache=Mock(),
    )


class _FakeDTensor(DTensor):
    """Minimal DTensor subclass for testing 1D/2D multi-shard placement logic on a single CPU process."""

    @staticmethod
    def __new__(cls, local_tensor: torch.Tensor, global_shape: tuple[int, ...], mesh, placements):
        # Construct an empty base Tensor wrapper with the requested global shape and metadata.
        t = torch.Tensor._make_subclass(cls, torch.empty(global_shape, dtype=local_tensor.dtype), require_grad=False)
        t._fake_local = local_tensor
        t._fake_mesh = mesh
        t._fake_placements = tuple(placements)
        return t

    @property
    def device_mesh(self):
        return self._fake_mesh

    @property
    def placements(self):
        return self._fake_placements

    def to_local(self):
        return self._fake_local


class _FakeMesh:
    def __init__(self, shape: tuple[int, ...], local_rank: int = 0):
        self._shape = shape
        self._local_rank = local_rank

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def size(self, dim: int = 0) -> int:
        return self._shape[dim]

    def get_local_rank(self, mesh_dim: int = 0) -> int:
        return self._local_rank


class _FakeStridedShard(Shard):
    """Subclasses Shard just like torch.distributed.tensor._StridedShard."""


# ---------------------------------------------------------------------------
# 1. 4D block-causal packed attention mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_causal_mask_structure_and_no_softmax_nans(dtype: torch.dtype) -> None:
    cu_seqlens_host = (0, 3, 5, 8)
    mask = actor_module.build_packed_causal_mask(
        cu_seqlens_host,
        num_documents=2,
        seq_len=8,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 8, 8)
    assert mask.dtype is dtype

    min_val = torch.finfo(dtype).min
    m = mask[0, 0]

    # Document 0: [0, 3), Document 1: [3, 5), Padding: [5, 8)
    for q in range(8):
        for k in range(8):
            if q < 3:
                allowed = k < 3 and k <= q
            elif q < 5:
                allowed = 3 <= k < 5 and k <= q
            else:
                # Each padding token attends only to itself (diagonal = 0.0)
                allowed = k == q
            expected = 0.0 if allowed else min_val
            assert m[q, k].item() == expected, f"mismatch at ({q}, {k})"

    # Softmax over random scores + mask must never produce NaNs or Infs (including padded rows).
    scores = torch.randn(1, 2, 8, 8, dtype=dtype)
    probs = torch.softmax((scores + mask).float(), dim=-1)
    assert torch.isfinite(probs).all()
    # Cross-doc attention weights must be identically 0.
    assert (probs[:, :, :3, 3:] == 0).all()
    assert (probs[:, :, 3:5, :3] == 0).all()
    assert (probs[:, :, 3:5, 5:] == 0).all()
    # Padding rows attend solely to themselves.
    for p in range(5, 8):
        assert probs[0, 0, p, p].item() == pytest.approx(1.0)


def test_packed_causal_mask_validates_bounds() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        actor_module.build_packed_causal_mask(
            (0, 3), num_documents=2, seq_len=4, dtype=torch.float32, device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="exceeds mask sequence length"):
        actor_module.build_packed_causal_mask(
            (0, 3, 6), num_documents=2, seq_len=5, dtype=torch.float32, device=torch.device("cpu")
        )


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen3_packed_4d_mask_prevents_cross_doc_leakage_and_matches_unpacked(
    attn_impl: str, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attn_implementation=attn_impl,
    )
    model = Qwen3ForCausalLM(cfg).to(dtype).train()

    doc0 = torch.tensor([[1, 2, 3, 4, 5]])
    doc1_a = torch.tensor([[10, 11, 12, 13]])
    doc1_b = torch.tensor([[25, 26, 27, 28]])
    pad = torch.zeros((1, 7), dtype=torch.long)

    inp_a = torch.cat([doc0, doc1_a, pad], dim=1)
    inp_b = torch.cat([doc0, doc1_b, pad], dim=1)
    pos_ids = torch.cat([torch.arange(5), torch.arange(4), torch.zeros(7, dtype=torch.long)]).unsqueeze(0)
    mask = actor_module.build_packed_causal_mask(
        (0, 5, 9, 16), num_documents=2, seq_len=16, dtype=dtype, device=torch.device("cpu")
    )

    with torch.no_grad():
        out_a = model(input_ids=inp_a, position_ids=pos_ids, attention_mask=mask, use_cache=False).logits
        out_b = model(input_ids=inp_b, position_ids=pos_ids, attention_mask=mask, use_cache=False).logits
        out_doc0 = model(input_ids=doc0, position_ids=torch.arange(5).unsqueeze(0), use_cache=False).logits
        out_doc1 = model(input_ids=doc1_a, position_ids=torch.arange(4).unsqueeze(0), use_cache=False).logits

    assert torch.isfinite(out_a).all()
    # Zero cross-document leakage: changing doc1 leaves doc0 bit-for-bit identical.
    assert torch.equal(out_a[:, :5], out_b[:, :5])
    # Packed outputs match separate unpacked runs.
    atol = 1e-2 if dtype is torch.bfloat16 else 1e-5
    torch.testing.assert_close(out_a[:, :5], out_doc0, atol=atol, rtol=atol)
    torch.testing.assert_close(out_a[:, 5:9], out_doc1, atol=atol, rtol=atol)

    # Backward through packed forward with padding tail produces finite grads everywhere.
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=inp_a, position_ids=pos_ids, attention_mask=mask, use_cache=False).logits
    loss = logits[:, :9].float().sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# 2. Static-shape padding helpers & _get_model_inputs_args
# ---------------------------------------------------------------------------


def test_pad_packed_inputs_pads_and_validates_overflow() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    same_t, same_p = actor_module.pad_packed_inputs(tokens, pos, target_len=5)
    assert same_t is tokens and same_p is pos

    pad_t, pad_p = actor_module.pad_packed_inputs(tokens, pos, target_len=8)
    assert pad_t.tolist() == [[1, 2, 3, 4, 5, 0, 0, 0]]
    assert pad_p.tolist() == [[0, 1, 2, 0, 1, 0, 0, 0]]

    with pytest.raises(ValueError, match="exceeds static padded length 4"):
        actor_module.pad_packed_inputs(tokens, pos, target_len=4)


def test_resolve_static_seq_len_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    args = Namespace(
        max_tokens_per_gpu=1000,
        max_seq_len=None,
        rollout_max_prompt_len=None,
        rollout_max_response_len=None,
        micro_batch_size=1,
        data_pad_size_multiplier=128,
    )

    monkeypatch.setattr(platforms, "current_platform", _fake_cuda_platform())
    assert actor_module.resolve_static_seq_len(args) is None

    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    # 1000 rounded up to multiple of 128 -> 1024
    assert actor_module.resolve_static_seq_len(args) == 1024

    # Fallback to micro_batch_size * max_seq_len
    args_fallback = Namespace(
        max_tokens_per_gpu=None,
        max_seq_len=256,
        rollout_max_prompt_len=None,
        rollout_max_response_len=None,
        micro_batch_size=2,
        data_pad_size_multiplier=128,
    )
    assert actor_module.resolve_static_seq_len(args_fallback) == 512

    # Missing all length bounds raises clearly
    args_missing = Namespace(
        max_tokens_per_gpu=None,
        max_seq_len=None,
        rollout_max_prompt_len=None,
        rollout_max_response_len=None,
        micro_batch_size=None,
        data_pad_size_multiplier=128,
    )
    with pytest.raises(ValueError, match="requires --max-tokens-per-gpu"):
        actor_module.resolve_static_seq_len(args_missing)


def test_actor_model_inputs_static_padding_and_4d_mask_skip_flash_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())

    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.arch_adapter = resolve_arch_adapter(SimpleNamespace(model_type="qwen3"))
    actor._uses_4d_packed_mask = True
    actor._static_seq_len = 12
    actor._packed_mask_dtype = torch.bfloat16

    cu_seqlens_host = (0, 3, 5, 8)
    batch = {
        "tokens": torch.tensor([[11, 12, 13, 21, 22, 0, 0, 0]], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]], dtype=torch.long),
        "cu_seqlens": torch.tensor(cu_seqlens_host, dtype=torch.int32),
        "cu_seqlens_host": cu_seqlens_host,
        "max_seqlen": 3,
        "unconcat_tokens": [torch.tensor([11, 12, 13]), torch.tensor([21, 22])],
    }

    model_args = actor._get_model_inputs_args(batch)

    assert model_args["input_ids"].shape == (1, 12)
    assert model_args["position_ids"].shape == (1, 12)
    assert model_args["attention_mask"].shape == (1, 1, 12, 12)
    assert model_args["attention_mask"].dtype is torch.bfloat16
    assert model_args["use_cache"] is False
    # FlashAttention-only kwargs must be omitted for sdpa/eager 4D mask path.
    assert "cu_seq_lens_q" not in model_args
    assert "cu_seq_lens_k" not in model_args
    assert "max_length_q" not in model_args
    assert "max_length_k" not in model_args
    # Original batch tensors were not mutated in place.
    assert batch["tokens"].shape == (1, 8)


# ---------------------------------------------------------------------------
# 3. Platform-agnostic FSDP knobs (args, apply_fsdp2, precision, offload, parallel, checkpoint)
# ---------------------------------------------------------------------------


def test_fsdp_args_attn_implementation_defaults_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(platforms, "current_platform", _fake_cuda_platform())
    assert FSDPArgs().attn_implementation == "flash_attention_2"
    assert parse_fsdp_cli().attn_implementation == "flash_attention_2"
    assert parse_fsdp_cli().fsdp_compile is False

    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    assert FSDPArgs().attn_implementation == "sdpa"
    assert parse_fsdp_cli().attn_implementation == "sdpa"


def test_apply_fsdp2_sets_pin_memory_and_force_sum_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFSDPModule(FSDPModule):
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

    class _FakeFSDPBlock(torch.nn.Module, _FakeFSDPModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.force_sum_calls: list[bool] = []

        def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
            self.force_sum_calls.append(enable)

    class _ToyModel(torch.nn.Module, _FakeFSDPModule):
        _no_split_modules = ["_FakeFSDPBlock"]

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.block = _FakeFSDPBlock()
            self.force_sum_calls: list[bool] = []

        def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
            self.force_sum_calls.append(enable)

    args = SimpleNamespace(backend=SimpleNamespace(fp16=False))

    # TPU: pin_memory=False and set_force_sum_reduction_for_comms(True)
    tpu_captured = []
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda mod, **kw: tpu_captured.append(kw))
    tpu_model = _ToyModel()
    actor_module.apply_fsdp2(tpu_model, cpu_offload=True, args=args)

    assert all(kw["offload_policy"].pin_memory is False for kw in tpu_captured)
    assert tpu_model.force_sum_calls == [True]
    assert tpu_model.block.force_sum_calls == [True]

    # CUDA: pin_memory=True and set_force_sum_reduction_for_comms NOT called
    cuda_captured = []
    monkeypatch.setattr(platforms, "current_platform", _fake_cuda_platform())
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda mod, **kw: cuda_captured.append(kw))
    cuda_model = _ToyModel()
    actor_module.apply_fsdp2(cuda_model, cpu_offload=True, args=args)

    assert all(kw["offload_policy"].pin_memory is True for kw in cuda_captured)
    assert cuda_model.force_sum_calls == []
    assert cuda_model.block.force_sum_calls == []


def test_precision_forward_context_uses_platform_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = []

    @contextmanager
    def fake_autocast(*, device_type, dtype):
        entered.append((device_type, dtype))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    policy = PrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, autocast_dtype=torch.bfloat16)

    with precision_forward_context(policy):
        pass

    assert entered == [("tpu", torch.bfloat16)]


def test_offload_respects_platform_supports_pin_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned + non-blocking on CUDA; pageable + blocking on TPU (a non-blocking D2H needs pinned memory)."""
    pin_flags = []
    non_blocking_flags = []
    orig_empty_like = torch.empty_like

    class _SpyHost(torch.Tensor):
        def copy_(self, src, non_blocking=False):
            non_blocking_flags.append(non_blocking)
            return super().copy_(src)

    def spy_empty_like(tensor, *args, **kwargs):
        pin_flags.append(kwargs.get("pin_memory"))
        return orig_empty_like(tensor, *args, **{**kwargs, "pin_memory": False}).as_subclass(_SpyHost)

    monkeypatch.setattr(torch, "empty_like", spy_empty_like)
    x = torch.ones(2, 2)

    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    offload_module._to_pinned_host(x)
    monkeypatch.setattr(platforms, "current_platform", _fake_cuda_platform())
    offload_module._to_pinned_host(x)

    assert pin_flags == [False, True]
    assert non_blocking_flags == [False, True]


def test_create_fsdp_parallel_state_uses_platform_device_type_and_ordered_subgroups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    mesh_calls = []
    subgroup_calls = []
    fake_dp_group = object()
    fake_gloo_group = object()
    fake_self_group = object()

    def fake_build_meshes(*, device_type, world_size, dp_replicate_size):
        mesh_calls.append((device_type, world_size, dp_replicate_size))
        fake_mesh = SimpleNamespace(shape=(world_size,), get_group=lambda: fake_dp_group)
        return {"dp": fake_mesh, "fsdp": fake_mesh}

    def fake_subgroups(ranks_list):
        subgroup_calls.append(ranks_list)
        return fake_self_group, [fake_self_group] * len(ranks_list)

    monkeypatch.setattr(parallel_module, "build_fsdp_meshes", fake_build_meshes)
    monkeypatch.setattr(
        parallel_module.dist, "get_world_size", lambda group=None: 1 if group is fake_self_group else 4
    )
    monkeypatch.setattr(parallel_module.dist, "get_rank", lambda group=None: 0 if group is fake_self_group else 2)
    monkeypatch.setattr(parallel_module.dist, "new_subgroups_by_enumeration", fake_subgroups)
    monkeypatch.setattr(parallel_module, "get_gloo_group", lambda: fake_gloo_group)

    state = parallel_module.create_fsdp_parallel_state(Namespace(backend=Namespace(dp_replicate_size=1)))

    assert mesh_calls == [("tpu", 4, 1)]
    assert subgroup_calls == [[[0], [1], [2], [3]]]
    assert state.effective_dp.gloo_group is fake_gloo_group
    assert state.cp.group is fake_self_group


def test_checkpoint_save_and_finalize_skip_cuda_rng_when_cuda_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_module.dist, "get_rank", lambda *a, **kw: 0)
    monkeypatch.setattr(checkpoint_module.dist, "get_world_size", lambda *a, **kw: 1)
    monkeypatch.setattr(checkpoint_module.dist, "barrier", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", Mock(side_effect=AssertionError("must not call CUDA RNG")))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(checkpoint_module.dcp, "save", lambda *a, **kw: None)

    actor = SimpleNamespace(
        model=torch.nn.Linear(2, 1, bias=False),
        optimizer=None,
        lr_scheduler=None,
        args=SimpleNamespace(
            backend=FsdpArgsNamespace(
                save=str(tmp_path),
                no_save_optim=True,
                no_load_rng=False,
            ),
            start_rollout_id=None,
        ),
        global_step=3,
        micro_step=5,
    )

    checkpoint_module.save(actor, iteration=0)
    rng_saved = torch.load(tmp_path / "iter_0000001" / "rng.pt", map_location="cpu")
    assert "torch" in rng_saved and "cuda" not in rng_saved

    checkpoint_module.finalize_load(
        actor, {"rng": rng_saved, "metadata": {"global_step": 3, "micro_step": 5}, "iteration": 1}
    )


def test_ref_context_uses_platform_empty_cache_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_plat = _fake_tpu_platform(device=torch.device("cpu"))
    monkeypatch.setattr(platforms, "current_platform", fake_plat)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())

    moved_to = []
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.optimizer_cpu_offload = False
    actor.model = SimpleNamespace(
        cpu=lambda: moved_to.append("cpu"),
        to=lambda dev: moved_to.append(dev),
        cuda=Mock(side_effect=AssertionError("must not call .cuda()")),
    )
    actor.ref_model = SimpleNamespace(eval=lambda: moved_to.append("ref_eval"))

    with actor._ref_context():
        moved_to.append("inside")

    assert moved_to == ["cpu", "ref_eval", "inside", torch.device("cpu")]
    assert fake_plat.empty_cache.call_count == 2


def test_premove_is_a_noop_where_fully_shard_can_move_states_itself() -> None:
    """CPU/CUDA tensor types are shallow-copy compatible, so fully_shard keeps doing the move unchanged."""
    model = torch.nn.Linear(4, 4)
    before = [id(p) for p in model.parameters()]
    actor_module._premove_states_fully_shard_cannot_move(model, torch.device("cpu"))
    assert [id(p) for p in model.parameters()] == before


def test_premove_replaces_host_states_on_an_incompatible_device_and_keeps_meta_ones() -> None:
    """torch_tpu tensors are not shallow-copy compatible with CPU ones (fully_shard's `.data =` would raise).

    CPU -> meta has the same incompatibility, so it stands in for TPU here.
    """

    class _Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.host = torch.nn.Linear(4, 4)
            self.register_buffer("host_buf", torch.ones(4))
            with torch.device("meta"):
                self.deferred = torch.nn.Linear(4, 4)

    target = torch.device("meta")
    assert not torch._has_compatible_shallow_copy_type(torch.empty(0), torch.empty(0, device=target))
    model = _Mixed()
    host_weight = model.host.weight
    deferred_weight = model.deferred.weight

    actor_module._premove_states_fully_shard_cannot_move(model, target)

    assert all(t.device == target for t in [*model.parameters(), *model.buffers()])
    assert all(isinstance(p, torch.nn.Parameter) for p in model.parameters())
    # the host param is replaced (no set_data), the deferred meta param is left as fully_shard would leave it
    assert model.host.weight is not host_weight and host_weight.device.type == "cpu"
    assert model.deferred.weight is deferred_weight


def test_premove_keeps_tied_weights_tied() -> None:
    """Qwen3-0.6B ties lm_head to embed_tokens; the rollout engine relies on that, so the move must not split them."""

    class _Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.head.weight = self.emb.weight

    model = _Tied()
    actor_module._premove_states_fully_shard_cannot_move(model, torch.device("meta"))

    assert model.head.weight is model.emb.weight
    assert model.emb.weight.device.type == "meta"
    assert len(list(model.parameters())) == 1


def test_compile_fsdp_blocks_targets_no_split_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled_backends = []

    def fake_compile(fn, *, backend, fullgraph):
        compiled_backends.append((backend, fullgraph))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class _DecoderLayer(torch.nn.Module):
        def forward(self, x):
            return x

    class _Model(torch.nn.Module):
        _no_split_modules = ["_DecoderLayer"]

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
            self.head = torch.nn.Linear(2, 2)

    actor_module._compile_fsdp_blocks(_Model(), backend="tpu")
    assert compiled_backends == [("tpu", True), ("tpu", True)]


# ---------------------------------------------------------------------------
# 4. PaddedRows activation & FSDP + PaddedRows end-to-end forward match
# ---------------------------------------------------------------------------


def test_padded_rows_enabled_for_fsdp_on_tpu_and_matches_packed_slices(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    fsdp_args = Namespace(
        train_backend="fsdp",
        loss_type="policy_loss",
        qkv_format="thd",
        rollout_temperature=1.0,
        entropy_coef=0.0,
        log_probs_backend="torch",
        log_probs_chunk_size=32,
        allgather_cp=False,
        true_on_policy_mode=False,
        vocab_size=32,
        debug_unified_grad_fused_logprob=False,
    )
    assert padded_rows_enabled(fsdp_args) is True

    monkeypatch.setattr(platforms, "current_platform", _fake_cuda_platform())
    assert padded_rows_enabled(fsdp_args) is False

    # End-to-end: FSDPTrainRayActor static-padded 4D-mask forward feeds PaddedRows cleanly.
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())
    created_pg = False
    if not dist.is_initialized():
        rendezvous = tmp_path / "pg"
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        created_pg = True
    try:
        trivial = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
        set_parallel_state(
            ParallelState(
                intra_dp=trivial,
                intra_dp_cp=trivial,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )

        torch.manual_seed(7)
        cfg = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            attn_implementation="sdpa",
        )
        model = Qwen3ForCausalLM(cfg).eval()

        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor.arch_adapter = resolve_arch_adapter(cfg)
        actor._uses_4d_packed_mask = True
        actor._static_seq_len = 16
        actor._packed_mask_dtype = torch.float32

        s0 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)  # prompt 3, response 2
        s1 = torch.tensor([6, 7, 8, 9], dtype=torch.long)  # prompt 2, response 2
        cu_seqlens_host = (0, 5, 9)
        batch = {
            "tokens": torch.cat([s0, s1]).unsqueeze(0),
            "position_ids": torch.cat([torch.arange(5), torch.arange(4)]).unsqueeze(0),
            "cu_seqlens": torch.tensor(cu_seqlens_host, dtype=torch.int32),
            "cu_seqlens_host": cu_seqlens_host,
            "max_seqlen": 5,
            "unconcat_tokens": [s0, s1],
            "total_lengths": [5, 4],
            "response_lengths": [2, 2],
            "loss_masks": [torch.ones(2, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        }

        with torch.no_grad():
            padded_logits = model(**actor._get_model_inputs_args(batch)).logits
        assert padded_logits.shape == (1, 16, 32)

        (padded_out,) = PaddedRows.finish_forward_pass(
            [
                PaddedRows.build(fsdp_args, batch, padded_logits).forward_log_probs(
                    padded_logits, with_entropy=True, use_sampling_mask=False
                )
            ]
        )
        packed_out = PackedSlices.build(fsdp_args, batch, padded_logits[:, :9]).forward_log_probs(
            padded_logits[:, :9], with_entropy=True, use_sampling_mask=False
        )
        for key in ("log_probs", "entropy"):
            for got, want in zip(padded_out[key], packed_out[key], strict=True):
                torch.testing.assert_close(got.reshape(-1), want.reshape(-1), rtol=1e-5, atol=1e-5)
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 5. FSDPHfWeightIterator.iter_local_weights() invariants
# ---------------------------------------------------------------------------


def _make_iterator(state: dict, *, sync_dtypes: dict | None = None, model_type: str = "qwen3", adapter=None):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type), state_dict=lambda: state)
    if sync_dtypes is not None:
        model._fsdp_sync_dtypes = sync_dtypes
    with patch.object(
        iterator_module, "resolve_arch_adapter", return_value=adapter or resolve_arch_adapter(model.config)
    ):
        return iterator_module.FSDPHfWeightIterator.build(
            Namespace(update_weight_buffer_size=1 << 30),
            model,
            required_placement=WeightUpdatePlacement(gather_pp=True),
            model_name="qwen3forcausallm",
            quantization_config=None,
        )


def test_iter_local_weights_sharded_2d_skips_gather_and_enforces_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())

    # Backing storage slice to test view / alias cloning on the local shard.
    backing = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    local_view = backing[2:4]  # _base is not None, storage_offset() != 0
    sharded_2d = _FakeDTensor(
        local_view, global_shape=(8, 4), mesh=_FakeMesh((4,), local_rank=2), placements=(Shard(0),)
    )

    # 1D sharded tensor MUST gather full tensor (never sharded for tpu-sync).
    sharded_1d = _FakeDTensor(
        torch.ones(2, dtype=torch.float32),
        global_shape=(8,),
        mesh=_FakeMesh((4,), local_rank=2),
        placements=(Shard(0),),
    )
    full_1d = torch.arange(8, dtype=torch.float32)

    # Uneven 2D sharded tensor (7 % 4 != 0) MUST gather full tensor.
    uneven_2d = _FakeDTensor(
        torch.ones(2, 4, dtype=torch.float32),
        global_shape=(7, 4),
        mesh=_FakeMesh((4,), local_rank=2),
        placements=(Shard(0),),
    )
    full_uneven = torch.arange(28, dtype=torch.float32).reshape(7, 4)

    # 2D HSDP mesh tensor MUST gather full tensor.
    hsdp_2d = _FakeDTensor(
        torch.ones(2, 4, dtype=torch.float32),
        global_shape=(8, 4),
        mesh=_FakeMesh((2, 4), local_rank=1),
        placements=(Replicate(), Shard(0)),
    )
    full_hsdp = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    # _StridedShard (Shard subclass) MUST gather full tensor.
    strided_2d = _FakeDTensor(
        torch.ones(2, 4, dtype=torch.float32),
        global_shape=(8, 4),
        mesh=_FakeMesh((4,), local_rank=2),
        placements=(_FakeStridedShard(0),),
    )
    full_strided = torch.ones(8, 4, dtype=torch.float32)

    # Tied plain tensors sharing identical storage must yield distinct owned buffers.
    shared_weight = torch.randn(4, 4, dtype=torch.bfloat16)
    non_contig = torch.randn(4, 4, dtype=torch.bfloat16).t()

    state = {
        "a_sharded_2d": sharded_2d,
        "b_sharded_1d": sharded_1d,
        "c_uneven_2d": uneven_2d,
        "d_hsdp_2d": hsdp_2d,
        "e_strided_2d": strided_2d,
        "f_tied_1": shared_weight,
        "g_tied_2": shared_weight,
        "h_non_contig": non_contig,
    }
    sync_dtypes = {
        "a_sharded_2d": torch.bfloat16,
        "b_sharded_1d": torch.bfloat16,
        "c_uneven_2d": torch.bfloat16,
        "d_hsdp_2d": torch.bfloat16,
        "e_strided_2d": torch.bfloat16,
        "f_tied_1": torch.bfloat16,
        "g_tied_2": torch.bfloat16,
        "h_non_contig": torch.bfloat16,
    }

    gathered_ids = []
    full_map = {
        id(sharded_1d): full_1d,
        id(uneven_2d): full_uneven,
        id(hsdp_2d): full_hsdp,
        id(strided_2d): full_strided,
    }

    def fake_gather(param, *, async_op=False):
        gathered_ids.append(id(param))
        return full_map.get(id(param), param)

    it = _make_iterator(state, sync_dtypes=sync_dtypes)
    with patch.object(iterator_module, "gather_full_param", side_effect=fake_gather):
        results = list(it.iter_local_weights())

    names = [name for name, _, _ in results]
    assert names == sorted(state.keys())
    # Critically: sharded_2d was NEVER passed to gather_full_param!
    assert id(sharded_2d) not in gathered_ids
    assert id(sharded_1d) in gathered_ids
    assert id(uneven_2d) in gathered_ids
    assert id(hsdp_2d) in gathered_ids
    assert id(strided_2d) in gathered_ids

    by_name = {name: (tensor, block) for name, tensor, block in results}
    assert by_name["a_sharded_2d"][1] == (4, 2)
    assert by_name["a_sharded_2d"][0].shape == (2, 4)
    torch.testing.assert_close(by_name["a_sharded_2d"][0], local_view.to(torch.bfloat16))

    for full_name in (
        "b_sharded_1d",
        "c_uneven_2d",
        "d_hsdp_2d",
        "e_strided_2d",
        "f_tied_1",
        "g_tied_2",
        "h_non_contig",
    ):
        assert by_name[full_name][1] is None

    # Every yielded tensor must satisfy tpu-sync invariants: contiguous, bf16, on target device, owned, distinct storage.
    ptrs = set()
    for name, tensor, _ in results:
        assert tensor.dtype is torch.bfloat16, name
        assert tensor.device == torch.device("cpu"), name
        assert tensor.is_contiguous(), name
        assert tensor._base is None, name
        assert tensor.storage_offset() == 0, name
        assert not tensor.requires_grad, name
        assert tensor.data_ptr() not in ptrs, f"aliased buffer for {name}"
        ptrs.add(tensor.data_ptr())


def test_iter_local_weights_already_bf16_view_is_cloned(monkeypatch: pytest.MonkeyPatch) -> None:
    """When keep_fp32_master=False, to_local() is already bf16; both nonzero- and zero-offset views must be cloned."""
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())

    backing = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    nonzero_view = backing[2:4]
    zero_offset_view = backing[:2]
    assert nonzero_view._base is not None and nonzero_view.storage_offset() == 8
    assert zero_offset_view._base is not None and zero_offset_view.storage_offset() == 0

    sharded_nz = _FakeDTensor(
        nonzero_view, global_shape=(8, 4), mesh=_FakeMesh((4,), local_rank=1), placements=(Shard(0),)
    )
    sharded_z = _FakeDTensor(
        zero_offset_view, global_shape=(8, 4), mesh=_FakeMesh((4,), local_rank=0), placements=(Shard(0),)
    )

    it = _make_iterator({"w_nz": sharded_nz, "w_z": sharded_z}, sync_dtypes=None)
    results = dict((name, (tensor, block)) for name, tensor, block in it.iter_local_weights())

    t_nz, block_nz = results["w_nz"]
    assert block_nz == (4, 1)
    assert t_nz._base is None and t_nz.storage_offset() == 0
    assert t_nz.data_ptr() != nonzero_view.data_ptr()
    torch.testing.assert_close(t_nz, nonzero_view)

    t_z, block_z = results["w_z"]
    assert block_z == (4, 0)
    assert t_z._base is None and t_z.storage_offset() == 0
    assert t_z.data_ptr() != zero_offset_view.data_ptr()
    torch.testing.assert_close(t_z, zero_offset_view)


def test_iter_local_weights_applies_param_transform_and_clones_views(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transformed params (e.g. qwen3_moe batched experts) gather full, expand, and yield owned buffers with block=None."""
    monkeypatch.setattr(platforms, "current_platform", _fake_tpu_platform())

    batched = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    sharded = _FakeDTensor(
        batched[:1], global_shape=(2, 4, 2), mesh=_FakeMesh((2,), local_rank=0), placements=(Shard(0),)
    )
    adapter = SimpleNamespace(
        param_transform=lambda name, param: (
            lambda n, full, model: ((f"expert.{i}", full[i]) for i in range(full.shape[0]))
        )
    )
    it = _make_iterator(
        {"experts.gate_up_proj": sharded},
        sync_dtypes={"expert.0": torch.bfloat16, "expert.1": torch.bfloat16},
        adapter=adapter,
    )
    with patch.object(iterator_module, "gather_full_param", return_value=batched) as gather_mock:
        results = list(it.iter_local_weights())

    assert gather_mock.call_count == 1
    assert [(name, block, tuple(t.shape)) for name, t, block in results] == [
        ("expert.0", None, (4, 2)),
        ("expert.1", None, (4, 2)),
    ]
    for _name, tensor, _ in results:
        assert tensor.dtype is torch.bfloat16
        assert tensor.is_contiguous()
        assert tensor._base is None and tensor.storage_offset() == 0
