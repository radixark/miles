"""Dist-checkpoint tests for the native LoRA adapter format.

CPU-only: gloo process groups plus MCore ``dist_checkpointing`` on CPU
tensors. The cross-layout test saves under TP=2 (two spawned workers) and
reloads the same checkpoint under TP=1, exercising MCore resharding end to
end — the property the legacy per-rank ``.pt`` format could not provide.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from miles_plugins.lora import checkpointing
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.modules.linear import NATIVE_LORA_SHARDED_STATE_FLAG, LoRALinear
from miles_plugins.lora.spec.base import AttachContext, ProjectionSpec, ShardLayout

_HIDDEN = 8
_OUT = 12  # divisible by every tested TP size
_RANK = 4


def _patch_dist_ckpt_writer_for_cpu():
    """Make MCore's async checkpoint writer usable in the test environment.

    Two upstream GPU/Linux assumptions break here, in production neither
    matters (training always has GPUs and runs on Linux):

    - ``FileSystemWriterAsync.preload_tensors`` hardcodes ``non_blocking=True``
      and then calls ``torch.cuda.synchronize()`` unconditionally, which raises
      on CUDA-less builds and GPU-less CI lanes.
    - ``write_preloaded_data_multiproc`` forks worker processes; forking a
      threaded (gloo) parent deadlocks/dies on macOS, so run the same writes
      in-process there. Linux CI keeps the real multi-process writer.

    Returns an undo callable so the fixture can restore process-wide state for
    subsequent tests in the same session (spawned workers skip the undo — their
    process exits).
    """
    import queue as queue_module
    import sys

    from megatron.core.dist_checkpointing.strategies import filesystem_async

    originals: list[tuple[Any, str, Any]] = []

    def _set(owner, attribute, value):
        originals.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    if not torch.cuda.is_available():
        # MCore's saver has unconditional CUDA calls (preload_tensors'
        # synchronize, state_dict_saver's failure-broadcast tensor device).
        # Without CUDA these would raise anyway, so a CPU fallback cannot mask
        # real behavior.
        _set(torch.cuda, "current_device", lambda: "cpu")
        _set(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    if sys.platform == "darwin":
        write_one_bucket = filesystem_async.FileSystemWriterAsync.write_preloaded_data

        class _NoopCountQueue:
            def get(self):
                return None

            def task_done(self):
                return None

        def _inline_multiproc(transform_list, use_msc, rank, write_buckets, global_results_queue):
            results: dict | Exception = {}
            for index, write_bucket in enumerate(write_buckets):
                results_queue = queue_module.SimpleQueue()
                write_one_bucket(transform_list, index, write_bucket, results_queue, _NoopCountQueue(), use_fsync=True)
                local_index, local_results_or_exc = results_queue.get()
                if isinstance(local_results_or_exc, Exception):
                    results = local_results_or_exc
                    break
                results[local_index] = local_results_or_exc
            global_results_queue.put(results)

        _set(filesystem_async.FileSystemWriterAsync, "write_preloaded_data_multiproc", staticmethod(_inline_multiproc))
        # With in-process writes there is no cross-process result passing, so a
        # plain queue replaces the spawn-Manager queue (whose server process
        # also fails to start on macOS under a threaded gloo parent).
        _set(filesystem_async, "_get_write_results_queue", queue_module.Queue)

    def _undo():
        for owner, attribute, value in reversed(originals):
            setattr(owner, attribute, value)

    return _undo


@pytest.fixture(autouse=True)
def _cpu_safe_dist_ckpt_writer():
    undo = _patch_dist_ckpt_writer_for_cpu()
    try:
        yield
    finally:
        undo()


def _context(tp_rank: int, tp_size: int) -> AttachContext:
    lora = LoRAConfig(rank=_RANK, alpha=8.0, dropout=0.0, target_modules=frozenset({"q_proj", "o_proj"}))
    transformer_config = SimpleNamespace(hidden_size=_HIDDEN, layernorm_epsilon=1e-5, sequence_parallel=False)
    return AttachContext(
        lora=lora,
        transformer_config=transformer_config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        layer_prefix="",
        shared_expert="",
    )


class _WalkedModule(nn.Module):
    """Minimal MCore-style module: default-walks children for sharded state."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import sharded_state_dict_default

        sharded = {}
        for name, module in self.named_children():
            sharded.update(sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata))
        return sharded


def _build_chunk(tp_rank: int, tp_size: int) -> nn.Module:
    """One fake block: a COLUMN adapter (B sharded dim 0), a ROW adapter (A sharded dim 1), a base linear."""
    context = _context(tp_rank, tp_size)
    out_local = _OUT // tp_size
    block = _WalkedModule()
    block.base_linear = nn.Linear(_HIDDEN, out_local, bias=False)
    block.q_adapter = LoRALinear(
        hf_prefix="layers.0.",
        projection=ProjectionSpec(hf="q_proj", attr="q", layout=ShardLayout.COLUMN),
        reference=block.base_linear.weight,
        context=context,
        in_features=_HIDDEN,
        out_features=out_local,
    )
    block.o_adapter = LoRALinear(
        hf_prefix="layers.0.",
        projection=ProjectionSpec(hf="o_proj", attr="o", layout=ShardLayout.ROW),
        reference=block.base_linear.weight,
        context=context,
        in_features=out_local,
        out_features=_HIDDEN,
    )
    chunk = _WalkedModule()
    chunk.decoder = block
    return chunk


def _global_reference_tensors() -> dict[str, torch.Tensor]:
    """Deterministic full (TP-gathered) values for every adapter parameter."""
    return {
        "q_A": torch.arange(_RANK * _HIDDEN, dtype=torch.float32).view(_RANK, _HIDDEN),
        "q_B": torch.arange(_OUT * _RANK, dtype=torch.float32).view(_OUT, _RANK) + 50,
        "o_A": torch.arange(_RANK * _OUT, dtype=torch.float32).view(_RANK, _OUT) + 100,
        "o_B": torch.arange(_HIDDEN * _RANK, dtype=torch.float32).view(_HIDDEN, _RANK) + 200,
    }


def _fill_chunk(chunk: nn.Module, tp_rank: int, tp_size: int) -> None:
    """Give every rank its shard of the deterministic global values."""
    full = _global_reference_tensors()
    out_local = _OUT // tp_size
    rows = slice(tp_rank * out_local, (tp_rank + 1) * out_local)
    with torch.no_grad():
        chunk.decoder.q_adapter.q_A.copy_(full["q_A"])  # replicated
        chunk.decoder.q_adapter.q_B.copy_(full["q_B"][rows])  # TP-sharded dim 0
        chunk.decoder.o_adapter.o_A.copy_(full["o_A"][:, rows])  # TP-sharded dim 1
        chunk.decoder.o_adapter.o_B.copy_(full["o_B"])  # replicated


def _assert_chunk_matches_reference(chunk: nn.Module, tp_rank: int, tp_size: int) -> None:
    full = _global_reference_tensors()
    out_local = _OUT // tp_size
    rows = slice(tp_rank * out_local, (tp_rank + 1) * out_local)
    torch.testing.assert_close(chunk.decoder.q_adapter.q_A, full["q_A"])
    torch.testing.assert_close(chunk.decoder.q_adapter.q_B, full["q_B"][rows])
    torch.testing.assert_close(chunk.decoder.o_adapter.o_A, full["o_A"][:, rows])
    torch.testing.assert_close(chunk.decoder.o_adapter.o_B, full["o_B"])


class _FakeMegatronOptimizer:
    """Implements Megatron's optimizer checkpoint interface for plumbing tests."""

    def __init__(self):
        self.exp_avg: dict[str, torch.Tensor] = {}
        self.loaded_state = None
        self.reloaded_model_params = False
        self.seen_metadata: list[dict | None] = []

    def fill_from(self, sharded_model_state, offset: float) -> None:
        self.exp_avg = {
            key: torch.full_like(entry.data.detach(), offset) for key, entry in sharded_model_state.items()
        }

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        self.seen_metadata.append(metadata)
        state = {}
        for key, entry in model_sharded_state_dict.items():
            if key not in self.exp_avg:
                self.exp_avg[key] = torch.zeros_like(entry.data.detach())
            state[key] = replace(entry, key=f"optimizer.state.exp_avg.{entry.key}", data=self.exp_avg[key])
        return {"exp_avg": state}

    def load_state_dict(self, state):
        self.loaded_state = state

    def reload_model_params(self):
        self.reloaded_model_params = True


class _FakeScheduler:
    def __init__(self, lr: float):
        self.lr = lr

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state):
        self.lr = state["lr"]


def _init_single_rank(tmp_path) -> None:
    from megatron.core import parallel_state

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/single_rank_pg", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()


def _destroy_parallel() -> None:
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture
def single_rank_parallel(tmp_path):
    _init_single_rank(tmp_path)
    try:
        yield
    finally:
        _destroy_parallel()


def test_sharded_state_dict_gated_by_metadata(single_rank_parallel):
    chunk = _build_chunk(tp_rank=0, tp_size=1)
    adapter = chunk.decoder.q_adapter

    assert adapter.sharded_state_dict(prefix="p.") == {}
    assert adapter.sharded_state_dict(prefix="p.", metadata={"unrelated": True}) == {}

    sharded = adapter.sharded_state_dict(prefix="p.", metadata={NATIVE_LORA_SHARDED_STATE_FLAG: True})
    assert set(sharded) == {"p.q_A", "p.q_B"}
    assert sharded["p.q_A"].data is adapter.q_A  # keep_vars: identity is load-bearing
    # COLUMN layout: B is TP-sharded on dim 0, A is replicated.
    assert sharded["p.q_B"].axis_fragmentations[0] == 1  # tp_size == 1 here
    assert sharded["p.q_A"].replica_id[1] == 0


def test_save_load_roundtrip_single_rank(single_rank_parallel, tmp_path):
    ckpt_dir = tmp_path / "adapter" / checkpointing.NATIVE_DIST_CKPT_DIRNAME
    source = _build_chunk(tp_rank=0, tp_size=1)
    _fill_chunk(source, tp_rank=0, tp_size=1)
    optimizer = _FakeMegatronOptimizer()
    optimizer.fill_from(checkpointing.native_adapter_sharded_state_dict([source]), offset=3.5)
    scheduler = _FakeScheduler(lr=0.25)

    checkpointing.save_native_adapter_dist_checkpoint(
        [source], ckpt_dir, optimizer=optimizer, opt_param_scheduler=scheduler, iteration=7
    )
    assert checkpointing.is_native_adapter_dist_checkpoint(ckpt_dir)
    assert optimizer.seen_metadata[-1] == {"distrib_optim_sharding_type": "fully_reshardable"}

    target = _build_chunk(tp_rank=0, tp_size=1)
    target_optimizer = _FakeMegatronOptimizer()
    target_scheduler = _FakeScheduler(lr=0.0)
    iteration = checkpointing.load_native_adapter_dist_checkpoint(
        [target], ckpt_dir, optimizer=target_optimizer, opt_param_scheduler=target_scheduler
    )

    assert iteration == 7
    _assert_chunk_matches_reference(target, tp_rank=0, tp_size=1)
    assert target_scheduler.lr == 0.25
    assert target_optimizer.loaded_state is not None
    for tensor in target_optimizer.loaded_state["exp_avg"].values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 3.5))


def test_load_without_optimizer_state_reloads_model_params(single_rank_parallel, tmp_path):
    ckpt_dir = tmp_path / "adapter" / checkpointing.NATIVE_DIST_CKPT_DIRNAME
    source = _build_chunk(tp_rank=0, tp_size=1)
    _fill_chunk(source, tp_rank=0, tp_size=1)
    checkpointing.save_native_adapter_dist_checkpoint([source], ckpt_dir, iteration=7)

    target = _build_chunk(tp_rank=0, tp_size=1)
    optimizer = _FakeMegatronOptimizer()
    iteration = checkpointing.load_native_adapter_dist_checkpoint([target], ckpt_dir, optimizer=optimizer)

    # No optimizer state in the checkpoint: resume restarts the schedule, and the
    # fp32 mains must be re-snapshotted from the just-loaded adapter values.
    assert iteration is None
    assert optimizer.reloaded_model_params
    _assert_chunk_matches_reference(target, tp_rank=0, tp_size=1)


def test_walk_that_skips_adapters_fails_loudly():
    class _OpaqueChunk(nn.Module):
        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return {}

    chunk = _OpaqueChunk()
    chunk.q_adapter = _build_chunk(tp_rank=0, tp_size=1).decoder.q_adapter
    with pytest.raises(AssertionError, match="never surfaced"):
        checkpointing.native_adapter_sharded_state_dict([chunk])


class _VppChunk(nn.Module):
    """Simulates an MCore VPP chunk: dict keys stay chunk-local, only the ShardedTensor .key is globalized."""

    def __init__(self, block: nn.Module, global_chunk_index: int):
        super().__init__()
        self.decoder = block
        self._global_chunk_index = global_chunk_index

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
        from megatron.core.transformer.utils import sharded_state_dict_default

        sharded = {}
        for name, module in self.named_children():
            sharded.update(sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata))
        replace_prefix_for_sharding(sharded, f"{prefix}decoder.", f"{prefix}decoder.chunk{self._global_chunk_index}.")
        return sharded


def test_vpp_chunks_with_identical_local_dict_keys(single_rank_parallel, tmp_path):
    """Two VPP chunks emit identical chunk-local dict keys; the merge must key by ShardedTensor identity."""
    ckpt_dir = tmp_path / "adapter" / checkpointing.NATIVE_DIST_CKPT_DIRNAME

    def build_vpp_chunks() -> list[nn.Module]:
        return [_VppChunk(_build_chunk(tp_rank=0, tp_size=1).decoder, index) for index in range(2)]

    source = build_vpp_chunks()
    for offset, chunk in enumerate(source):
        with torch.no_grad():
            for parameter in chunk.parameters():
                if parameter.requires_grad:
                    parameter.fill_(float(offset + 1))

    sharded = checkpointing.native_adapter_sharded_state_dict(source)
    adapter_param_count = sum(1 for c in source for n, _ in c.named_parameters() if "adapter" in n)
    assert len(sharded) == adapter_param_count  # both chunks fully represented, no collision

    checkpointing.save_native_adapter_dist_checkpoint(source, ckpt_dir, iteration=1)
    target = build_vpp_chunks()
    checkpointing.load_native_adapter_dist_checkpoint(target, ckpt_dir)
    for offset, chunk in enumerate(target):
        for name, parameter in chunk.named_parameters():
            if "adapter" in name:
                torch.testing.assert_close(
                    parameter, torch.full_like(parameter, float(offset + 1)), msg=f"chunk {offset}: {name}"
                )


def test_legacy_namespaced_plan_maps_by_chunk_index():
    """Shards written by earlier revisions of this branch (namespaced keys) stay readable, even with VPP chunks."""
    chunks = [_build_chunk(tp_rank=0, tp_size=1), _build_chunk(tp_rank=0, tp_size=1)]
    state = {
        f"_miles_model_chunks.{index}.{name}": torch.zeros_like(parameter)
        for index, chunk in enumerate(chunks)
        for name, parameter in chunk.named_parameters()
        if "adapter" in name
    }
    plan = checkpointing.native_adapter_load_plan(chunks, state)
    assert plan.compatible
    assert len(plan.assignments) == len(state)


def test_legacy_flat_plan_keeps_single_candidate_and_rejects_vpp_ambiguity():
    chunk_a = _build_chunk(tp_rank=0, tp_size=1)
    state = {name: torch.zeros_like(parameter) for name, parameter in chunk_a.named_parameters() if "adapter" in name}

    plan = checkpointing.native_adapter_load_plan([chunk_a], state)
    assert plan.compatible
    assert len(plan.assignments) == len(state)

    # Two VPP chunks expose identical unqualified names: legacy flat keys are ambiguous.
    chunk_b = _build_chunk(tp_rank=0, tp_size=1)
    plan = checkpointing.native_adapter_load_plan([chunk_a, chunk_b], state)
    assert not plan.compatible
    assert any("ambiguous" in message for message in plan.shape_mismatches)


def _cross_layout_save_worker(rank: int, world_size: int, tmp_path_str: str) -> None:
    from megatron.core import parallel_state

    _patch_dist_ckpt_writer_for_cpu()  # spawned process: fixtures do not apply
    dist.init_process_group("gloo", init_method=f"file://{tmp_path_str}/save_pg", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        chunk = _build_chunk(tp_rank=rank, tp_size=world_size)
        _fill_chunk(chunk, tp_rank=rank, tp_size=world_size)
        optimizer = _FakeMegatronOptimizer()
        optimizer.fill_from(checkpointing.native_adapter_sharded_state_dict([chunk]), offset=3.5)
        checkpointing.save_native_adapter_dist_checkpoint(
            [chunk],
            f"{tmp_path_str}/adapter/{checkpointing.NATIVE_DIST_CKPT_DIRNAME}",
            optimizer=optimizer,
            opt_param_scheduler=_FakeScheduler(lr=0.25),
            iteration=7,
        )
    finally:
        _destroy_parallel()


def test_cross_layout_tp2_save_tp1_load(tmp_path):
    """Save under TP=2, reload under TP=1: MCore reshards weights and optimizer state."""
    mp.spawn(_cross_layout_save_worker, args=(2, str(tmp_path)), nprocs=2, join=True)

    ckpt_dir = tmp_path / "adapter" / checkpointing.NATIVE_DIST_CKPT_DIRNAME
    assert checkpointing.is_native_adapter_dist_checkpoint(ckpt_dir)

    _init_single_rank(tmp_path)
    try:
        target = _build_chunk(tp_rank=0, tp_size=1)
        optimizer = _FakeMegatronOptimizer()
        scheduler = _FakeScheduler(lr=0.0)
        iteration = checkpointing.load_native_adapter_dist_checkpoint(
            [target], ckpt_dir, optimizer=optimizer, opt_param_scheduler=scheduler
        )
        assert iteration == 7
        # Full (TP-gathered) tensors reassembled from the two TP=2 shards.
        _assert_chunk_matches_reference(target, tp_rank=0, tp_size=1)
        assert scheduler.lr == 0.25
        assert optimizer.loaded_state is not None
        for tensor in optimizer.loaded_state["exp_avg"].values():
            torch.testing.assert_close(tensor, torch.full_like(tensor, 3.5))
    finally:
        _destroy_parallel()
