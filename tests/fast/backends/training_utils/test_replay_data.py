from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.fast.utils.output_store.test_plan import Ref, Spec, stored_parts

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.replay.data import (
    attach_replay_data,
    register_replay_list_sequential,
    stream_blocks,
)
from miles.utils.output_store.plan import IndexerTopkSegments, RoutedExpertsSegments, Seg, StorePlanError
from miles.utils.replay_base import BaseReplayManager


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    @property
    def top_indices_list(self):
        return self.recorded

    def record(self, value):
        self.recorded.append(value)

    def clear(self):
        self.recorded = []


def _manager(replays, *, if_sp_region=True, indexer=False, kept_in_train_dump=False):
    manager = BaseReplayManager()
    manager.data_key = "routes"
    manager.replays = replays
    manager.register_replay_list_func = register_replay_list_sequential
    manager.if_sp_region = if_sp_region
    manager.replay_indices_are_token_positions = indexer
    manager.kept_in_train_dump = kept_in_train_dump
    return manager


def _attach(manager, **kwargs):
    """`attach_replay_data`, waited for: the manager's worker records the entries."""
    attach_replay_data(manager, **kwargs)
    manager.wait_filled(sum(kwargs["num_microbatches"]) - 1)


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, [replay_data], blocks=stream_blocks(replays))

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # PP case: 2 local modules at global streams 1 and 3 out of 4, whose rows this rank holds as two blocks.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]
    blocks = stream_blocks(replays)
    assert blocks == [(1, 2), (3, 4)]

    register_replay_list_sequential(replays, [replay_data[:, a:b] for a, b in blocks], blocks=blocks)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, [replay_data], blocks=[(0, 4)])


def test_stream_blocks_are_the_runs_of_the_registered_streams():
    # A VPP stage's indexer layers: two runs; a replay without stream_idx takes its position.
    replays = [_Replay(stream_idx=s) for s in (4, 5, 0, 1)] + [_Replay()]
    assert stream_blocks(replays) == [(0, 2), (4, 6)]
    assert stream_blocks([_Replay(), _Replay()]) == [(0, 2)]
    assert stream_blocks([]) == []


@pytest.mark.parametrize("lengths", [(3, 3), (3, 5), (3,)])
@pytest.mark.parametrize("sequence_first", [True, False], ids=["moe", "indexer"])
@pytest.mark.parametrize("cp_size,allgather_cp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, False), (2, True)])
def test_fill_replay_bshd_token_alignment(
    monkeypatch, lengths, sequence_first, cp_size, allgather_cp, tp_size, sequence_parallel
):
    max_seqlen = 8
    tokens = [torch.arange(1, length + 1) + sample * 100 for sample, length in enumerate(lengths)]
    # Each token has unique expert IDs in both streams and top-k slots.
    expert_offsets = torch.arange(4).reshape(2, 2)
    routes = [token_ids[:-1, None, None] * 10 + expert_offsets for token_ids in tokens]
    args = SimpleNamespace(
        qkv_format="bshd",
        allgather_cp=allgather_cp,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=sequence_parallel),
        save_debug_train_data=None,
    )

    for cp_rank in range(cp_size):
        positions = list(range(max_seqlen))
        if cp_size > 1:
            if allgather_cp:
                width = max_seqlen // cp_size
                positions = positions[cp_rank * width : (cp_rank + 1) * width]
            else:
                width = max_seqlen // (2 * cp_size)
                back = 2 * cp_size - cp_rank - 1
                positions = (
                    positions[cp_rank * width : (cp_rank + 1) * width] + positions[back * width : (back + 1) * width]
                )

        for tp_rank in range(tp_size):
            state = SimpleNamespace(
                cp=SimpleNamespace(size=cp_size, rank=cp_rank), tp=SimpleNamespace(size=tp_size, rank=tp_rank)
            )
            monkeypatch.setattr(parallel, "_parallel_state", state)
            rollout_data = {
                "routes": routes,
                "tokens": tokens,
                "max_seq_lens": [max_seqlen] * len(tokens),
            }
            replays = [_Replay(), _Replay()]
            _attach(
                _manager(replays, if_sp_region=sequence_first),
                args=args,
                models=None,
                data_iterator=[DataIterator(rollout_data, micro_batch_size=len(tokens))],
                num_microbatches=[1],
                rollout_data=rollout_data,
                sequence_first=sequence_first,
            )

            local_positions = positions
            if sequence_parallel and sequence_first:
                width = len(positions) // tp_size
                local_positions = positions[tp_rank * width : (tp_rank + 1) * width]
            # Enumerate token identities in the consumer's order, including masked rows.
            samples = range(len(tokens))
            token_order = (
                [(sample, pos) for pos in local_positions for sample in samples]
                if sequence_first
                else [(sample, pos) for sample in samples for pos in local_positions]
            )
            expected = torch.stack(
                [
                    tokens[sample][pos] * 10 + expert_offsets if pos < lengths[sample] - 1 else torch.full((2, 2), -1)
                    for sample, pos in token_order
                ]
            )
            for stream, replay in enumerate(replays):
                assert len(replay.recorded) == 1
                torch.testing.assert_close(replay.recorded[0], expected[:, stream])


@pytest.mark.parametrize("kept_in_train_dump", [False, True])
@pytest.mark.parametrize("save_debug_train_data", [None, "/tmp/{rollout_id}_{rank}.pt"])
@pytest.mark.parametrize("sequence_first", [False, True])
@pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, True)])
def test_fill_replay_thd_preserves_packed_order(
    monkeypatch, sequence_first, tp_size, sequence_parallel, save_debug_train_data, kept_in_train_dump
):
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=sequence_parallel),
        data_pad_size_multiplier=2,
        save_debug_train_data=save_debug_train_data,
    )
    expected = torch.tensor([0, 1, -1, 2, 3, -1])
    if tp_size == 2:
        expected = torch.cat([expected, torch.tensor([-1, -1])])
    for tp_rank in range(tp_size):
        state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=tp_size, rank=tp_rank))
        monkeypatch.setattr(parallel, "_parallel_state", state)
        rollout_data = {
            "routes": [torch.tensor([0, 1]).reshape(2, 1, 1), torch.tensor([2, 3]).reshape(2, 1, 1)],
            "tokens": [torch.arange(3), torch.arange(3)],
            "max_seq_lens": None,
        }
        replay = _Replay()
        _attach(
            _manager([replay], kept_in_train_dump=kept_in_train_dump),
            args=args,
            models=None,
            data_iterator=[DataIterator(rollout_data, micro_batch_size=2)],
            num_microbatches=[1],
            rollout_data=rollout_data,
            sequence_first=sequence_first,
        )
        local_expected = expected.chunk(tp_size)[tp_rank] if sequence_parallel else expected
        torch.testing.assert_close(replay.recorded[0][:, 0], local_expected)
        # Only a train-data dump keeps the consumed column, and only for a replay kept in it (R3).
        assert ("routes" in rollout_data) == (kept_in_train_dump and save_debug_train_data is not None)


def test_a_dump_keeps_none_for_a_sample_no_micro_batch_holds(monkeypatch):
    # A fixed schedule of one micro batch of 2 leaves sample 2 out, as torch_native does with untrimmed samples.
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data="/tmp/{rollout_id}_{rank}.pt",
    )
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    rollout_data = {
        "routes": [torch.full((2, 1, 1), i) for i in range(3)],
        "tokens": [torch.arange(3)] * 3,
        "max_seq_lens": None,
    }
    _attach(
        _manager([_Replay()], kept_in_train_dump=True),
        args=args,
        models=None,
        data_iterator=[DataIterator(rollout_data, micro_batch_size=2)],
        num_microbatches=[1],
        rollout_data=rollout_data,
    )
    assert [rows is None for rows in rollout_data["routes"]] == [False, False, True]


def _fill_thd_cp2(monkeypatch, *, cp_rank, allgather_cp, lengths, data_pad_size_multiplier, indexer):
    """Fill one thd micro batch on CP rank ``cp_rank`` of 2; sample ``i`` has replay rows ``10(i+1) + j``."""
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=allgather_cp,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=data_pad_size_multiplier,
        save_debug_train_data=None,
    )
    state = SimpleNamespace(cp=SimpleNamespace(size=2, rank=cp_rank), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    rollout_data = {
        "routes": [10 * (i + 1) + torch.arange(length - 1).reshape(-1, 1, 1) for i, length in enumerate(lengths)],
        "tokens": [torch.arange(length) for length in lengths],
        "max_seq_lens": None,
    }
    replay = _Replay()
    _attach(
        _manager([replay], indexer=indexer),
        args=args,
        models=None,
        data_iterator=[DataIterator(rollout_data, micro_batch_size=len(lengths))],
        num_microbatches=[1],
        rollout_data=rollout_data,
    )
    return replay.recorded[0][:, 0].tolist()


@pytest.mark.parametrize(
    "indexer,expected_by_rank",
    [
        # Sample 0 (T = 5) is padded to 8 in chunks of 2: rank 0 holds chunks 0 and 3, rank 1 chunks 1 and 2.
        # Sample 1 (T = 3) is padded to 4 in chunks of 1. The micro batch is then padded to a multiple of 4.
        (False, [[10, 11, -1, -1, 20, -1, -1, -1], [12, 13, -1, -1, 21, -1, -1, -1]]),
        # Indexer positions of sample 1 are shifted by sample 0's 4 local rows.
        (True, [[10, 11, -1, -1, 24, -1, -1, -1], [12, 13, -1, -1, 25, -1, -1, -1]]),
    ],
    ids=["moe", "indexer"],
)
def test_fill_replay_thd_zigzag_cp(monkeypatch, indexer, expected_by_rank):
    for cp_rank, expected in enumerate(expected_by_rank):
        recorded = _fill_thd_cp2(
            monkeypatch,
            cp_rank=cp_rank,
            allgather_cp=False,
            lengths=(5, 3),
            data_pad_size_multiplier=4,
            indexer=indexer,
        )
        assert recorded == expected


@pytest.mark.parametrize("indexer", [False, True], ids=["moe", "indexer"])
def test_fill_replay_thd_allgather_cp(monkeypatch, indexer):
    # The packed rows [10 11 12 13 -1 | 20 21 22 -1 | 30 -1] are padded to a multiple of 2 * 2 and split in two:
    # sample 0 stays on rank 0, sample 1 spans both ranks, sample 2 and the pack padding land on rank 1.
    # Indexer positions are not shifted under all-gather CP.
    expected_by_rank = [[10, 11, 12, 13, -1, 20], [21, 22, -1, 30, -1, -1]]
    for cp_rank, expected in enumerate(expected_by_rank):
        recorded = _fill_thd_cp2(
            monkeypatch,
            cp_rank=cp_rank,
            allgather_cp=True,
            lengths=(5, 4, 2),
            data_pad_size_multiplier=2,
            indexer=indexer,
        )
        assert recorded == expected


@pytest.mark.parametrize(
    "schedule,num_microbatches,expected",
    [
        (dict(micro_batch_size=1), [1, 1], [[0, 1, -1], [2, 3, 4, -1]]),
        (dict(micro_batch_indices=[[1], [0]]), [2], [[2, 3, 4, -1], [0, 1, -1]]),
    ],
    ids=["fixed", "balanced"],
)
def test_attach_fills_micro_batches_in_the_iterator_order(monkeypatch, schedule, num_microbatches, expected):
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    rollout_data = {
        "routes": [torch.tensor([0, 1]).reshape(2, 1, 1), torch.tensor([2, 3, 4]).reshape(3, 1, 1)],
        "tokens": [torch.arange(3), torch.arange(4)],
        "max_seq_lens": None,
    }
    replay = _Replay()
    iterator = DataIterator(rollout_data, **schedule)

    _attach(
        _manager([replay]),
        args=args,
        models=None,
        data_iterator=[iterator],
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
    )

    assert [entry[:, 0].tolist() for entry in replay.recorded] == expected
    assert "routes" not in rollout_data
    assert iterator.offset == 0


def test_attach_aligns_each_recorded_entry_once(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    rollout_data = {
        "routes": [torch.tensor([0, 1]).reshape(2, 1, 1), torch.tensor([2, 3]).reshape(2, 1, 1)],
        "tokens": [torch.arange(3), torch.arange(3)],
        "max_seq_lens": None,
    }
    replays = [_Replay(stream_idx=0), _Replay(stream_idx=0)]

    def append_pad_row(entry, pad_value):
        return torch.cat([entry, entry.new_full((1, *entry.shape[1:]), pad_value)])

    _attach(
        _manager(replays),
        args=args,
        models=None,
        data_iterator=[DataIterator(rollout_data, micro_batch_size=1)],
        num_microbatches=[2],
        rollout_data=rollout_data,
        align=append_pad_row,
    )

    for replay in replays:
        assert [entry[:, 0].tolist() for entry in replay.recorded] == [[0, 1, -1, -1], [2, 3, -1, -1]]


class TestNativeReplayOwnership:
    def test_fsdp_replay_does_not_require_megatron_sequence_parallel_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Native replay preserves token routing and padding without reading Megatron-only sequence parallelism."""
        state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
        monkeypatch.setattr(parallel, "_parallel_state", state)
        args = SimpleNamespace(
            qkv_format="bshd",
            allgather_cp=False,
            train_backend="fsdp",
            backend=SimpleNamespace(),
            save_debug_train_data=None,
        )
        batch = {
            "routes": [torch.tensor([[[11]], [[12]]])],
            "tokens": [torch.tensor([1, 2, 3])],
            "max_seq_lens": [4],
        }
        replay = _Replay()

        _attach(
            _manager([replay]),
            args=args,
            models=None,
            data_iterator=[DataIterator(batch, micro_batch_size=1)],
            num_microbatches=[1],
            rollout_data=batch,
        )

        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], torch.tensor([[11], [12], [-1], [-1]]))


# ============================== output store columns ==============================


class _FakeOutputStore:
    """``read_buffer`` / ``read_ranges`` over plain arrays, recording buffers and the bytes read."""

    def __init__(self, objects):
        self.objects, self.buffers, self.read_bytes = objects, [], 0

    def read_buffer(self, nbytes):
        buffer = SimpleNamespace(nbytes=nbytes, view=memoryview(bytearray(nbytes)), closed=False)
        buffer.close = lambda: setattr(buffer, "closed", True)
        self.buffers.append(buffer)
        return buffer

    def read_ranges(self, reads, *, into):
        assert not into.closed
        for read in reads:
            stored = self.objects[read.ref.key][read.field].reshape(-1).view(np.uint8)
            into.view[read.dst_offset : read.dst_offset + read.nbytes] = stored[
                read.src_offset : read.src_offset + read.nbytes
            ].tobytes()
            self.read_bytes += read.nbytes


@pytest.fixture
def fake_plugin(monkeypatch):
    """CPU CI has no radix_transfer_plugins; replay builds its RangeReads from it."""
    module = types.ModuleType("radix_transfer_plugins")
    module.RangeRead = lambda **fields: SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "radix_transfer_plugins", module)


def _stored(routes, *, indexer, splits=()):
    """Each sample's rows as segments of two objects (a two-turn session), and the objects.

    Each object holds the field's parts split at the layers ``splits``, as SGLang stores them.
    """
    cls = IndexerTopkSegments if indexer else RoutedExpertsSegments
    column, objects = [], {}
    for i, rows in enumerate(routes):
        rows = rows.numpy().astype(np.int32)
        cut = len(rows) // 2
        segs = []
        for key, part in ((f"{i}/a", rows[:cut]), (f"{i}/b", rows[cut:])):
            if len(part):
                objects[key] = stored_parts(cls.name, part, list(splits))
                specs = tuple(Spec(name, "int32", array.shape) for name, array in objects[key].items())
                segs.append(Seg(Ref(key, specs), 0, len(part)))
        column.append(cls(tuple(segs)))
    return column, objects


def _attach_both(
    monkeypatch,
    args,
    *,
    state,
    lengths,
    max_seq_len,
    indexer,
    store,
    manager=None,
    dump=False,
    num_streams=2,
    streams=(None, None),
    blocks=None,
    splits=(),
):
    """Recorded entries and the column left behind, filling 2 micro batches from inline or store rows.

    A replay per entry of ``streams``, its ``stream_idx``; ``blocks`` and ``splits`` as ``attach_replay_data``
    takes them and as SGLang stores the rows.
    """
    monkeypatch.setattr(parallel, "_parallel_state", state)
    routes = [
        100 * (i + 1) + torch.arange((n - 1) * num_streams * 2, dtype=torch.int32).reshape(n - 1, num_streams, 2)
        for i, n in enumerate(lengths)
    ]
    column, objects = _stored(routes, indexer=indexer, splits=splits) if store is not None else (routes, None)
    if store is not None:
        store.objects.update(objects)
    rollout_data = {
        "routes": column,
        "tokens": [torch.arange(n) for n in lengths],
        "max_seq_lens": [max_seq_len] * len(lengths),
    }
    replays = [_Replay(stream_idx=stream) for stream in streams]
    manager = manager or _manager(replays, indexer=indexer, kept_in_train_dump=dump)
    manager.replays = replays
    _attach(
        manager,
        args=args,
        models=None,
        data_iterator=[DataIterator(rollout_data, micro_batch_size=2)],
        num_microbatches=[(len(lengths) + 1) // 2],
        rollout_data=rollout_data,
        output_store=store,
        blocks=blocks,
    )
    return [replay.recorded for replay in replays], rollout_data.get("routes")


@pytest.mark.parametrize("indexer", [False, True], ids=["moe", "indexer"])
@pytest.mark.parametrize(
    "qkv_format,cp_size,allgather_cp",
    [
        ("thd", 1, False),
        ("thd", 2, False),
        ("thd", 2, True),
        ("bshd", 1, False),
        ("bshd", 2, False),
        ("bshd", 2, True),
    ],
)
def test_store_rows_fill_the_inline_entries(monkeypatch, fake_plugin, qkv_format, cp_size, allgather_cp, indexer):
    lengths, max_seq_len = (5, 3, 8, 2), 8
    args = SimpleNamespace(
        qkv_format=qkv_format,
        allgather_cp=allgather_cp,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=2,
        save_debug_train_data="/tmp/{rollout_id}_{rank}.pt",
    )
    read_bytes = 0
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=cp_rank), tp=SimpleNamespace(size=1, rank=0))
        common = dict(state=state, lengths=lengths, max_seq_len=max_seq_len, indexer=indexer, dump=True)
        inline, inline_dump = _attach_both(monkeypatch, args, store=None, **common)
        store = _FakeOutputStore({})
        stored, stored_dump = _attach_both(monkeypatch, args, store=store, **common)
        # Nothing recorded refers to the slot: overwriting it changes no entry (B2, AS5).
        for buffer in store.buffers:
            buffer.view[:] = b"\xff" * buffer.nbytes
        for inline_entries, stored_entries in zip(inline, stored, strict=True):
            assert len(inline_entries) == len(stored_entries) == 2
            for a, b in zip(inline_entries, stored_entries, strict=True):
                assert a.dtype == b.dtype and torch.equal(a, b)
        # A dump holds each sample's local rows, the same both ways.
        for a, b in zip(inline_dump, stored_dump, strict=True):
            assert torch.equal(a, b)
        read_bytes += store.read_bytes
    # Each real row is read once across the CP ranks.
    assert read_bytes == sum(n - 1 for n in lengths) * 2 * 2 * 4


def test_the_read_slot_is_kept_across_steps_and_replaced_when_too_small(monkeypatch, fake_plugin):
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    store = _FakeOutputStore({})
    manager = _manager([], indexer=False)
    attach = lambda lengths, store: _attach_both(  # noqa: E731
        monkeypatch, args, state=state, lengths=lengths, max_seq_len=None, indexer=False, store=store, manager=manager
    )

    attach((5, 3), store)
    attach((5, 3), store)
    assert len(store.buffers) == 1 and not store.buffers[0].closed
    attach((9, 3), store)
    assert len(store.buffers) == 2 and store.buffers[0].closed and not store.buffers[1].closed
    other = _FakeOutputStore({})
    attach((5, 3), other)
    assert store.buffers[1].closed and len(other.buffers) == 1


def test_store_rows_need_an_output_store_client(monkeypatch, fake_plugin):
    args = SimpleNamespace(qkv_format="thd", save_debug_train_data=None)
    column, _ = _stored([torch.ones(2, 2, 2, dtype=torch.int32)], indexer=False)
    rollout_data = {"routes": column, "tokens": [torch.arange(3)], "max_seq_lens": None}
    with pytest.raises(StorePlanError, match="no output store client"):
        _attach(
            _manager([_Replay()]),
            args=args,
            models=None,
            data_iterator=[DataIterator(rollout_data, micro_batch_size=1)],
            num_microbatches=[1],
            rollout_data=rollout_data,
        )


def _real_rows(dump):
    """A dump's rows that hold replay data rather than -1 padding."""
    return sum(int((rows != -1).flatten(1).all(dim=1).sum()) for rows in dump if rows is not None)


@pytest.mark.parametrize("indexer", [False, True], ids=["moe", "indexer"])
@pytest.mark.parametrize("blocks", [[(1, 3)], [(0, 1), (3, 4)]], ids=["pp-stage", "vpp-stage"])
@pytest.mark.parametrize("qkv_format,cp_size,allgather_cp", [("thd", 1, False), ("thd", 2, False), ("bshd", 2, True)])
def test_a_pipeline_stage_reads_and_registers_only_its_blocks(
    monkeypatch, fake_plugin, blocks, qkv_format, cp_size, allgather_cp, indexer
):
    # 4 streams stored in parts split at 1 and 3, as a two-stage pipeline (or two VPP chunks each) has them;
    # this rank's replays are the streams of its blocks.
    lengths, max_seq_len = (5, 3, 8, 2), 8
    args = SimpleNamespace(
        qkv_format=qkv_format,
        allgather_cp=allgather_cp,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=2,
        save_debug_train_data="/tmp/{rollout_id}_{rank}.pt",
    )
    streams = [stream for start, end in blocks for stream in range(start, end)]
    row_bytes = len(streams) * 2 * 4
    read_bytes = 0
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=cp_rank), tp=SimpleNamespace(size=1, rank=0))
        common = dict(
            state=state,
            lengths=lengths,
            max_seq_len=max_seq_len,
            indexer=indexer,
            dump=True,
            num_streams=4,
            streams=streams,
        )
        whole, _ = _attach_both(monkeypatch, args, store=None, **common)
        inline, inline_dump = _attach_both(monkeypatch, args, store=None, blocks=blocks, **common)
        store = _FakeOutputStore({})
        stored, stored_dump = _attach_both(monkeypatch, args, store=store, blocks=blocks, splits=(1, 3), **common)
        # Each replay records what it records from whole rows.
        for entries in zip(whole, inline, stored, strict=True):
            for a, b, c in zip(*entries, strict=True):
                assert torch.equal(a, b) and torch.equal(a, c)
        # A dump holds each sample's local rows of this rank's streams, the same both ways.
        for a, b in zip(inline_dump, stored_dump, strict=True):
            assert a.shape[1] == len(streams) and torch.equal(a, b)
        # The rank reads only its local rows of its own parts.
        assert store.read_bytes == _real_rows(stored_dump) * row_bytes
        read_bytes += store.read_bytes
    assert read_bytes == sum(n - 1 for n in lengths) * row_bytes


def test_a_block_the_objects_do_not_store_fails_the_fill(monkeypatch, fake_plugin):
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    # SGLang split at layer 2, but this rank asks for layers 1-3.
    with pytest.raises(
        StorePlanError, match=r"streams 1-3 are not a stored part: object 0/a has parts \[\(0, 2\), \(2, 4\)\]"
    ):
        _attach_both(
            monkeypatch,
            args,
            state=state,
            lengths=(5, 3),
            max_seq_len=None,
            indexer=False,
            store=_FakeOutputStore({}),
            num_streams=4,
            streams=(1, 2),
            blocks=[(1, 3)],
            splits=(2,),
        )


def test_all_zero_routed_experts_fail_only_a_rank_that_replays_them(monkeypatch, fake_plugin):
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="megatron",
        backend=SimpleNamespace(sequence_parallel=False),
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0)),
    )
    # Layer 0 is dense: SGLang stores zeros for it. Layer 1 is MoE.
    rows = torch.stack([torch.zeros(4, 2, dtype=torch.int32), torch.full((4, 2), 7, dtype=torch.int32)], dim=1)
    column, objects = _stored([rows], indexer=False, splits=(1,))

    def attach(blocks, replays):
        rollout_data = {"routes": list(column), "tokens": [torch.arange(5)], "max_seq_lens": None}
        _attach(
            _manager(replays),
            args=args,
            models=None,
            data_iterator=[DataIterator(rollout_data, micro_batch_size=1)],
            num_microbatches=[1],
            rollout_data=rollout_data,
            output_store=_FakeOutputStore(objects),
            blocks=blocks,
        )

    # A rank holding both parts (two VPP chunks), and a stage of the dense layer only, which replays nothing.
    attach([(0, 1), (1, 2)], [_Replay(stream_idx=1)])
    attach([(0, 1)], [])
    with pytest.raises(StorePlanError, match="all zeros"):
        attach([(0, 1)], [_Replay(stream_idx=0)])
