from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.replay.data import attach_replay_data, register_replay_list_sequential
from miles.utils.replay_base import BaseReplayManager


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

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


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, replay_data)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # PP case: 2 local modules at global streams 1 and 3 out of 4.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]

    register_replay_list_sequential(replays, replay_data)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, replay_data)


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
            attach_replay_data(
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
        attach_replay_data(
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
    attach_replay_data(
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
    "schedule,expected",
    [
        (dict(micro_batch_size=1), [[0, 1, -1], [2, 3, -1]]),
        (dict(micro_batch_indices=[[1], [0]]), [[2, 3, -1], [0, 1, -1]]),
    ],
    ids=["fixed", "balanced"],
)
def test_attach_fills_micro_batches_in_the_iterator_order(monkeypatch, schedule, expected):
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
    replay = _Replay()

    attach_replay_data(
        _manager([replay]),
        args=args,
        models=None,
        data_iterator=[DataIterator(rollout_data, **schedule)],
        num_microbatches=[2],
        rollout_data=rollout_data,
    )

    assert [entry[:, 0].tolist() for entry in replay.recorded] == expected
    assert "routes" not in rollout_data


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

        attach_replay_data(
            _manager([replay]),
            args=args,
            models=None,
            data_iterator=[DataIterator(batch, micro_batch_size=1)],
            num_microbatches=[1],
            rollout_data=batch,
        )

        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], torch.tensor([[11], [12], [-1], [-1]]))
