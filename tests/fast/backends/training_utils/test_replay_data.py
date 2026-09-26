from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


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
    args = SimpleNamespace(qkv_format="bshd", allgather_cp=allgather_cp, sequence_parallel=sequence_parallel)

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
            iterator = SimpleNamespace(reset=lambda: None, get_next=lambda keys, batch=rollout_data: batch)
            replays = [_Replay(), _Replay()]
            fill_replay_data(
                args=args,
                models=None,
                data_iterator=[iterator],
                num_microbatches=[1],
                rollout_data=rollout_data,
                data_key="routes",
                replay_list=replays,
                register_replay_list_func=register_replay_list_sequential,
                if_sp_region=sequence_first,
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


@pytest.mark.parametrize("sequence_first", [False, True])
@pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, True)])
def test_fill_replay_thd_preserves_packed_order(monkeypatch, sequence_first, tp_size, sequence_parallel):
    args = SimpleNamespace(
        qkv_format="thd", allgather_cp=False, sequence_parallel=sequence_parallel, data_pad_size_multiplier=2
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
        iterator = SimpleNamespace(reset=lambda: None, get_next=lambda keys, batch=rollout_data: batch)
        replay = _Replay()
        fill_replay_data(
            args=args,
            models=None,
            data_iterator=[iterator],
            num_microbatches=[1],
            rollout_data=rollout_data,
            data_key="routes",
            replay_list=[replay],
            register_replay_list_func=register_replay_list_sequential,
            sequence_first=sequence_first,
        )
        local_expected = expected.chunk(tp_size)[tp_rank] if sequence_parallel else expected
        torch.testing.assert_close(replay.recorded[0][:, 0], local_expected)
