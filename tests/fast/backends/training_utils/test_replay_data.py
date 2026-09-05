from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.training_utils.replay_data import _bshd_replay_to_megatron_order, register_replay_list_sequential


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


def test_bshd_replay_uses_megatron_sequence_major_order():
    replay_data = torch.arange(2 * 4).reshape(2, 4, 1, 1)

    actual = _bshd_replay_to_megatron_order(
        replay_data,
        sequence_parallel=False,
        tp_rank=0,
        tp_size=2,
    )

    assert actual[:, 0, 0].tolist() == [0, 4, 1, 5, 2, 6, 3, 7]


@pytest.mark.parametrize(
    ("tp_rank", "expected"),
    [
        (0, [0, 4, 1, 5]),
        (1, [2, 6, 3, 7]),
    ],
)
def test_bshd_replay_sequence_parallel_slices_sequence_dimension(tp_rank, expected):
    replay_data = torch.arange(2 * 4).reshape(2, 4, 1, 1)

    actual = _bshd_replay_to_megatron_order(
        replay_data,
        sequence_parallel=True,
        tp_rank=tp_rank,
        tp_size=2,
    )

    assert actual[:, 0, 0].tolist() == expected
