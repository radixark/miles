from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import replay_utils
from miles.backends.megatron_utils.replay_utils import assign_routed_experts_streams
from miles.backends.training_utils.replay.data import stream_blocks
from miles.utils.replay_base import Replay


def _models(monkeypatch, moe_layer_freq, *, num_layers, stages):
    """One model per virtual pipeline stage of this rank, building layers `stages[i] = (start, end)`."""
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage: stages[vp_stage][0])
    monkeypatch.setattr(
        replay_utils, "get_num_layers_to_build", lambda config, vp_stage: stages[vp_stage][1] - stages[vp_stage][0]
    )
    config = SimpleNamespace(moe_layer_freq=moe_layer_freq, num_layers=num_layers)
    return [SimpleNamespace(module=SimpleNamespace(config=config)) for _ in stages]


@pytest.mark.parametrize(
    "moe_layer_freq, layer_ids, blocks",
    [
        ([0, 1, 1, 0, 1], [1, 2, 4], [(1, 3), (4, 5)]),
        (2, [0, 2, 4], [(0, 1), (2, 3), (4, 5)]),
        (1, [0, 1, 2, 3, 4], [(0, 5)]),
    ],
)
def test_a_routing_replay_stream_is_the_moe_layer_its_router_sits_in(monkeypatch, moe_layer_freq, layer_ids, blocks):
    # Two chunks of one rank: the embedding alone (--account-for-embedding-in-pipeline-split), then layers 0 to 4.
    models = _models(monkeypatch, moe_layer_freq, num_layers=5, stages=[(0, 0), (0, 5)])
    replays = [Replay() for _ in layer_ids]

    assign_routed_experts_streams(replays, models)

    assert [replay.stream_idx for replay in replays] == layer_ids
    assert stream_blocks(replays) == blocks


def test_a_stage_of_dense_layers_has_no_routing_replay_stream(monkeypatch):
    models = _models(monkeypatch, [0, 1], num_layers=2, stages=[(0, 1)])

    assign_routed_experts_streams([], models)


def test_routers_that_do_not_match_the_moe_layers_fail(monkeypatch):
    models = _models(monkeypatch, [0, 1, 1], num_layers=3, stages=[(0, 3)])

    with pytest.raises(ValueError, match=r"1 routers registered .* layers \[1, 2\]"):
        assign_routed_experts_streams([Replay()], models)
