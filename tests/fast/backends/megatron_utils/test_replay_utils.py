from types import SimpleNamespace

from miles.backends.megatron_utils import replay_utils
from miles.backends.megatron_utils.replay_utils import register_replay_list_moe, replay_blocks


def test_a_chunk_without_decoder_layers_is_not_a_block_to_read(monkeypatch):
    # The first chunk holds only the embedding (--account-for-embedding-in-pipeline-split).
    monkeypatch.setattr(replay_utils, "vp_stage_layers", lambda models: [(0, 0), (0, 2)])
    manager = SimpleNamespace(register_replay_list_func=register_replay_list_moe)

    assert replay_blocks(manager, models=None) == [(0, 2)]
