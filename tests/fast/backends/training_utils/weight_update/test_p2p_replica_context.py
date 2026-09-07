"""The CPU replica's load_weights runs under the same parallelism context it was built in."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p"


def test_replica_loads_inside_its_parallelism_context():
    """sglang's sharded weight loaders read the attention-TP rank at call time, which only
    exists while the replica's ParallelismContext is entered; a load outside it asserts."""
    active = {"inside": False}
    seen = []

    @contextmanager
    def fake_context(config):
        active["inside"] = True
        seen.append(config)
        try:
            yield
        finally:
            active["inside"] = False

    replica = MagicMock()
    replica.load_weights.side_effect = lambda tensors: seen.append(("load", active["inside"]))

    protocol = object.__new__(UpdateWeightP2P)
    protocol.is_sender = True
    protocol._transfer_engine_meta_list = [(replica, [SimpleNamespace(session_id="s0")], "cfg-0")]
    protocol.transfer_manager = MagicMock()
    protocol._do_p2p_write_one_session = MagicMock()
    ready = [("model.embed_tokens.weight", torch.zeros(1))]
    protocol._get_transfer_ready_params = lambda tensors: (["model.embed_tokens.weight"], ready)

    with patch(f"{_MODULE}.ParallelismContext", fake_context):
        protocol.send_bucket(list(ready))

    assert seen == ["cfg-0", ("load", True)]
    assert active["inside"] is False
