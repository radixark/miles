"""The trainer's slot transport runs on serving's slot configs; Megatron's DeepEP buffer is sized to hold them."""

from unittest.mock import patch

import pytest


def test_slot_buffer_holds_the_slot_configs():
    # deep_ep refuses to import without a usable CUDA device.
    deep_ep = pytest.importorskip("deep_ep", exc_type=ImportError)
    from miles_plugins.top import deepep_transport as tx

    slot = (deep_ep.Config(64, 16, 128, 6, 128), deep_ep.Config(64, 4, 128, 6, 128))
    with patch.object(tx, "_slot_configs", return_value=slot):
        for hidden_bytes in (32, 512, 4096, 4608):
            sized = tx._sized_hidden_bytes(hidden_bytes, 8)
            assert sized >= hidden_bytes
            assert tx._default_nvl_bytes(sized, 8) >= max(c.get_nvl_buffer_size_hint(hidden_bytes, 8) for c in slot)
    with patch.object(tx, "_slot_configs", return_value=(None, None)):
        assert tx._sized_hidden_bytes(4096, 8) == 4096
