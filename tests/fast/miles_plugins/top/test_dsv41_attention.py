"""DeepSeek-V4.1 TOP attention: what the trainer binding refuses instead of running another program."""

import pytest
import torch

from miles_plugins.top.dsv41_attention import TopDSV41Attention


def _bare(installed):
    module = TopDSV41Attention.__new__(TopDSV41Attention)
    torch.nn.Module.__init__(module)
    module.installed = installed
    return module


def test_uninstalled_weights_are_refused():
    """Megatron's initialized weights are not the checkpoint serving runs."""
    with pytest.raises(RuntimeError, match="not installed"):
        _bare(False).forward(torch.zeros(4, 1, 8))


def test_unpacked_batches_are_refused():
    """The runtime prefills one request per packed segment; a padded BSHD batch would attend across padding."""
    with pytest.raises(NotImplementedError, match="packed THD"):
        _bare(True).forward(torch.zeros(4, 1, 8), packed_seq_params=None)
