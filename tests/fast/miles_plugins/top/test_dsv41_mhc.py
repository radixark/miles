"""DeepSeek-V4.1 TOP mHC: what the trainer binding refuses instead of running another program."""

import pytest
import torch

from miles_plugins.top.dsv41_mhc import top_collapse


def test_the_head_refuses_a_missing_pre():
    """Without the last hyper-connection's pre the head would fall back to one stream, not serving's collapse."""
    with pytest.raises(RuntimeError, match="last hyper-connection's pre"):
        top_collapse(torch.zeros(2, 1, 32, dtype=torch.bfloat16), None, 4)


def test_the_head_refuses_a_bf16_pre():
    """serving carries pre in FP32; a rounded pre is another program."""
    with pytest.raises(NotImplementedError, match="FP32 pre"):
        top_collapse(torch.zeros(2, 1, 32, dtype=torch.bfloat16), torch.zeros(2, 4, dtype=torch.bfloat16), 4)
