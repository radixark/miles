from argparse import Namespace

import torch

from miles.backends.megatron_utils.parallel import get_packed_seq_params


def test_thd_packed_seq_params_mark_every_packed_sample_for_mamba() -> None:
    # Three samples of 5, 2 and 4 tokens, then 5 tokens of padding appended as its own segment.
    cu_seqlens_host = (0, 5, 7, 11, 16)
    batch = {
        "cu_seqlens": torch.tensor(cu_seqlens_host, dtype=torch.int),
        "cu_seqlens_host": cu_seqlens_host,
        "max_seqlen": 5,
    }

    params = get_packed_seq_params(batch, Namespace(qkv_format="thd"))

    assert params.total_tokens == 16
    expected = torch.tensor([[0] * 5 + [1] * 2 + [2] * 4 + [3] * 5], dtype=torch.int32)
    torch.testing.assert_close(params.seq_idx, expected)
