import pickle
from argparse import Namespace
from dataclasses import fields

import torch

from miles.backends.megatron_utils.parallel import PackedSeqParamsWithCPLayout, get_packed_seq_params
from miles_plugins.models.cp_layout import build_cp_layout_metadata


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


def test_cp_layout_is_shared_and_not_a_transformer_engine_field():
    layout = build_cp_layout_metadata((0, 8, 12), 2, 0)
    batch = {
        "cu_seqlens": torch.tensor(layout.global_boundaries, dtype=torch.int),
        "cu_seqlens_host": layout.global_boundaries,
        "max_seqlen": 8,
        "cp_layout_metadata": layout,
    }
    params = get_packed_seq_params(batch, Namespace(qkv_format="thd"))
    assert isinstance(params, PackedSeqParamsWithCPLayout)
    assert params.cp_layout_metadata is layout
    assert "cp_layout_metadata" not in {item.name for item in fields(params)}
    restored = pickle.loads(pickle.dumps(params))
    assert restored.cp_layout_metadata == layout
    assert restored.cu_seqlens_host == layout.global_boundaries
