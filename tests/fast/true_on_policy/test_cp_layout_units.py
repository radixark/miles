"""The single-process parts of the Ulysses CP layout: KV replication and the length bookkeeping.

The all-to-all round trip needs a real process group and lives in a gloo script; what is testable
here is the index math around it, which is where the bugs were: the archived version ASSERTED
`num_heads % cp_size == 0` instead of replicating (carrying the `cp <= num_kv_heads` ceiling), and
`local_packed_lengths` is the guard that catches a LOCAL cu_seqlens being passed where a GLOBAL one
is required -- a mistake that would otherwise describe the wrong sequence silently.
"""

from __future__ import annotations

import pytest
import torch

from miles_plugins.top.cp_layout import UlyssesCPLayout, replicate_kv_heads_for_cp

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4  # synthetic head layout for replication coverage


def _kv(num_heads=NUM_KV_HEADS, tokens=8):
    return torch.arange(tokens * num_heads * HEAD_DIM, dtype=torch.float32).reshape(
        tokens, num_heads, HEAD_DIM
    )


# --- KV replication: the thing that lifts the cp <= num_kv_heads ceiling ----------------------

@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_replication_is_a_noop_when_kv_heads_already_cover_cp(cp_size):
    x = _kv()
    assert replicate_kv_heads_for_cp(x, cp_size) is x


@pytest.mark.parametrize("cp_size,expected_heads", [(8, 8), (16, 16)])
def test_replication_expands_kv_heads_past_the_ceiling(cp_size, expected_heads):
    out = replicate_kv_heads_for_cp(_kv(), cp_size)
    assert out.shape[-2] == expected_heads


def test_replication_repeats_each_group_contiguously():
    """Group g must serve cp ranks [g*n, (g+1)*n) so the post-a2a q/kv pairing stays correct."""
    out = replicate_kv_heads_for_cp(_kv(num_heads=2), 4)
    assert out.shape[-2] == 4
    torch.testing.assert_close(out[:, 0], out[:, 1])  # both copies of group 0
    torch.testing.assert_close(out[:, 2], out[:, 3])  # both copies of group 1
    assert not torch.equal(out[:, 1], out[:, 2])      # ...and the groups stay distinct


def test_replication_refuses_a_non_multiple_rather_than_guessing():
    with pytest.raises(ValueError, match="multiple of the KV head count"):
        replicate_kv_heads_for_cp(_kv(num_heads=3), 4)


def test_a_gqa_model_reaches_cp_8_only_by_replicating():
    """The concrete bound: 4 KV heads caps Ulysses at cp=4 unless the heads are replicated."""
    kv = _kv(num_heads=NUM_KV_HEADS)
    layout = UlyssesCPLayout(cp_group=None, cp_size=8)
    with pytest.raises(ValueError, match="divisible by cp_size"):
        layout.sequence_to_head_parallel(kv, torch.tensor([0, 8], dtype=torch.int32))
    replicated = replicate_kv_heads_for_cp(kv, 8)
    assert replicated.shape[-2] % 8 == 0


# --- length bookkeeping: the GLOBAL-vs-LOCAL cu_seqlens guard ---------------------------------

def test_local_lengths_divide_the_global_ones():
    layout = UlyssesCPLayout(cp_group=None, cp_size=4)
    cu = torch.tensor([0, 16, 40], dtype=torch.int32)  # global lengths 16 and 24
    assert layout.local_packed_lengths(cu, local_tokens=10) == [4, 6]


def test_a_length_not_divisible_by_cp_is_refused_and_names_the_collator():
    layout = UlyssesCPLayout(cp_group=None, cp_size=4)
    with pytest.raises(ValueError, match="divisible by cp_size"):
        layout.local_packed_lengths(torch.tensor([0, 18], dtype=torch.int32), local_tokens=4)


def test_passing_a_LOCAL_cu_seqlens_is_caught():
    """The failure this exists for: local cu_seqlens sum to the shard, so lengths/cp are too small."""
    layout = UlyssesCPLayout(cp_group=None, cp_size=4)
    local_cu = torch.tensor([0, 16], dtype=torch.int32)  # already per-rank
    with pytest.raises(ValueError, match="must be GLOBAL"):
        layout.local_packed_lengths(local_cu, local_tokens=16)
