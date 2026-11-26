"""GLM-5.3-Flash context parallelism (--allgather-cp): query ids, the KDA CP setup, the startup check — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles_plugins.models.glm5_next.dsa import local_query_token_ids
from miles_plugins.models.glm5_next.glm5_next import get_glm5_next_spec
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention
from miles_plugins.models.hf_attention import detect_and_setup_hybrid_cp


def test_without_context_parallelism_the_queries_are_the_whole_stream():
    assert local_query_token_ids(12, cp_rank=0, cp_size=1, device="cpu") is None


@pytest.mark.parametrize("cp_size", [2, 4])
def test_rank_chunks_tile_the_packed_stream_in_gather_order(cp_size):
    """The keys are gathered in rank order, so rank r's queries must be the r-th contiguous chunk."""
    local = 6
    chunks = [local_query_token_ids(local, cp_rank=rank, cp_size=cp_size, device="cpu") for rank in range(cp_size)]
    torch.testing.assert_close(torch.cat(chunks), torch.arange(local * cp_size))


def test_the_kda_core_gets_the_context_parallel_group():
    """Without the group, the KDA core runs each rank's slice of the stream as if it started a sequence."""
    attention = Glm5NextKDAAttention.__new__(Glm5NextKDAAttention)
    nn.Module.__init__(attention)
    attention.kda = nn.Module()
    model = nn.Module()
    model.attention = attention
    group = object()

    assert detect_and_setup_hybrid_cp(model, group, cp_rank=1, cp_world_size=2) == 1
    assert attention.kda.cp_group is group
    assert (attention.kda.cp_rank, attention.kda.cp_world_size) == (1, 2)
    assert attention.hybrid_cp


def test_context_parallelism_without_allgather_cp_is_refused():
    with pytest.raises(AssertionError, match="--allgather-cp"):
        get_glm5_next_spec(SimpleNamespace(allgather_cp=False), SimpleNamespace(context_parallel_size=2))
