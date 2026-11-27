import pytest

from miles_plugins.top import glm_indexer
from miles_plugins.top.glm_ops import TopGLMOps


@pytest.mark.parametrize("num_keys", [1, 3584, 32768])
def test_short_contexts_keep_the_native_block(num_keys):
    """Up to 32k keys an 8192-row block already fits the budget, so selection keeps the native block."""
    assert TopGLMOps.select_block_rows(num_keys, 8192) == 8192


@pytest.mark.parametrize("num_keys", [32769, 65536, 131072, 1 << 20, 1 << 24])
def test_long_contexts_bound_the_logits_block(num_keys):
    rows = TopGLMOps.select_block_rows(num_keys, 8192)
    assert rows % 64 == 0 and 64 <= rows < 8192
    assert rows * num_keys * 4 <= glm_indexer._SELECT_LOGITS_BYTES or rows == 64


def test_128k_keys_select_in_2048_row_blocks():
    """At 8192 rows, 128k keys take 8.1 GiB of transient logits; 2048-row blocks bound them."""
    assert TopGLMOps.select_block_rows(131072, 8192) == 2048
