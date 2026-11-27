"""A flat trie must refuse a layout whose segments do not account for its tokens and positions."""

import pytest

from miles.utils.flat_token_trie import FlatTrie, FlatTrieSegment


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        pytest.param(
            (FlatTrieSegment(parent_idx=None, length=2, start_depth=0),),
            "3 token ids for segments of 2 tokens",
            id="token-ids-the-segments-do-not-cover",
        ),
        pytest.param(
            (
                FlatTrieSegment(parent_idx=None, length=1, start_depth=0),
                FlatTrieSegment(parent_idx=None, length=2, start_depth=5),
            ),
            "segment 1 starts at depth 5, expected 0",
            id="root-off-depth-0",
        ),
    ],
)
def test_a_malformed_flat_trie_is_rejected(segments, message):
    """The layout gathers a segment's ancestors as its first ``start_depth`` keys, so a root off depth 0, or
    token ids the segments do not account for, would gather keys that are not there."""
    with pytest.raises(ValueError, match=message):
        FlatTrie(token_ids=(1, 2, 3), segments=segments)
