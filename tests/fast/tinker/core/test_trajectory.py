"""Coverage plan for turns → sequences; every test is a skeleton until WP-3 lands."""

import pytest

SKELETON = pytest.mark.skip(reason="skeleton: implementation lands with trajectory.py (WP-3)")


@SKELETON
def test_chain_merges_extending_turns():
    """Turns whose input_ids extend the previous input_ids + output_ids merge into one sequence with gaps masked out."""


@SKELETON
def test_chain_returns_none_on_prefix_break():
    """A turn that does not extend the previous one makes chain() return None."""


@SKELETON
def test_per_turn_masks_only_outputs():
    """per_turn() gives one sequence per turn with loss_mask 1 exactly on the output ids."""


@SKELETON
def test_datum_arrays_lengths_and_zero_prefix():
    """datum_arrays: target_tokens = tokens[1:], logprobs/advantages are zero before the first loss position, all len(tokens) - 1."""
