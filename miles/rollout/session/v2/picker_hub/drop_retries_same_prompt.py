"""The default pick hook: trim retries, i.e. identical re-sends."""

from miles.rollout.session.v2.picker_hub._supersession import drop_superseded_leaves
from miles.utils.types import Sample


def drop_retries_same_prompt(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Drop the childless leaves an identical re-send replaced.

    A retry re-sends the same request: a later sibling, or a later root for a
    re-sent first turn, whose prompt tokens equal an earlier childless leaf's
    supersedes that leaf. A later sibling with a different request is a new
    branch, so both leaves stay samples.
    """
    return drop_superseded_leaves(leaf_samples, session_metadata, sibling_prompt_must_match=True)
