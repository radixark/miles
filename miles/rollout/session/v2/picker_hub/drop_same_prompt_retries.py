"""The default pick hook: trim retries, i.e. identical re-sends."""

from miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves import drop_rolled_back_leaves
from miles.utils.types import Sample


def drop_same_prompt_retries(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Drop the leaves that retries leave behind; a retry re-sends the same prompt tokens.

    Like ``drop_rolled_back_leaves``, but a later sibling with a different
    request is a new branch, so both leaves stay samples.
    """
    return drop_rolled_back_leaves(leaf_samples, session_metadata, sibling_prompt_must_match=True)
