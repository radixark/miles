"""Pick hook: trim the leaves an agent rolled back from, whatever it sent next."""

from miles.rollout.session.v2.picker_hub._supersession import drop_superseded_leaves
from miles.utils.types import Sample


def drop_rolled_back_leaves(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Drop every childless leaf a later sibling branched past, like v1's single-step rollback.

    The new request may differ from the abandoned one (a re-run tool, an edited
    follow-up); the earlier leaf is dropped either way. A re-sent first turn
    opens a new root and supersedes an earlier root leaf only when their prompt
    tokens match.

    Example — turn 2 was rolled back and sent again (``seq`` is commit order):

        seq=0  turn 1: user asks, assistant answers
        ├── seq=1  turn 2 attempt: cut off at the length cap  (leaf) -> trim
        └── seq=2  turn 2 again: the same or an edited message
            └── seq=3  turn 3: the new path continues       (leaf) -> keep
    """
    return drop_superseded_leaves(leaf_samples, session_metadata, sibling_prompt_must_match=False)
