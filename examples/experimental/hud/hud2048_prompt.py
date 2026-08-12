"""2048's briefing -- the one taskset-specific string in this recipe.

It lives apart from ``make_hud_data.py`` so that file stays readable as what it
is: a generic taskset-to-jsonl converter. A row's own ``system_prompt`` and
``prompt`` are what normally reach the model; this substitutes for both, because
2048's shipped prompts are written for a tool-calling browser agent and name a
max-tile goal, while this recipe trains on score.

The strategy paragraph is a deliberately imperfect prior, kept as-is. It is sound
advice for a human playing 2048 with unlimited moves and a highest-tile goal, and
it is wrong for this setup: an episode gets ~89 key presses and is scored on
merges, and pressing only Left/Down wastes most of that budget on no-ops
(measured: 48 presses -> 14 effective moves, score 60; using all four directions
-> 48 moves, score 368). Training discovers this on its own -- Up + Right rose
from 5% to 34% of presses while mean score went 200 -> 712 -- which is the
clearest evidence in this recipe that the policy learns from the environment
rather than reciting the prompt. Fixing the hint would remove that signal.
"""

import json
from typing import Any

GAME_2048_HINT = """You are playing 2048 in a browser. Tiles slide with arrow keys; equal tiles \
merge and double. Your score grows with every merge, so keep merging.

Strategy: keep your biggest tile in one corner and build a descending row toward it. Prefer Left \
and Down. Use Right only when Left and Down both change nothing. Avoid Up, which breaks the corner."""


def applies(setup_tool: Any) -> bool:
    """True for rows whose setup call launches 2048."""
    return "2048" in json.dumps(setup_tool or {})
