"""Offline tests: no GPU, no network, no Daytona, no live model.

make_hud_data.py's translation: a HUD task template -> one training
row.
The HUD side of the seam is faked at exactly the boundary the real code
touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

hud = pytest.importorskip("hud", reason="pip install hud -- the recipe's one extra dependency")

from examples.experimental.hud.make_hud_data import task_row


def test_task_row_carries_what_the_rollout_needs():
    task = SimpleNamespace(env="browser", id="2048-score", args={"target_score": 5000}, slug="s")
    row = task_row(task)
    assert row == {"env": "browser", "id": "2048-score", "args": {"target_score": 5000}, "slug": "s"}
