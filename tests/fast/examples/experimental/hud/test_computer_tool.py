"""Offline tests: no GPU, no network, no Daytona, no live model.

computer_tool.py's translation: function-call arguments -> RFB
primitives, with coordinate scaling and focus semantics.
The HUD side of the seam is faked at exactly the boundary the real code
touches.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

hud = pytest.importorskip("hud", reason="pip install hud -- the recipe's one extra dependency")

from examples.experimental.hud.computer_tool import ChatComputerTool, computer_tool_class, to_keysym


class _Recorder:
    """Stands in for the RFB primitives; records what dispatch decided."""

    def __init__(self, tool):
        self.calls = []
        tool.click = self._make("click")
        tool.press_keys = self._make("press_keys")
        tool.type_text = self._make("type_text")
        tool.scroll = self._make("scroll")
        tool.wait = self._make("wait")

        async def _obs():
            return "OBS"

        tool._observation = _obs

    def _make(self, name):
        async def record(*a, **k):
            self.calls.append((name, a, k))

        return record


def _tool(shot_width=640, display_width=1920, focused=True):
    cls = computer_tool_class(shot_width)
    tool = cls.__new__(cls)  # skip RFBTool.__init__: no live VNC in tests
    tool.spec = cls.default_spec("m")
    tool.client = SimpleNamespace(width=display_width, height=1080)
    # Most tests are about translation, not the one-time focus click.
    tool._pointer_used = focused
    return tool


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_press_sequence_is_one_call_per_key():
    """press(['Left','Down']) must be two keystrokes; a single call with both
    keys would be a chord (hotkey), and 2048 would see one move, not two."""
    tool = _tool()
    rec = _Recorder(tool)
    _run(tool.execute({"action": "press", "keys": ["Left", "Down"]}))
    presses = [c for c in rec.calls if c[0] == "press_keys"]
    assert [c[1][0] for c in presses] == [["Left"], ["Down"]]


def test_plus_means_chord():
    tool = _tool()
    rec = _Recorder(tool)
    _run(tool.execute({"action": "press", "keys": ["ctrl+c"]}))
    presses = [c for c in rec.calls if c[0] == "press_keys"]
    assert presses[0][1][0] == ["Control_L", "c"]


def test_first_keystroke_takes_focus_and_only_once():
    """A screen that has focused nothing drops opening keystrokes silently, so
    the tool establishes focus once -- and only once, since a click carries
    meaning of its own."""
    tool = _tool(focused=False)
    rec = _Recorder(tool)
    _run(tool.execute({"action": "press", "keys": ["Left"]}))
    _run(tool.execute({"action": "press", "keys": ["Down"]}))
    assert len([c for c in rec.calls if c[0] == "click"]) == 1


def test_a_click_of_the_models_own_counts_as_focus():
    """The model clicking where it means to type is the real thing; the
    tool's fallback must not fire on top of it."""
    tool = _tool(focused=False)
    rec = _Recorder(tool)
    _run(tool.execute({"action": "click", "x": 10, "y": 20}))
    _run(tool.execute({"action": "press", "keys": ["Left"]}))
    clicks = [c for c in rec.calls if c[0] == "click"]
    assert len(clicks) == 1 and clicks[0][1] == (30, 60)  # the model's, scaled


def test_key_aliases_map_to_keysyms():
    assert to_keysym("enter") == "Return"
    assert to_keysym("arrowleft") == "Left"
    assert to_keysym("F5") == "F5"  # unknown keys pass through untouched


def test_click_scales_from_shot_to_screen():
    """The model clicks in the 640px frame it saw; the screen is 1920px."""
    tool = _tool(shot_width=640, display_width=1920)
    rec = _Recorder(tool)
    _run(tool.execute({"action": "click", "x": 320, "y": 180}))
    clicks = [c for c in rec.calls if c[0] == "click"]
    assert clicks[0][1] == (960, 540)


def test_unknown_action_is_an_error_not_a_crash():
    tool = _tool()
    _Recorder(tool)
    result = _run(tool.execute({"action": "fly"}))
    assert result.isError


def test_tool_params_are_valid_function_schema():
    tool = _tool()
    params = tool.to_params()
    assert params["type"] == "function"
    assert params["function"]["name"] == "computer"
    assert "press" in params["function"]["parameters"]["properties"]["action"]["enum"]


def test_shot_width_travels_on_the_class():
    assert computer_tool_class(640) is ChatComputerTool
    assert computer_tool_class(960).shot_width == 960
