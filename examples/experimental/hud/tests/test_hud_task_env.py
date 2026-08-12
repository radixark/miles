"""Offline tests for the HUD task adapter: no Daytona, no network, no GPU.

The env's whole job is translating between three contracts -- the taskset row,
the model's text actions, and MCP tool calls -- so these fake the MCP side and
assert on the translation. The bugs these lock down were all found the
expensive way, mid-training.
"""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from examples.experimental.hud import hud_task_env as env_mod
from examples.experimental.hud.hud_task_env import HudTaskEnv, image_of, parse_action
from examples.experimental.hud.make_hud_data import build_prompt, task_of
from examples.experimental.hud.mcp_client import ToolResult
from PIL import Image

TASK = {
    "id": "t1",
    "mcp_config": {
        "browser": {"url": "https://mcp.hud.ai/v3/mcp", "headers": {"Mcp-Image": "hudevals/hud-browser:0.1.6"}}
    },
    "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
    "evaluate_tool": {"name": "evaluate", "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}}},
}


def _png_b64(w: int = 1920, h: int = 1080) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeMCP:
    """Records calls; answers screenshots with a real PNG and grades with JSON."""

    def __init__(self, grades: dict[str, dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.grades = grades or {}
        self.closed = False

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "computer" and arguments.get("action") == "screenshot":
            return ToolResult("", _png_b64(), None)
        if name == "evaluate":
            inner = arguments.get("name") or ""
            payload = self.grades.get(inner, {"reward": 0.0})
            return ToolResult(json.dumps(payload), None, payload)
        return ToolResult("ok", None, None)

    def close(self) -> None:
        self.closed = True

    def presses(self) -> list[str]:
        return [a["keys"][0] for n, a in self.calls if n == "computer" and a.get("action") == "press"]


@pytest.fixture
def env(monkeypatch):
    """An env whose sandbox is stubbed out; reset() yields the FakeMCP."""
    fake = FakeMCP(
        grades={
            "game_2048_max_number": {
                "reward": 0.5714,
                "done": False,
                "content": "Target: 128, Highest: 16",
                "info": {"highest_tile": 16},
            },
            "game_2048_score_reached": {
                "reward": 0.37,
                "done": False,
                "content": "Score: 190 (target: 512)",
                "info": {"score": 190},
            },
        }
    )
    monkeypatch.setattr(env_mod, "start_reaper", lambda *a, **k: None)
    # The env sleeps to let apps paint and merges animate; irrelevant offline.
    monkeypatch.setattr(env_mod.time, "sleep", lambda *_: None)

    class StubSandbox:
        run_id = "stub1234"

        def __init__(self, *a, **k):
            self.deleted = False

        def start(self, gate, **k):
            return fake

        def delete(self):
            self.deleted = True

    monkeypatch.setattr(env_mod, "HudSandbox", StubSandbox)

    def make(**overrides):
        conf = {
            "hud_screenshot_width": 640,
            "hud_keys_per_turn": 4,
            "hud_reward_tool": None,
            "hud_max_sandboxes": 4,
            "hud_sandbox_max_age_min": 20,
        }
        conf.update(overrides)
        args = SimpleNamespace(**conf)
        sample = SimpleNamespace(metadata={"hud_task": TASK})
        e = HudTaskEnv(sample, args)
        e.reset()
        return e, fake

    return make


# ---- pure parsing ----


@pytest.mark.parametrize(
    "text, expected",
    [
        ('Action: press("Left","Down","Left","Down")', ("press", ["Left", "Down", "Left", "Down"])),
        ("Thinking.\nAction: press(Left, Down)", ("press", ["Left", "Down"])),
        ("Action: screenshot()", ("screenshot", [])),
        # Only the LAST action line counts: models often restate the format.
        ('Action: press("Up")\nActually no.\nAction: press("Down")', ("press", ["Down"])),
        ("no action at all", None),
    ],
)
def test_parse_action(text, expected):
    got = parse_action(text)
    if expected is None:
        assert got is None
    else:
        verb, keys = expected
        assert got[0] == verb
        assert [w for w in env_mod._WORD_RE.findall(got[1])] == keys


def test_image_of_reads_the_row_not_a_constant():
    assert image_of(TASK) == "hudevals/hud-browser:0.1.6"
    with pytest.raises(ValueError):
        image_of({"mcp_config": {"x": {"headers": {}}}})


# ---- action execution ----


def test_four_keys_become_four_separate_presses(env):
    """A single call with keys=[a,b,c] is a chord, not three moves, so the env
    must issue one press per key -- this is what makes the move budget real."""
    e, fake = env()
    e.step('Action: press("Left","Down","Left","Down")')
    assert fake.presses() == ["Left", "Down", "Left", "Down"]
    assert e.actions == 4


def test_non_arrow_keys_pass_through(env):
    """2048 only ever needs arrows, but a whitelist here would silently drop
    every key a desktop task needs."""
    e, fake = env()
    e.step('Action: press("Enter","Tab","F5","Escape")')
    assert fake.presses() == ["Enter", "Tab", "F5", "Escape"]


def test_plus_means_a_chord_and_separate_args_mean_a_sequence(env):
    """One call with keys=[a,b] is a pyautogui hotkey, not two keystrokes -- so
    the two have to be expressible apart."""
    e, fake = env()
    e.step('Action: press("ctrl+c","Left")')
    pressed = [a["keys"] for n, a in fake.calls if n == "computer" and a.get("action") == "press"]
    assert pressed == [["ctrl", "c"], ["Left"]]


def test_unquoted_prose_does_not_become_keystrokes(env):
    """Without quotes there is no telling a key name from a word, so only arrow
    names count -- otherwise 'and'/'then' would each burn a move."""
    e, fake = env()
    e.step("Action: press(Left and then Down)")
    assert fake.presses() == ["Left", "Down"]


def test_extra_keys_are_clipped_to_the_cap(env):
    e, fake = env()
    e.step('Action: press("Left","Down","Left","Down","Left","Down")')
    assert len(fake.presses()) == 4


def test_short_action_is_counted_and_flagged_to_the_model(env):
    """A model that emits one key per turn silently caps the episode's reward
    ceiling; the note and the counter make that visible instead."""
    e, _ = env()
    obs, done, _ = e.step('Action: press("Left")')
    assert not done
    assert e.short_actions == 1
    assert "exactly 4 keys" in obs["obs_str"]


def test_click_scales_coordinates_back_to_the_real_screen(env):
    """The model gives coordinates in the downscaled view it was shown; the
    scale factor comes from the screenshot the environment actually sent."""
    e, fake = env(hud_screenshot_width=640)
    e.step("Action: screenshot()")
    e.step("Action: click(320, 180)")
    click = [a for n, a in fake.calls if n == "computer" and a.get("action") == "click"][-1]
    assert (click["x"], click["y"]) == (960, 540)  # 640 -> 1920 is x3


def test_click_before_any_screenshot_is_refused(env):
    """Without a screenshot there is no scale factor, and the model has no basis
    for the coordinates either -- clicking blind would land anywhere."""
    e, fake = env()
    obs, done, _ = e.step("Action: click(320, 180)")
    assert not done
    assert not [a for n, a in fake.calls if n == "computer" and a.get("action") == "click"]
    assert "screenshot()" in obs["obs_str"]


def test_done_ends_the_episode(env):
    e, _ = env()
    _, done, info = e.step("Action: done()")
    assert done and info["reason"] == "done"


def test_three_unparseable_turns_end_the_episode(env):
    e, _ = env()
    assert e.step("junk")[1] is False
    assert e.step("junk")[1] is False
    assert e.step("junk")[1] is True


def test_observation_carries_the_downscaled_screenshot(env):
    e, _ = env()
    obs, _, _ = e.step("Action: screenshot()")
    msg = e.format_observation(obs)
    assert msg["role"] == "user"
    images = [c["image"] for c in msg["content"] if c["type"] == "image"]
    assert len(images) == 1 and images[0].size == (640, 360)


# ---- setup and grading ----


def test_setup_call_comes_from_the_row(env):
    _, fake = env()
    assert fake.calls[0] == ("launch_app", {"app_name": "2048"})


def test_grade_uses_the_rows_evaluate_tool_by_default(env):
    e, _ = env()
    verdict = e.compute_final_reward()
    assert verdict["reward"] == pytest.approx(0.5714)
    assert verdict["task_reward"] == pytest.approx(0.5714)
    assert verdict["task_highest_tile"] == 16


def test_surrogate_reward_trains_while_the_row_grade_is_still_reported(env):
    """Training optimizes the dense score; the taskset's own grade stays the
    reported metric."""
    e, _ = env(
        hud_reward_tool={
            "name": "evaluate",
            "arguments": {"name": "game_2048_score_reached", "arguments": {"target_score": 512}},
        }
    )
    verdict = e.compute_final_reward()
    assert verdict["reward"] == pytest.approx(0.37)  # dense surrogate drives GRPO
    assert verdict["task_reward"] == pytest.approx(0.5714)  # benchmark number kept
    assert verdict["dense_score"] == 190


def test_close_releases_the_sandbox_and_the_client(env):
    e, fake = env()
    e.close()
    assert fake.closed and e.sandbox.deleted


# ---- prompt building ----


def test_prompt_keeps_the_rows_instruction():
    """For any taskset but 2048 the row's prompt *is* the task, so dropping it
    would leave the model with an action vocabulary and nothing to do."""
    row = {
        "system_prompt": "You operate a desktop.",
        "prompt": "Open the spreadsheet and total column B.",
        "setup_tool": {"name": "launch_app", "arguments": {"app_name": "calc"}},
    }
    prompt = build_prompt(row, keys_per_turn=4)
    assert "Open the spreadsheet and total column B." in prompt
    assert "You operate a desktop." in prompt
    assert "Action: click(840, 320)" in prompt  # our DSL, not the row's


def test_2048_row_gets_the_recipes_own_briefing():
    """2048's shipped prompts target a tool-calling agent and a max-tile goal;
    this recipe trains on score, so it substitutes its own."""
    row = {
        "system_prompt": "Use the browser tools.",
        "prompt": "Reach the 512 tile.",
        "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
    }
    prompt = build_prompt(row, keys_per_turn=4)
    assert "Reach the 512 tile." not in prompt and "browser tools" not in prompt
    assert "Tiles slide with arrow keys" in prompt


def test_press_coaching_only_when_there_is_a_key_budget():
    """Telling the model to use its whole key budget is advice for a task with a
    move budget and near-free keystrokes. A taskset where a turn is one
    deliberate keystroke must not be told to burn four, and needs the chord form
    instead."""
    row = {"prompt": "Total column B.", "setup_tool": {"name": "launch_app", "arguments": {"app_name": "calc"}}}
    assert "use all 4" in build_prompt(row, keys_per_turn=4)
    single = build_prompt(row, keys_per_turn=1)
    assert "use all" not in single
    assert "ctrl+c" in single


def test_task_of_extracts_what_the_env_needs():
    assert task_of({"mcp_config": json.dumps(TASK["mcp_config"]), "id": "t1"})["mcp_config"] == TASK["mcp_config"]


def test_missing_task_row_fails_loudly():
    with pytest.raises(ValueError, match="no HUD task row"):
        HudTaskEnv(SimpleNamespace(metadata={}), SimpleNamespace())
