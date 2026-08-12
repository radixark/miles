"""Interaction environment for any HUD task, driven by the task row itself.

A HUD taskset row (the format its HuggingFace datasets ship in) is:

    prompt          the instruction
    mcp_config      {"<name>": {"url": ..., "headers": {"Mcp-Image": <image>}}}
    setup_tool      {"name": ..., "arguments": {...}}   -- one MCP call
    evaluate_tool   {"name": ..., "arguments": {...}}   -- one MCP call, returns reward
    system_prompt / agent_config

so an episode is: bring the image up, make the setup call, loop
{screenshot -> model -> computer call}, make the evaluate call. The row travels
in ``sample.metadata["hud_task"]`` (see make_hud_data.py).

Which image to boot, how to set the task up and how to grade it are entirely the
row's business, so a different taskset is a data change. What this file does fix
is the *action vocabulary*: press / click / type / done, keys passed through to
the image's ``computer`` tool as the model wrote them. Only that vocabulary, not
any one task, is the assumption -- a taskset needing drag or scroll adds a verb
here, and a taskset where one turn should be one keystroke sets
``hud_keys_per_turn: 1`` (which also turns off the use-your-whole-budget
coaching below).

The model acts through a text DSL rather than provider tool-calling, because
the emitted tokens *are* the training signal: the loss is on the text the
policy generates, so an action has to be text the policy generated.

Reward: the row's ``evaluate_tool`` is the task's own grade and is always
recorded as ``task_reward``. Training can optimize a denser surrogate from the
same environment via ``hud_reward_tool`` -- for 2048 the graded metric is
log2(max_tile)/log2(target), a five-level step function that goes flat the
moment a population converges on one tile, while score is continuous.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Any

from examples.experimental.hud.sandbox import HudSandbox, get_gate, start_reaper
from PIL import Image

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# `Action: verb(arg, arg)` on the last such line of a response.
_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\(([^)]*)\)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")
_INT_RE = re.compile(r"-?\d+")
_QUOTED_RE = re.compile(r"[\"']([^\"']*)[\"']")

# Capitalization the arrow keys are known to work with. Everything else is
# passed to the image's ``computer`` tool as the model wrote it: a whitelist here
# would silently drop every key a desktop task needs.
_ARROWS = {"left": "Left", "right": "Right", "up": "Up", "down": "Down"}


def parse_keys(raw: str) -> list[list[str]]:
    """``press`` arguments as a list of key-groups, in order.

    Each group is one MCP ``press`` call. A group of several keys is a chord
    (``"ctrl+c"``); separate arguments are pressed in sequence, which is what
    makes a per-turn move budget mean anything -- one call with ``keys=[a,b,c]``
    is a pyautogui hotkey, not three moves.

    Quoted arguments are taken as written, so any key the image accepts works.
    Unquoted ones fall back to arrow names only: ``press(Left and then Down)``
    is common enough to honour, and prose is not distinguishable from key names
    without the quotes, so pressing every word would just burn the move budget.
    """
    tokens = _QUOTED_RE.findall(raw)
    if not tokens:
        tokens = [w for w in _WORD_RE.findall(raw) if w.lower() in _ARROWS]
    groups = []
    for token in tokens:
        keys = [_ARROWS.get(k.lower(), k) for k in token.split("+") if k.strip()]
        if keys:
            groups.append(keys)
    return groups


def parse_action(text: str) -> tuple[str, str] | None:
    """The last ``Action:`` line as (verb, raw-args), or None if absent."""
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    verb, raw = matches[-1]
    return verb.lower(), raw


def image_from_b64(data: str, width: int) -> tuple[Image.Image, int]:
    """Decode a screenshot, downscaled to *width*. Also returns its native width,
    which is the scale factor for click coordinates the model gives in the
    downscaled view."""
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    native = img.width
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    return img, native


def image_of(task: dict) -> str:
    """The container image a HUD row asks for (its ``Mcp-Image`` header)."""
    for entry in (task.get("mcp_config") or {}).values():
        if isinstance(entry, dict):
            image = (entry.get("headers") or {}).get("Mcp-Image")
            if image:
                return image
    raise ValueError(f"no Mcp-Image in mcp_config: {task.get('mcp_config')!r}")


def _as_calls(spec: Any) -> list[dict]:
    """Normalize a row's tool field: one call, a list of calls, or JSON text."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = json.loads(spec)
    if isinstance(spec, dict):
        return [spec]
    return [c for c in spec if isinstance(c, dict)]


class HudTaskEnv:
    """One episode against one HUD task. Synchronous by design: the rollout
    calls it from a worker thread so its network I/O never blocks the event
    loop shared by every concurrent episode."""

    def __init__(self, sample: Sample | None, args: Any) -> None:
        self.args = args
        meta = (getattr(sample, "metadata", None) or {}) if sample is not None else {}
        self.task: dict = meta.get("hud_task") or getattr(args, "hud_task", None) or {}
        if not self.task:
            raise ValueError("no HUD task row: expected sample.metadata['hud_task']")

        self.shot_width = int(getattr(args, "hud_screenshot_width", 640))
        self.keys_per_turn = int(getattr(args, "hud_keys_per_turn", 4))
        self.reward_tool = getattr(args, "hud_reward_tool", None)
        max_age = float(getattr(args, "hud_sandbox_max_age_min", 20))

        self.sandbox = HudSandbox(image_of(self.task), max_age_min=max_age)
        self.gate = get_gate(int(getattr(args, "hud_max_sandboxes", 32)))
        start_reaper(max_age)

        self.mcp = None
        self.native_width = 0
        self.turn = 0
        self.actions = 0
        self.parse_failures = 0
        self.short_actions = 0

    # ---- lifecycle ----

    def reset(self):
        self.mcp = self.sandbox.start(self.gate)
        for call in _as_calls(self.task.get("setup_tool")):
            result = self.mcp.call_tool(call["name"], call.get("arguments") or {})
            logger.debug("[hud %s] setup %s -> %s", self.sandbox.run_id, call["name"], result.text[:120])
        time.sleep(2)  # let the app paint before the first screenshot
        self.turn = self.actions = self.parse_failures = self.short_actions = 0
        return {}, {"run_id": self.sandbox.run_id}

    def close(self):
        try:
            if self.mcp is not None:
                self.mcp.close()
        finally:
            self.mcp = None
            self.sandbox.delete()

    # ---- interaction ----

    def _screenshot(self) -> Image.Image:
        result = self.mcp.call_tool("computer", {"action": "screenshot"})
        if not result.image_b64:
            raise RuntimeError(f"screenshot returned no image: {result.text[:200]!r}")
        img, self.native_width = image_from_b64(result.image_b64, self.shot_width)
        return img

    def _do(self, verb: str, raw: str) -> str:
        """Execute one parsed action. Returns a note to show the model, if any."""
        if verb == "press":
            groups = parse_keys(raw)[: self.keys_per_turn]
            if not groups:
                return 'No key in that action, e.g. press("Left") or press("ctrl+c"). '
            for keys in groups:
                self.mcp.call_tool("computer", {"action": "press", "keys": keys})
                self.actions += 1
                time.sleep(0.25)
            if len(groups) < self.keys_per_turn:
                self.short_actions += 1
                return (
                    f"You gave {len(groups)} key(s); give exactly {self.keys_per_turn} keys "
                    f"per action to use the turn fully. "
                )
            return ""
        if verb == "click":
            xy = [int(v) for v in _INT_RE.findall(raw)[:2]]
            if len(xy) < 2:
                return "click needs two coordinates, e.g. click(840, 320). "
            if not self.native_width:
                return "Take Action: screenshot() before clicking, so you can see where to click. "
            # Coordinates are in the model's downscaled view; scale back up to
            # the screen the environment actually renders.
            scale = self.native_width / self.shot_width if self.shot_width else 1
            self.mcp.call_tool("computer", {"action": "click", "x": round(xy[0] * scale), "y": round(xy[1] * scale)})
            self.actions += 1
            time.sleep(0.4)
            return ""
        if verb in ("type", "write"):
            m = _QUOTED_RE.search(raw)
            self.mcp.call_tool("computer", {"action": "write", "text": m.group(1) if m else raw.strip()})
            self.actions += 1
            time.sleep(0.4)
            return ""
        if verb in ("screenshot", "look"):
            return ""  # the observation below is the screenshot
        return f'Unknown action "{verb}". '

    def step(self, response_text: str):
        self.turn += 1
        parsed = parse_action(response_text)

        if parsed is None:
            self.parse_failures += 1
            if self.parse_failures >= 3:
                return {}, True, {"reason": "parse_failures"}
            note = 'Invalid action format. Put e.g. Action: press("Left","Down") on the last line. '
        else:
            self.parse_failures = 0
            verb, raw = parsed
            if verb in ("done", "stop", "submit"):
                return {}, True, {"reason": "done"}
            note = self._do(verb, raw)

        obs = {
            "multi_modal_data": {"image": [self._screenshot()]},
            "obs_str": f"{note}Turn {self.turn}, actions so far: {self.actions}. Next action?",
        }
        return obs, False, {}

    def format_observation(self, observation: dict) -> dict:
        content: list[dict] = []
        for images in (observation.get("multi_modal_data") or {}).values():
            content.extend({"type": "image", "image": img} for img in images)
        content.append({"type": "text", "text": observation.get("obs_str", "")})
        return {"role": "user", "content": content}

    # ---- grading ----

    def _grade(self, call: dict) -> dict:
        result = self.mcp.call_tool(call["name"], call.get("arguments") or {})
        payload = result.payload()
        return {
            # A grader may answer with a null reward or with content blocks
            # rather than a string; neither should take an episode down.
            "reward": float(payload.get("reward") or 0.0),
            "done": bool(payload.get("done", False)),
            "info": payload.get("info"),
            "content": str(payload.get("content") or result.text)[:200],
        }

    def compute_final_reward(self) -> dict:
        """Grade with the task's own evaluate_tool, plus the training surrogate.

        ``task_reward`` is the number the taskset defines and the one to report;
        ``reward`` is what GRPO optimizes and may be the denser surrogate.
        """
        out: dict[str, Any] = {"turns": self.turn, "actions": self.actions, "short_actions": self.short_actions}

        task_calls = _as_calls(self.task.get("evaluate_tool"))
        graded = self._grade(task_calls[0]) if task_calls else {"reward": 0.0, "info": None, "content": ""}
        out["task_reward"] = graded["reward"]
        out["task_grade"] = graded["content"]
        if isinstance(graded.get("info"), dict):
            out.update({f"task_{k}": v for k, v in graded["info"].items()})

        surrogate = _as_calls(self.reward_tool)
        if surrogate:
            dense = self._grade(surrogate[0])
            out["reward"] = dense["reward"]
            if isinstance(dense.get("info"), dict):
                out.update({f"dense_{k}": v for k, v in dense["info"].items()})
        else:
            out["reward"] = graded["reward"]
        return out


def build_env(sample: Sample | None = None, args: Any = None) -> HudTaskEnv:
    return HudTaskEnv(sample, args)
