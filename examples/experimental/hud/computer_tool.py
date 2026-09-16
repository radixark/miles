"""A chat-completions computer tool over HUD's rfb capability.

HUD ships the screen primitives (``hud.agents.tools.rfb.RFBTool``: VNC
screenshot, click, keys, scroll) plus provider-native facades for Claude,
Gemini and OpenAI. Nothing serves plain ``chat.completions`` function calling
-- which is what a self-hosted policy (sglang, vLLM) speaks. This is that
facade, and it is the only "harness" code this example adds: the loop, the
tool plumbing and the trace all stay HUD's.

Two behaviours are training decisions rather than plumbing:

- Screenshots are downscaled to ``shot_width`` before the model sees them.
  Every turn re-prefills the whole screenshot history through the vision
  encoder, so screenshot resolution multiplies the cost of every turn after
  it; pick the smallest width at which the task's detail is still readable
  (the recipe's config annotates its choice). Click coordinates arrive in the
  downscaled frame and are scaled back to the real screen here.
- ``press`` distinguishes keystrokes from key combos, which the underlying
  VNC primitive does not: given several keys at once it holds them all down
  together (a hotkey). So each element of ``keys`` is pressed separately, in
  order -- ``["Left", "Down"]`` is two keystrokes -- and a ``+`` joins keys
  into one combo: ``["ctrl+c"]`` is copy. Without the distinction, a model
  batching several keystrokes into one call would land one chord instead.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, ClassVar

import mcp.types as mcp_types

from hud.agents.openai.tools.computer import OPENAI_KEY_ALIASES
from hud.agents.tools.base import AgentToolSpec, tool_err
from hud.agents.tools.rfb import RFBTool
from hud.types import MCPToolResult

logger = logging.getLogger(__name__)


# HUD's map covers provider spellings ("arrowright", "enter"); models also
# emit bare direction words, and X11 keysyms are case-sensitive, so "right"
# would be a dead key without these.
_EXTRA_ALIASES = {"left": "Left", "right": "Right", "up": "Up", "down": "Down"}


def to_keysym(key: str) -> str:
    """Model-friendly key names to the X11 keysyms asyncvnc wants."""
    k = key.strip()
    return OPENAI_KEY_ALIASES.get(k.lower()) or _EXTRA_ALIASES.get(k.lower(), k)


class ChatComputerTool(RFBTool):
    """Drive the environment's screen via OpenAI-style function calling."""

    name = "computer"
    #: Width the model sees. The real screen stays at native resolution; click
    #: coordinates are scaled back up. Configure via :func:`computer_tool_class`.
    shot_width: ClassVar[int] = 640

    description = (
        "Control the computer's screen, mouse and keyboard. Every action "
        "returns a screenshot taken after it, so you always see the result. "
        "Use action='screenshot' just to look. Keys go to whatever the screen "
        "has focused, so click the thing you mean to type into first."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "click", "double_click", "type", "press", "scroll", "wait"],
                "description": "What to do.",
            },
            "x": {"type": "integer", "description": "click/scroll: x coordinate in the screenshot"},
            "y": {"type": "integer", "description": "click/scroll: y coordinate in the screenshot"},
            "button": {
                "type": "string",
                "enum": ["left", "middle", "right"],
                "description": "click: default left",
            },
            "text": {"type": "string", "description": "type: literal text to type"},
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "press: keys pressed one after another, in order -- send "
                    "several at once when you know the next few, e.g. "
                    "['Left','Down','Left','Down']. An element containing '+' "
                    "is a chord instead, e.g. ['ctrl+c']. Arrow keys: "
                    "Left/Right/Up/Down."
                ),
            },
            "scroll_y": {"type": "integer", "description": "scroll: clicks; >0 down, <0 up"},
            "ms": {"type": "integer", "description": "wait: milliseconds"},
        },
        "required": ["action"],
    }

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return AgentToolSpec(api_type="function", api_name=cls.name)

    def to_params(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    #: Whether the pointer has established keyboard focus this episode.
    _pointer_used: bool = False

    # ---- geometry ----

    @property
    def _scale(self) -> float:
        """Downscaled-frame -> real-screen factor."""
        if self.shot_width and self.display_width > self.shot_width:
            return self.display_width / self.shot_width
        return 1.0

    def _to_screen(self, v: Any) -> int:
        return round(int(v) * self._scale)

    async def _observation(self) -> MCPToolResult:
        """A screenshot at the width the model is trained to look at."""
        result = await self.screenshot()
        if self._scale == 1.0:
            return result
        content: list[Any] = []
        for block in result.content:
            if isinstance(block, mcp_types.ImageContent):
                from PIL import Image  # noqa: PLC0415 - pillow ships with the training image

                img = Image.open(io.BytesIO(base64.b64decode(block.data)))
                w = self.shot_width
                img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "PNG")
                block = mcp_types.ImageContent(
                    type="image",
                    mimeType="image/png",
                    data=base64.b64encode(buf.getvalue()).decode("ascii"),
                )
            content.append(block)
        return MCPToolResult(content=content)

    async def _ensure_focus(self) -> None:
        """Give the screen keyboard focus before the first typed input.

        A desktop routes key events to whatever it has focused, and a freshly
        booted one may have focused nothing -- in which case an episode that
        opens with keystrokes silently does nothing. One click at the centre,
        once per episode, and only before input the pointer has not already
        established focus for.
        """
        if self._pointer_used:
            return
        self._pointer_used = True
        await self.click(self.display_width // 2, self.display_height // 2)
        await self.wait(200)

    # ---- dispatch ----

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        action = str(arguments.get("action", ""))
        # One line per action, at INFO, because the alternative is finding out
        # what the policy did only after the batch lands: a run whose reward
        # stays zero is diagnosed by what it is *doing* (clicking rather than
        # typing, say), and the harness's own log says only "-> computer".
        logger.info("[hud act] %s %s", action, {k: v for k, v in arguments.items() if k != "action"})
        try:
            if action == "screenshot":
                pass
            elif action in ("click", "double_click"):
                if arguments.get("x") is None or arguments.get("y") is None:
                    return tool_err("click needs x and y")
                await self.click(
                    self._to_screen(arguments["x"]),
                    self._to_screen(arguments["y"]),
                    button=arguments.get("button") or "left",
                    count=2 if action == "double_click" else 1,
                )
                self._pointer_used = True
            elif action in ("type", "press"):
                await self._ensure_focus()
                if action == "type":
                    await self.type_text(str(arguments.get("text") or ""))
                else:
                    keys = arguments.get("keys") or []
                    if isinstance(keys, str):
                        keys = [keys]
                    if not keys:
                        return tool_err("press needs keys, e.g. {'keys': ['Left', 'Down']}")
                    for element in keys:
                        chord = [to_keysym(k) for k in str(element).split("+") if k.strip()]
                        if chord:
                            await self.press_keys(chord)
            elif action == "scroll":
                await self.scroll(
                    self._to_screen(arguments["x"]) if arguments.get("x") is not None else None,
                    self._to_screen(arguments["y"]) if arguments.get("y") is not None else None,
                    scroll_y=int(arguments.get("scroll_y") or 0),
                )
            elif action == "wait":
                await self.wait(min(int(arguments.get("ms") or 500), 5000))
            else:
                return tool_err(f"unknown action {action!r}")
            await self.wait(300)  # let the UI settle before looking
            return await self._observation()
        except Exception as e:  # noqa: BLE001 - a bad action must not kill the episode
            return tool_err(f"{action} failed: {e}")


def computer_tool_class(shot_width: int) -> type[ChatComputerTool]:
    """The tool with a specific screenshot width baked in.

    The harness constructs tools with a fixed ``(spec, client, encoding)``
    signature, so per-run configuration has to travel on the class.
    """
    if shot_width == ChatComputerTool.shot_width:
        return ChatComputerTool
    return type("ChatComputerTool", (ChatComputerTool,), {"shot_width": shot_width})
