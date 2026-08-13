"""The agent Miles drives HUD environments with: HUD's own OpenAI-compatible
harness, plus the one thing it lacks for computer use -- a screen
(:mod:`.computer_tool` explains that gap). The loop, tool plumbing, trace
recording and token-level ``Sample`` collection are all inherited unchanged.
"""

from __future__ import annotations

from examples.experimental.hud.computer_tool import ChatComputerTool, computer_tool_class
from hud.agents.openai_compatible.agent import OpenAIChatAgent


SYSTEM_PROMPT = """You are operating a computer through screenshots.

You have at most {max_steps} turns, and a turn is the expensive thing: each one \
re-reads the whole screen history. Actions are not -- a single computer call \
can carry several keystrokes, played in order. So plan a few steps ahead and \
send them together rather than one per turn.

Keys go to whatever the screen has focused, so click what you mean to type \
into before typing. Look at each screenshot to check that your last action did \
what you expected, and change approach if it did not."""


class ComputerChatAgent(OpenAIChatAgent):
    """OpenAIChatAgent whose only tool is the screen.

    Beyond the tool, the one thing this adds is a system prompt, because the
    task template's prompt says *what* to do and something has to say *how* to
    act here: chiefly that turns are scarce and keystrokes are not. Without it
    a model spends one turn per keystroke and runs out of episode long before
    it runs out of task.
    """

    tool_catalog = (ChatComputerTool,)

    @classmethod
    def for_shot_width(cls, shot_width: int) -> type[ComputerChatAgent]:
        """The agent with a specific screenshot width baked into its tool."""
        tool = computer_tool_class(shot_width)
        if tool is ChatComputerTool:
            return cls
        return type(cls.__name__, (cls,), {"tool_catalog": (tool,)})
