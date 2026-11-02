import base64
import io
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import mcp.types
import pytest
import torch
from examples.experimental.hud import rollout
from PIL import Image


@dataclass
class _ToolStep:
    result: Any


@pytest.fixture
def failed_hud_episode(monkeypatch: pytest.MonkeyPatch) -> Any:
    def prepare(outcome: str) -> SimpleNamespace:
        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        screenshot = mcp.types.ImageContent(
            type="image", data=base64.b64encode(image.getvalue()).decode(), mimeType="image/png"
        )
        turns = (
            []
            if outcome == "no_tokens"
            else [SimpleNamespace(prompt_token_ids=[1], output_token_ids=[2, 2], output_logprobs=[-0.1, -0.2])]
        )
        trace = SimpleNamespace(
            status="done",
            collect=lambda selector: turns,
            steps=[_ToolStep(result=SimpleNamespace(content=[screenshot]))] if outcome == "vision_mismatch" else [],
        )
        run = SimpleNamespace(reward=0.75, trace=trace)
        task = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(runs=[] if outcome == "no_run" else [run])))
        sdk = ModuleType("hud")
        sdk.Task = lambda **kwargs: task
        sdk_types = ModuleType("hud.agents.types")
        sdk_types.AgentStep = type("AgentStep", (), {})
        sdk_types.ToolStep = _ToolStep
        monkeypatch.setitem(sys.modules, "hud", sdk)
        monkeypatch.setitem(sys.modules, "hud.agents.types", sdk_types)
        monkeypatch.setattr(rollout, "_runtime", object())
        monkeypatch.setattr(rollout, "_gate", None)
        monkeypatch.setattr(rollout, "_build_agent", lambda args, params: object())
        tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: {"<|vision_start|>": 1, "<|image_pad|>": 2, "<|vision_end|>": 3}[token]
        )
        processor = SimpleNamespace(
            tokenizer=tokenizer,
            image_processor=lambda **kwargs: {
                "pixel_values": torch.ones(1),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
        )
        return SimpleNamespace(
            args=SimpleNamespace(partial_rollout=False, hud_max_sandboxes=1, hud_episode_timeout_s=1),
            tokenizer=tokenizer,
            processor=processor,
        )

    return prepare
