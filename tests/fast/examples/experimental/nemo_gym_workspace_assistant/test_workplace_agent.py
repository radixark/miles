import asyncio
import runpy
from pathlib import Path

import httpx
import pytest

_MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[5] / "examples/experimental/nemo-gym-workspace-assistant/workplace_agent.py")
)


def test_a_max_seq_len_refusal_is_the_context_budget_stop():
    """completion() turns the session's 409 for a prompt that fills --max-seq-len into ContextBudgetExceeded, as an engine 400."""
    transport = httpx.MockTransport(lambda request: httpx.Response(409, json={"error": "fills max_seq_len"}))

    async def call() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await _MODULE["completion"](client, "http://session/v1/chat/completions", {"max_tokens": 8}, 81920)

    with pytest.raises(_MODULE["ContextBudgetExceeded"]):
        asyncio.run(call())
