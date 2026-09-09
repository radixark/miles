"""Focused tests for the chess rollout adapter's load controls."""

import asyncio
from typing import Any

import pytest
from chess_eval.tito_v2 import build_runtime

import chess_agent
import run as chess_run


def test_stateful_recipe_metadata_reaches_harness_runtime() -> None:
    args = chess_run.ScriptArgs(
        hardware="H200",
        harness_mode="stateful",
        kl_loss_type="k3",
        max_llm_retries_per_move=0,
        repetition_reward_penalty=0.5,
        system_prompt_variant="random",
    )
    metadata = chess_run._prompt_rows(args)[0]["metadata"]
    runtime = build_runtime(
        base_url="http://session-server/sessions/test",
        metadata=metadata,
        request_kwargs=chess_agent._request_with_thinking({"max_tokens": 16384}),
    )
    assert runtime.eval_config.harness_mode == "stateful"
    assert runtime.eval_config.max_llm_retries_per_move == 0
    assert runtime.eval_config.max_response_tokens == 16384
    assert metadata["chess"]["repetition_reward_penalty"] == 0.5


def test_training_recipe_uses_native_qwen38_tito() -> None:
    agent_args = chess_run._agent_args(chess_run.ScriptArgs(hardware="H200"))

    assert "--tito-model qwen38small" in agent_args
    assert "--tito-model qwen35" not in agent_args


@pytest.mark.parametrize("penalty", [0.0, 0.5])
def test_stockfish_game_limiter_caps_complete_rollouts(
    monkeypatch: pytest.MonkeyPatch, penalty: float,
) -> None:
    active = 0
    maximum_active = 0

    async def fake_run_chess(**kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum_active
        del kwargs
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"ok": True}

    async def exercise() -> list[dict[str, Any]]:
        tasks = [
            chess_agent.run(
                base_url="http://session-server/sessions/test",
                prompt="play",
                metadata={"chess": {"stockfish_max_concurrent_games": 3, "repetition_reward_penalty": penalty}},
            )
            for _ in range(12)
        ]
        return await asyncio.gather(*tasks)

    monkeypatch.setattr(chess_agent, "run_chess", fake_run_chess)
    results = asyncio.run(exercise())

    assert results == [{"ok": True, "repetition_reward_penalty": penalty}] * 12
    assert maximum_active == 3


@pytest.mark.parametrize("value", (0, -1))
def test_stockfish_game_limiter_rejects_nonpositive_limits(value: int) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        chess_agent._stockfish_max_concurrent_games({"chess": {"stockfish_max_concurrent_games": value}})


@pytest.mark.parametrize("value", (True, 1.5, "16"))
def test_stockfish_game_limiter_rejects_noninteger_limits(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        chess_agent._stockfish_max_concurrent_games({"chess": {"stockfish_max_concurrent_games": value}})
