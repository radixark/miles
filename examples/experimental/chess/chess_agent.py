"""Qwen 3.5 TITO adapter for one chess rollout."""

import asyncio
import weakref
from collections.abc import Mapping
from typing import Any

from chess_eval.tito_v2 import run as run_chess

_REQUIRED_CHAT_TEMPLATE_KWARGS: dict[str, bool] = {
    "clear_thinking": False,
    "enable_thinking": True,
}
_DEFAULT_STOCKFISH_MAX_CONCURRENT_GAMES = 16
_GAME_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    tuple[int, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()


def _stockfish_max_concurrent_games(metadata: Mapping[str, Any] | None) -> int:
    chess_options = (metadata or {}).get("chess", {})
    if not isinstance(chess_options, Mapping):
        raise TypeError("metadata.chess must be a mapping")
    value = chess_options.get(
        "stockfish_max_concurrent_games",
        _DEFAULT_STOCKFISH_MAX_CONCURRENT_GAMES,
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("metadata.chess.stockfish_max_concurrent_games must be an integer")
    if value < 1:
        raise ValueError("metadata.chess.stockfish_max_concurrent_games must be at least 1")
    return value


def _game_limiter(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    existing = _GAME_LIMITERS.get(loop)
    if existing is None:
        limiter = asyncio.Semaphore(limit)
        _GAME_LIMITERS[loop] = (limit, limiter)
        return limiter
    configured_limit, limiter = existing
    if configured_limit != limit:
        raise ValueError("stockfish_max_concurrent_games must be consistent within one rollout loop")
    return limiter


def _request_with_thinking(request_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    request = dict(request_kwargs or {})
    existing = request.get("chat_template_kwargs", {})
    if not isinstance(existing, Mapping):
        raise TypeError("request_kwargs.chat_template_kwargs must be a mapping")

    merged = dict(existing)
    for key, required_value in _REQUIRED_CHAT_TEMPLATE_KWARGS.items():
        if key in merged and merged[key] != required_value:
            raise ValueError(f"chat_template_kwargs.{key} must be {required_value!r}")
        merged[key] = required_value
    request["chat_template_kwargs"] = merged
    return request


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one game while preserving Qwen thinking in the TITO conversation."""

    metadata_values = dict(metadata or {})
    limit = _stockfish_max_concurrent_games(metadata_values)
    async with _game_limiter(limit):
        return await run_chess(
            base_url=base_url,
            prompt=prompt,
            request_kwargs=_request_with_thinking(request_kwargs),
            metadata=metadata_values,
            **kwargs,
        )
