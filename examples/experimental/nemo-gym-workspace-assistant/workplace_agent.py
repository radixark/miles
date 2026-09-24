"""Workplace tool loop: all policy calls pass through the Miles TITO session."""

import json
import logging
import os
import re
import time
from copy import deepcopy
from typing import Any

import httpx

from miles.rollout.agentic.session import resolve_session_url

logger = logging.getLogger(__name__)


def chat_tools(params: dict) -> list[dict]:
    return [
        {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}} for tool in params["tools"]
    ]


def assistant_message(message: dict) -> dict:
    result = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        result["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    return result


async def post_json(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    response = await client.post(url, json=body)
    if response.is_error:
        logger.error("Workplace HTTP %s: %s", response.status_code, response.text[:2000])
    response.raise_for_status()
    return response.json()


class ContextBudgetExceeded(ValueError):
    """The accumulated transcript leaves no space for another policy token."""


async def completion(client: httpx.AsyncClient, url: str, body: dict, context: int) -> dict:
    try:
        return await post_json(client, url, body)
    except httpx.HTTPStatusError as error:
        text = error.response.text
        if error.response.status_code != 400:
            raise
        match = re.search(r"(\d+) tokens from the input messages", text)
        if match is None:
            if "input" in text and "longer than" in text and "context length" in text:
                raise ContextBudgetExceeded(text) from error
            raise
        available = min(body["max_tokens"], context - int(match.group(1)) - 8)
        if available <= 0:
            raise ContextBudgetExceeded(text) from error
        if available >= body["max_tokens"]:
            raise
        # Rejected requests contain no generated tokens. The session server
        # leaves non-200 responses unrecorded, so this transcript can be retried.
        return await post_json(client, url, {**body, "max_tokens": available})


async def policy_loop(
    client: httpx.AsyncClient,
    base_url: str,
    resource_url: str,
    params: dict,
    request_kwargs: dict,
    context: int,
    max_turns: int,
) -> dict:
    messages = deepcopy(params["input"])
    tools = chat_tools(params)
    budget = request_kwargs["max_tokens"]
    generated = 0
    tool_time = 0.0
    stop = "turn_limit"
    turns = 0
    for _ in range(max_turns):
        # The inference server owns tokenization. If the remaining response
        # budget exceeds the actual available context, completion() retries
        # with the exact input count from the server's rejected request.
        available = budget - generated
        if available <= 0:
            stop = "response_budget"
            break
        body = {
            **request_kwargs,
            "model": "nemotron35-lightning",
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "max_tokens": available,
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
        }
        try:
            response = await completion(client, resolve_session_url(base_url) + "/chat/completions", body, context)
        except ContextBudgetExceeded:
            stop = "context_limit"
            break
        choice = response["choices"][0]
        if choice["finish_reason"] == "abort":
            return {"workplace_episode_valid": False, "workplace_stop": "abort"}
        usage = response["usage"]
        generated += usage["completion_tokens"]
        turns += 1
        message = assistant_message(choice["message"])
        messages.append(message)
        for call in message.get("tool_calls", []):
            started = time.monotonic()
            output = await post_json(client, resource_url + "/tool", call["function"])
            tool_time += time.monotonic() - started
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output, sort_keys=True, ensure_ascii=False),
                }
            )
        if choice["finish_reason"] == "length":
            stop = "truncated"
            break
        if not message.get("tool_calls"):
            stop = "final_answer"
            break
    started = time.monotonic()
    verification = await post_json(client, resource_url + "/verify", {})
    tool_time += time.monotonic() - started
    if not verification.get("valid") or not verification.get("state_replay_consistent"):
        raise RuntimeError("Native Workplace verification failed")
    return {
        "workplace_episode_valid": True,
        "workplace_reward": verification["reward"],
        "workplace_stop": stop,
        "workplace_turns": turns,
        "workplace_completion_tokens": generated,
        "workplace_tool_calls": verification["tool_calls"],
        "workplace_state_replay_consistent": True,
        "agent_metrics": {"total_tool_time": tool_time},
    }


async def run(base_url: str, prompt: Any, request_kwargs: dict, metadata: dict, **kwargs: Any) -> dict:
    # Neither ground truth nor synthesis provenance is part of this payload.
    params = metadata["workplace_policy"]
    if prompt != params["input"]:
        raise ValueError("Miles prompt and Workplace policy input disagree")
    resource = os.environ["WORKPLACE_RESOURCE_URL"].rstrip("/")
    timeout = httpx.Timeout(3600, connect=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await post_json(client, f"{resource}/sessions/{metadata['workplace_task_id']}", {})
        session_url = f"{resource}/sessions/{response['session_id']}"
        try:
            return await policy_loop(
                client,
                base_url,
                session_url,
                params,
                request_kwargs,
                metadata["max_seq_len"],
                metadata.get("workplace_max_turns", 24),
            )
        finally:
            try:
                cleanup = await client.delete(session_url)
                cleanup.raise_for_status()
            except (httpx.HTTPError, TimeoutError):
                # The server also expires abandoned sessions after three hours.
                logger.exception("Could not release Workplace episode %s", response["session_id"])
