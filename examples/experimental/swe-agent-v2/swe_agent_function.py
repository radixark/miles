"""
Custom agent function for agentic_tool_call.generate.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    model_name = os.getenv(
        "AGENT_MODEL_NAME",
        os.getenv("SWE_AGENT_MODEL_NAME", "model"),
    )

    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    request = {
        **metadata,
        "base_url": session_url,
        "model": f"hosted_vllm/{model_name}",
        "sampling_params": request_kwargs,
    }

    try:
        response = await post(f"{agent_server_url}/run", request)
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    return {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }
