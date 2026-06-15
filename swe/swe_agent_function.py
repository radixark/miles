"""
Custom agent function for agentic_tool_call.generate.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunparse

from swe.agent_run_client import AgentRunAborted, AgentRunRetryPolicy, post_agent_run

logger = logging.getLogger(__name__)

_AGENT_RUN_CLIENT_KWARGS = {
    "agent_run_max_attempts",
    "agent_run_retry_backoff_sec",
}


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    # variant: str = "",
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
    request: dict[str, Any] = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
        **{k: v for k, v in (kwargs or {}).items() if k not in _AGENT_RUN_CLIENT_KWARGS},
        # "environment_build_timeout_sec":
        # "trials_dir":
    }

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None:
        if external_host:
            port = urlsplit(f"http://{session_server_id}").port
            session_server_id = f"{external_host}:{port}"
        request["session_server_id"] = session_server_id

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    request_id = kwargs.get("session_id")
    if not request_id:
        logger.error("Missing session_id for Harbor request id")
        return None

    timeout_sec = kwargs.get("agent_timeout_sec", 3600)
    retry_policy = AgentRunRetryPolicy(
        max_attempts=int(kwargs.get("agent_run_max_attempts", 2)),
        backoff_sec=float(kwargs.get("agent_run_retry_backoff_sec", 0.5)),
    )
    try:
        response = await asyncio.wait_for(
            post_agent_run(
                f"{agent_server_url}/run",
                request,
                request_id=request_id,
                timeout_sec=timeout_sec,
                retry_policy=retry_policy,
            ),
            timeout=timeout_sec,
        )
    except AgentRunAborted:
        logger.info("Agent server call aborted request_id=%s", request_id)
        return None
    except asyncio.TimeoutError:
        logger.error(f"Agent server call timed out after {timeout_sec}s")
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled request_id=%s", request_id)
        raise
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None
    logger.debug(f"Agent server call response: {response}, from harbor server!")
    return {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }
