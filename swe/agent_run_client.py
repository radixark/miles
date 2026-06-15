import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AgentRunAborted(RuntimeError):
    """Raised when Harbor explicitly reports that a run was aborted."""


@dataclass(frozen=True)
class AgentRunRetryPolicy:
    max_attempts: int = 2
    backoff_sec: float = 0.5


RETRYABLE_RUN_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def is_aborted_response(response: httpx.Response) -> bool:
    if response.status_code != 409:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    detail = body.get("detail") if isinstance(body, dict) else None
    return isinstance(detail, dict) and detail.get("reason") == "aborted"


async def post_agent_run(
    url: str,
    request: dict[str, Any],
    *,
    request_id: str,
    timeout_sec: float,
    retry_policy: AgentRunRetryPolicy | None = None,
) -> dict[str, Any]:
    retry_policy = retry_policy or AgentRunRetryPolicy()
    headers = {"x-harbor-request-id": request_id}
    timeout = httpx.Timeout(connect=30.0, read=timeout_sec, write=30.0, pool=30.0)
    max_attempts = max(1, retry_policy.max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=request, headers=headers)
            if is_aborted_response(response):
                logger.info("Agent server run aborted request_id=%s response=%s", request_id, response.text)
                raise AgentRunAborted(request_id)
            response.raise_for_status()
            return response.json()
        except RETRYABLE_RUN_EXCEPTIONS as exc:
            if attempt >= max_attempts:
                raise
            delay = retry_policy.backoff_sec * attempt
            logger.warning(
                "Retrying agent /run after transient submit failure "
                "request_id=%s attempt=%d/%d delay=%.2fs error=%r",
                request_id,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")
