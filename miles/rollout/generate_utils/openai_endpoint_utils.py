"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str, session_server_instance_id: str | None = None):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace):
        session_ip = getattr(args, "session_server_ip", None)
        session_ports = getattr(args, "session_server_ports", None)
        if not session_ip or not session_ports:
            raise RuntimeError(
                "session_server_ip/session_server_ports are not set. "
                "Pass --use-session-server to start the session server."
            )
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL.
        session_port = random.choice(session_ports)
        session_url = f"http://{session_ip}:{session_port}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        session_server_instance_id = instance_ids.get(session_port)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
        )

    async def collect_records(self) -> tuple[list[SessionRecord], dict]:
        try:
            response = await asyncio.wait_for(
                post(self.base_url, {}, action="get"),
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timed out waiting for session {self.session_id} records after {_SESSION_REQUEST_TIMEOUT}s "
                f"(likely stale HTTP keepalive connection). Returning empty records."
            )
            # Still attempt to clean up the session.
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception:
                logger.warning(f"Failed to delete session {self.session_id} after timeout")
            return [], {}
        except Exception as e:
            logger.warning(f"Failed to get session {self.session_id} records: {e}")
            raise
        response = GetSessionResponse.model_validate(response)
        records = response.records
        metadata = response.metadata

        try:
            await asyncio.wait_for(
                post(self.base_url, {}, action="delete"),
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Failed to delete session {self.session_id} after collecting records: {e}")

        return (records or []), metadata
