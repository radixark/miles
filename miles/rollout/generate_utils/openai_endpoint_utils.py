"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

from miles.rollout.session.samples.codec import SamplesReply, decode_samples_reply
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

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

    async def collect_samples(self, input_sample: Sample, *, max_seq_len: int | None) -> SamplesReply:
        """Fetch the server-assembled training samples for this session.

        Single direct POST, no retries: a 5xx means the owning instance died and
        the session's records died with it, and a 422 is a deterministic
        assembly failure whose assertion text is
        the body — both must raise loudly, immediately. A timeout raises too
        (assembly is seconds server-side; the old records path silently ABORTed
        the sample on timeout and lost data). The session DELETE is attempted
        on every path, success or failure, matching the old cleanup semantics;
        a DELETE failure is only a warning.
        """
        try:
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                {"max_seq_len": max_seq_len},
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Failed to delete session {self.session_id} after collecting samples: {e}")

        return decode_samples_reply(payload, input_sample)
