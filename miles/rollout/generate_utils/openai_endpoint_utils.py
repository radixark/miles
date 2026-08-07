"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

from miles.rollout.session.samples.codec import SamplesReply, decode_samples_and_merge_input_sample
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
        session_addrs = getattr(args, "session_server_addrs", None)
        if not session_addrs:
            raise RuntimeError(
                "session_server_addrs is not set. Pass --use-session-server to start the session server."
            )
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL.
        session_addr = random.choice(session_addrs)
        session_url = f"http://{session_addr}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        session_server_instance_id = instance_ids.get(session_addr)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
        )

    async def collect_samples(self, input_sample: Sample, *, max_seq_len: int | None) -> SamplesReply:
        """Fetch server-assembled training samples for this session."""
        try:
            # `asyncio.TimeoutError` propagates after cleanup is attempted for `agentic_tool_call.generate` to handle.
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

        return decode_samples_and_merge_input_sample(payload, input_sample)
