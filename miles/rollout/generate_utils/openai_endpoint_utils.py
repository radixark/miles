"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace
from dataclasses import dataclass

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.rollout.session.types import SESSION_GENERATION_HEADER, SESSION_RECORD_ERROR_CODE
from miles.utils.http_utils import post, request_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120
_CLEANUP_TIMEOUT = 30
_MAX_CLEANUPS = 32
_cleanup_tasks: set[asyncio.Task] = set()


class SessionCollectError(RuntimeError):
    """The owning server could not read this session's records."""


@dataclass(frozen=True)
class CollectedSamples:
    reply: SamplesReply
    generation: int | None


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
    ):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

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
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
            samples_wire_fields=COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS,
        )

    async def collect_samples(
        self, input_sample: Sample, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> CollectedSamples:
        """Decode a snapshot. The caller schedules cleanup after assembling its output."""
        body: dict = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        response = await request_no_retry(
            f"{self.base_url}/samples", body, method="POST", timeout=_SESSION_REQUEST_TIMEOUT
        )
        if response.status_code == 503:
            try:
                error = response.json()
            except ValueError:
                error = None
            if (
                isinstance(error, dict)
                and isinstance(error.get("error"), dict)
                and error["error"].get("code") == SESSION_RECORD_ERROR_CODE
            ):
                raise SessionCollectError(response.text)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"POST {self.base_url}/samples failed with {response.status_code}: {response.text}")
        reply = decode_samples_and_merge_input_sample(response.content, input_sample, fields=self.samples_wire_fields)
        try:
            generation = int(response.headers[SESSION_GENERATION_HEADER])
            if generation <= 0:
                generation = None
        except (KeyError, ValueError):
            generation = None
        return CollectedSamples(reply, generation)

    def schedule_cleanup(self, generation: int | None) -> None:
        if type(generation) is not int or generation <= 0:
            logger.warning("Session %s has no cleanup generation; retaining until idle expiry", self.session_id)
            return
        if len(_cleanup_tasks) >= _MAX_CLEANUPS:
            logger.warning("Session cleanup is busy; retaining %s until idle expiry", self.session_id)
            return
        coroutine = self._cleanup(generation)
        try:
            task = asyncio.create_task(coroutine)
        except RuntimeError:
            coroutine.close()
            logger.warning("Could not schedule cleanup for session %s", self.session_id, exc_info=True)
            return
        _cleanup_tasks.add(task)
        task.add_done_callback(_cleanup_tasks.discard)

    async def _cleanup(self, generation: int) -> None:
        try:
            response = await request_no_retry(
                self.base_url,
                {},
                method="DELETE",
                timeout=_CLEANUP_TIMEOUT,
                headers={SESSION_GENERATION_HEADER: str(generation)},
            )
            if response.status_code not in (204, 404, 412):
                logger.warning("Cleanup for session %s returned %s", self.session_id, response.status_code)
        except Exception:
            logger.warning("Failed to clean up session %s", self.session_id, exc_info=True)
