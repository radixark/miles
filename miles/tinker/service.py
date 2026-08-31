"""TinkerService: sampling sessions, futures, admission, RID ownership, and a process-local HTTP client."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any

import httpx

from miles.tinker import codec, sampling

logger = logging.getLogger(__name__)

RETRIEVE_POLL_WINDOW_S = 12.0


def _fingerprint(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class TinkerService:
    """Owns every sampling-plane concern; the SGLang fleet is reached by plain HTTP, never Ray."""

    def __init__(self, args, abort_requests=None):
        self.args = args
        self.sessions: dict[str, float] = {}
        self.samplers: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.bindings: dict[str, dict[str, Any]] = {}
        self._abort_requests = abort_requests
        self._operation_cap = getattr(args, "tinker_sample_inflight_cap", 64)
        self._child_semaphore = asyncio.Semaphore(getattr(args, "tinker_sample_transport_cap", 64))
        self._client: httpx.AsyncClient | None = None

    # ------------------------- transport -------------------------

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        return self._client

    async def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        """One-shot POST on the service-owned client: a failed sample terminals its own Future only."""
        response = await self._http_client().post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        for record in self.records.values():
            task = record.get("task")
            if task is not None and not task.done():
                task.cancel()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------- control plane -------------------------

    def register_binding(self, key: str, info: dict[str, Any]) -> None:
        """Injected static binding table (decision #15); Y4 swaps in committed ServingRef resolution."""
        self.bindings[key] = dict(info)

    def create_session(self) -> str:
        session_id = f"sess-{uuid.uuid4().hex[:16]}"
        self.sessions[session_id] = time.time()
        return session_id

    def create_sampling_session(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if body.get("session_id") not in self.sessions:
            return 404, {"detail": "unknown session"}
        base_model = body.get("base_model")
        model_path = body.get("model_path")
        binding = None
        if model_path:
            binding = self.bindings.get(model_path)
            if binding is None:
                return 404, {"detail": f"unknown model_path {model_path!r}"}
            base_model = binding.get("base_model") or base_model
        sampler_id = f"samp-{uuid.uuid4().hex[:8]}-ss{body.get('sampling_session_seq_id', 0)}"
        self.samplers[sampler_id] = {"base_model": base_model, "model_path": model_path, "binding": binding}
        return 200, {"type": "create_sampling_session", "sampling_session_id": sampler_id}

    def get_sampler(self, sampler_id: str) -> tuple[int, dict[str, Any]]:
        sampler = self.samplers.get(sampler_id)
        if sampler is None:
            return 404, {"detail": "unknown sampler"}
        return 200, {
            "sampler_id": sampler_id,
            "base_model": sampler["base_model"],
            "model_path": sampler["model_path"],
        }

    # ------------------------- sampling plane -------------------------

    def submit_sample(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sampler = self.samplers.get(body.get("sampling_session_id"))
        if sampler is None:
            return 404, {"detail": "unknown sampling session"}
        if body.get("topk_prompt_logprobs"):
            return 422, {"detail": "topk_prompt_logprobs is not supported"}
        seq_id = body.get("seq_id")
        if not isinstance(seq_id, int) or seq_id < 0:
            return 400, {"detail": "seq_id must be >= 0"}
        request_id = f"{body['sampling_session_id']}:s{seq_id}"
        fingerprint = _fingerprint(body)
        known = self.records.get(request_id)
        if known is not None:
            if known["fingerprint"] != fingerprint:
                return 422, {"detail": "retry must resend the identical request"}
            return 200, {"request_id": request_id, "sample_sequence_ids": known["sequence_ids"]}
        in_flight = sum(1 for record in self.records.values() if not record["event"].is_set())
        if in_flight >= self._operation_cap:
            return 429, {"detail": f"{in_flight} samples in flight (cap {self._operation_cap})"}
        num_samples = int(body.get("num_samples") or 1)
        sequence_ids = [f"{request_id}::seq{index}" for index in range(num_samples)]
        binding = sampler["binding"]
        record = {
            "fingerprint": fingerprint,
            "sequence_ids": sequence_ids,
            "event": asyncio.Event(),
            "proto": None,
            "error": None,
            "root_rid": request_id if binding is None else f"{binding['name']}::{request_id}",
        }
        self.records[request_id] = record
        record["task"] = asyncio.create_task(self._run_sample(request_id, binding, body))
        return 200, {"request_id": request_id, "sample_sequence_ids": sequence_ids}

    async def _run_sample(self, request_id: str, binding: dict[str, Any] | None, body: dict[str, Any]) -> None:
        record = self.records[request_id]
        try:
            sequences, prompt_logprobs = await sampling.run_sample_operation(
                self.args,
                request_id=request_id,
                binding=binding,
                payload=body,
                post_json=self.post_json,
                child_semaphore=self._child_semaphore,
            )
            record["proto"] = codec.sample_response_proto_bytes(sequences, prompt_logprobs)
        except asyncio.CancelledError:
            record["error"] = "sample aborted"
        except Exception as exc:
            logger.warning(f"sample {request_id} failed: {exc!r}")
            record["error"] = str(exc)
        finally:
            record["event"].set()

    async def abort_sample(self, request_id: str) -> bool:
        """Request-scoped abort: cancel the operation task and best-effort abort its child rids."""
        record = self.records.get(request_id)
        if record is None:
            return False
        task = record.get("task")
        if task is not None and not task.done():
            task.cancel()
        if self._abort_requests is not None:
            await self._abort_requests(record["root_rid"], prefix=True)
        return True

    async def retrieve(self, request_id: str) -> tuple[str, Any]:
        """Long-poll one future: ('proto', bytes) | ('json', body) | ('status', (code, body))."""
        record = self.records.get(request_id)
        if record is None:
            return "status", (410, {"detail": "unknown request_id"})
        try:
            await asyncio.wait_for(record["event"].wait(), timeout=RETRIEVE_POLL_WINDOW_S)
        except asyncio.TimeoutError:
            return "json", {"type": "try_again", "request_id": request_id, "queue_state": "active"}
        if record["error"] is not None:
            return "json", {"error": record["error"], "category": "server"}
        return "proto", record["proto"]
