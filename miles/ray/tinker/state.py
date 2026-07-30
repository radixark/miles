"""In-memory state machine for the Tinker-compatible control plane."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from miles.ray.tinker.protocol import TinkerError

_CHECKPOINT_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class TinkerModelConfig:
    """Adapter configuration consumed by the existing multi-LoRA lifecycle."""

    rank: int
    alpha: int
    save: Path
    seed: int | None
    train_unembed: bool
    train_mlp: bool
    train_attn: bool
    user_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_global_batch_size: int = 1
    num_step: int | None = None


@dataclass
class SessionRecord:
    session_id: str
    tags: list[str]
    user_metadata: dict[str, Any] | None
    sdk_version: str
    project_id: str | None
    created_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = field(default_factory=time.time)


@dataclass
class ModelRecord:
    model_id: str
    session_id: str
    model_seq_id: int
    base_model: str
    config: TinkerModelConfig
    status: Literal["loading", "active", "unloading", "unloaded", "failed"] = "loading"
    next_seq_id: int = 1
    checkpoint_counter: int = 0
    created_at: float = field(default_factory=time.time)
    last_request_at: float = field(default_factory=time.time)


@dataclass
class SamplingSessionRecord:
    sampling_session_id: str
    session_id: str
    sampling_session_seq_id: int
    base_model: str
    model_path: str | None
    adapter_path: Path | None
    adapter_name: str | None
    next_seq_id: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class FutureRecord:
    request_id: str
    model_id: str | None
    status: Literal["pending", "completed", "failed"] = "pending"
    response: dict[str, Any] | None = None
    error: str | None = None
    category: Literal["unknown", "server", "user"] = "unknown"
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass(frozen=True)
class Operation:
    request_id: str
    kind: str
    model_id: str | None
    payload: dict[str, Any]

    def asdict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "model_id": self.model_id,
            "payload": self.payload,
        }


@dataclass
class CheckpointRecord:
    model_id: str
    checkpoint_id: str
    checkpoint_type: Literal["training", "sampler"]
    tinker_path: str
    local_path: Path
    seq_id: int
    checkpoint_step: int
    include_optimizer: bool
    status: Literal["pending", "ready", "failed"] = "pending"
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int | None = None


def request_fingerprint(kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


class TinkerState:
    """Controller-owned protocol state.

    Ray serializes calls to the controller actor, so these transitions need no
    process-level lock. FastAPI runs on the same actor event loop.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self.models: dict[str, ModelRecord] = {}
        self.sampling_sessions: dict[str, SamplingSessionRecord] = {}
        self.futures: dict[str, FutureRecord] = {}
        self.checkpoints: dict[str, CheckpointRecord] = {}
        self.model_create_keys: dict[tuple[str, int], tuple[str, str, str]] = {}
        self.sampling_create_keys: dict[tuple[str, int], tuple[str, str]] = {}
        self.model_request_keys: dict[tuple[str, int], tuple[str, str]] = {}
        self.sample_request_keys: dict[tuple[str, int], tuple[str, str]] = {}

    def create_session(
        self,
        *,
        tags: list[str],
        user_metadata: dict[str, Any] | None,
        sdk_version: str,
        project_id: str | None,
    ) -> SessionRecord:
        session_id = f"session-{uuid.uuid4().hex}"
        record = SessionRecord(
            session_id=session_id,
            tags=list(tags),
            user_metadata=user_metadata,
            sdk_version=sdk_version,
            project_id=project_id,
        )
        self.sessions[session_id] = record
        return record

    def heartbeat(self, session_id: str) -> None:
        session = self.require_session(session_id)
        session.last_heartbeat_at = time.time()

    def require_session(self, session_id: str) -> SessionRecord:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise TinkerError(f"session {session_id!r} does not exist", category="user") from None

    def begin_model_create(
        self,
        *,
        session_id: str,
        model_seq_id: int,
        base_model: str,
        config: TinkerModelConfig,
        payload: dict[str, Any],
    ) -> tuple[ModelRecord, FutureRecord, bool]:
        self.require_session(session_id)
        if model_seq_id < 0:
            raise TinkerError("model_seq_id must be non-negative", category="user")
        key = (session_id, model_seq_id)
        fingerprint = request_fingerprint("create_model", payload)
        if key in self.model_create_keys:
            old_fingerprint, model_id, request_id = self.model_create_keys[key]
            if old_fingerprint != fingerprint:
                raise TinkerError(
                    f"model_seq_id {model_seq_id} was already used with a different request",
                    category="user",
                )
            return self.models[model_id], self.futures[request_id], False

        model_id = f"model-{uuid.uuid4().hex}"
        model = ModelRecord(
            model_id=model_id,
            session_id=session_id,
            model_seq_id=model_seq_id,
            base_model=base_model,
            config=config,
        )
        future = self.new_future(model_id=model_id)
        self.models[model_id] = model
        self.model_create_keys[key] = (fingerprint, model_id, future.request_id)
        return model, future, True

    def require_model(self, model_id: str, *, active: bool = True) -> ModelRecord:
        try:
            model = self.models[model_id]
        except KeyError:
            raise TinkerError(f"model {model_id!r} does not exist", category="user") from None
        if active and model.status != "active":
            raise TinkerError(f"model {model_id!r} is {model.status}, not active", category="user")
        return model

    def rollback_model_create(self, model_id: str, request_id: str) -> None:
        """Remove a model-create reservation whose adapter registration failed."""
        model = self.models.get(model_id)
        future = self.futures.get(request_id)
        if model is None or future is None or model.status != "loading" or future.status != "pending":
            return
        key = (model.session_id, model.model_seq_id)
        reservation = self.model_create_keys.get(key)
        if reservation is not None and reservation[1:] == (model_id, request_id):
            self.model_create_keys.pop(key, None)
        self.models.pop(model_id, None)
        self.futures.pop(request_id, None)

    def submit_model_operation(
        self,
        *,
        model_id: str,
        seq_id: int | None,
        kind: str,
        payload: dict[str, Any],
        idempotency_payload: dict[str, Any] | None = None,
    ) -> tuple[FutureRecord, Operation | None]:
        actual_seq_id, duplicate = self.validate_model_operation(
            model_id=model_id,
            seq_id=seq_id,
            kind=kind,
            payload=idempotency_payload if idempotency_payload is not None else payload,
        )
        if duplicate is not None:
            return duplicate, None

        model = self.require_model(model_id)
        fingerprint_payload = idempotency_payload if idempotency_payload is not None else payload
        fingerprint = request_fingerprint(kind, fingerprint_payload)
        key = (model_id, actual_seq_id)
        model.next_seq_id += 1
        model.last_request_at = time.time()
        future = self.new_future(model_id=model_id)
        self.model_request_keys[key] = (fingerprint, future.request_id)
        operation = Operation(
            request_id=future.request_id,
            kind=kind,
            model_id=model_id,
            payload={**payload, "seq_id": actual_seq_id},
        )
        return future, operation

    def validate_model_operation(
        self,
        *,
        model_id: str,
        seq_id: int | None,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[int, FutureRecord | None]:
        """Validate sequence/idempotency without mutating state.

        Checkpoint operations use this before allocating a path so an invalid
        or duplicate sequence cannot leave a phantom checkpoint behind.
        """
        model = self.require_model(model_id)
        actual_seq_id = model.next_seq_id if seq_id is None else seq_id
        if actual_seq_id < 1:
            raise TinkerError("seq_id must be positive", category="user")
        fingerprint = request_fingerprint(kind, payload)
        key = (model_id, actual_seq_id)
        if key in self.model_request_keys:
            old_fingerprint, request_id = self.model_request_keys[key]
            if old_fingerprint != fingerprint:
                raise TinkerError(
                    f"seq_id {actual_seq_id} was already used with a different request",
                    category="user",
                )
            return actual_seq_id, self.futures[request_id]
        if actual_seq_id != model.next_seq_id:
            raise TinkerError(
                f"expected seq_id {model.next_seq_id} for model {model_id!r}, got {actual_seq_id}",
                category="user",
            )
        return actual_seq_id, None

    def begin_unload(self, model_id: str) -> tuple[FutureRecord, Operation | None]:
        model = self.require_model(model_id, active=False)
        existing = next(
            (future for future in self.futures.values() if future.model_id == model_id and future.response is not None and future.response.get("type") == "unload_model"),
            None,
        )
        if model.status == "unloaded" and existing is not None:
            return existing, None
        if model.status == "unloading":
            for future in reversed(list(self.futures.values())):
                if future.model_id == model_id and future.status == "pending":
                    return future, None
        if model.status != "active":
            raise TinkerError(f"model {model_id!r} cannot be unloaded while {model.status}", category="user")
        model.status = "unloading"
        future = self.new_future(model_id=model_id)
        return future, Operation(future.request_id, "unload_model", model_id, {})

    def create_sampling_session(
        self,
        *,
        session_id: str,
        sampling_session_seq_id: int,
        base_model: str,
        model_path: str | None,
        adapter_path: Path | None,
        adapter_name: str | None,
        payload: dict[str, Any],
    ) -> tuple[SamplingSessionRecord, bool]:
        self.require_session(session_id)
        if sampling_session_seq_id < 0:
            raise TinkerError("sampling_session_seq_id must be non-negative", category="user")
        key = (session_id, sampling_session_seq_id)
        fingerprint = request_fingerprint("create_sampling_session", payload)
        if key in self.sampling_create_keys:
            old_fingerprint, sampling_session_id = self.sampling_create_keys[key]
            if old_fingerprint != fingerprint:
                raise TinkerError(
                    f"sampling_session_seq_id {sampling_session_seq_id} was already used with a different request",
                    category="user",
                )
            return self.sampling_sessions[sampling_session_id], False

        sampling_session_id = f"sampling-session-{uuid.uuid4().hex}"
        record = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id=session_id,
            sampling_session_seq_id=sampling_session_seq_id,
            base_model=base_model,
            model_path=model_path,
            adapter_path=adapter_path,
            adapter_name=adapter_name,
        )
        self.sampling_sessions[sampling_session_id] = record
        self.sampling_create_keys[key] = (fingerprint, sampling_session_id)
        return record, True

    def submit_sample(
        self,
        *,
        sampling_session_id: str,
        seq_id: int | None,
        payload: dict[str, Any],
    ) -> tuple[FutureRecord, Operation | None]:
        try:
            session = self.sampling_sessions[sampling_session_id]
        except KeyError:
            raise TinkerError(
                f"sampling session {sampling_session_id!r} does not exist",
                category="user",
            ) from None
        actual_seq_id = session.next_seq_id if seq_id is None else seq_id
        if actual_seq_id < 0:
            raise TinkerError("sampling seq_id must be non-negative", category="user")
        fingerprint = request_fingerprint("sample", payload)
        key = (sampling_session_id, actual_seq_id)
        if key in self.sample_request_keys:
            old_fingerprint, request_id = self.sample_request_keys[key]
            if old_fingerprint != fingerprint:
                raise TinkerError(
                    f"sampling seq_id {actual_seq_id} was already used with a different request",
                    category="user",
                )
            return self.futures[request_id], None
        if actual_seq_id != session.next_seq_id:
            raise TinkerError(
                f"expected seq_id {session.next_seq_id} for sampling session {sampling_session_id!r}, got {actual_seq_id}",
                category="user",
            )

        session.next_seq_id += 1
        future = self.new_future(model_id=None)
        self.sample_request_keys[key] = (fingerprint, future.request_id)
        operation = Operation(
            request_id=future.request_id,
            kind="sample",
            model_id=None,
            payload={
                **payload,
                "seq_id": actual_seq_id,
                "sampling_session_id": sampling_session_id,
                "base_model": session.base_model,
                "adapter_path": str(session.adapter_path) if session.adapter_path is not None else None,
                "adapter_name": session.adapter_name,
            },
        )
        return future, operation

    def new_future(self, model_id: str | None) -> FutureRecord:
        request_id = f"request-{uuid.uuid4().hex}"
        future = FutureRecord(request_id=request_id, model_id=model_id)
        self.futures[request_id] = future
        return future

    def complete_future(self, request_id: str, response: dict[str, Any]) -> None:
        future = self.require_future(request_id)
        if future.status != "pending":
            return
        future.status = "completed"
        future.response = response
        future.completed_at = time.time()

    def fail_future(
        self,
        request_id: str,
        error: str,
        category: Literal["unknown", "server", "user"] = "server",
    ) -> None:
        future = self.require_future(request_id)
        if future.status != "pending":
            return
        future.status = "failed"
        future.error = error
        future.category = category
        future.completed_at = time.time()
        if future.model_id is not None:
            model = self.models.get(future.model_id)
            if model is not None and model.status == "loading":
                model.status = "failed"

    def require_future(self, request_id: str) -> FutureRecord:
        try:
            return self.futures[request_id]
        except KeyError:
            raise TinkerError(f"request {request_id!r} does not exist", category="user") from None

    def retrieve_future(self, request_id: str) -> dict[str, Any]:
        future = self.require_future(request_id)
        if future.status == "pending":
            return {"type": "try_again", "request_id": request_id, "queue_state": "active"}
        if future.status == "failed":
            return {"error": future.error or "request failed", "category": future.category}
        assert future.response is not None
        return future.response

    def allocate_checkpoint(
        self,
        *,
        model_id: str,
        seq_id: int,
        requested_name: str | None,
        checkpoint_type: Literal["training", "sampler"],
        ttl_seconds: int | None,
        overwrite: bool,
    ) -> CheckpointRecord:
        model = self.require_model(model_id)
        if seq_id < 1:
            raise TinkerError("checkpoint seq_id must be positive", category="user")
        if ttl_seconds is not None and ttl_seconds < 0:
            raise TinkerError("checkpoint ttl_seconds must be non-negative", category="user")
        name = requested_name or f"checkpoint-{seq_id:06d}"
        if not _CHECKPOINT_NAME.fullmatch(name) or name in (".", ".."):
            raise TinkerError(
                "checkpoint path must be a simple name containing only letters, digits, '.', '_' or '-'",
                category="user",
            )
        namespace = "weights" if checkpoint_type == "training" else "sampler_weights"
        tinker_path = f"tinker://{model_id}/{namespace}/{name}"
        existing = self.checkpoints.get(tinker_path)
        if existing is not None:
            if existing.seq_id == seq_id and existing.checkpoint_type == checkpoint_type:
                return existing
            if not overwrite:
                raise TinkerError(f"checkpoint {tinker_path!r} already exists", category="user")

        model.checkpoint_counter += 1
        checkpoint_step = model.checkpoint_counter
        expires_at = None if ttl_seconds is None else time.time() + ttl_seconds
        local_path = model.config.save / "checkpoints" / f"step_{checkpoint_step}"
        record = CheckpointRecord(
            model_id=model_id,
            checkpoint_id=f"{namespace}/{name}",
            checkpoint_type=checkpoint_type,
            tinker_path=tinker_path,
            local_path=local_path,
            seq_id=seq_id,
            checkpoint_step=checkpoint_step,
            include_optimizer=checkpoint_type == "training",
            expires_at=expires_at,
        )
        self.checkpoints[tinker_path] = record
        return record

    def require_checkpoint(self, tinker_path: str) -> CheckpointRecord:
        try:
            checkpoint = self.checkpoints[tinker_path]
        except KeyError:
            raise TinkerError(f"checkpoint {tinker_path!r} does not exist", category="user") from None
        if checkpoint.status != "ready":
            raise TinkerError(
                f"checkpoint {tinker_path!r} is {checkpoint.status}",
                category="user" if checkpoint.status == "failed" else "server",
            )
        if checkpoint.expires_at is not None and checkpoint.expires_at <= time.time():
            raise TinkerError(f"checkpoint {tinker_path!r} has expired", category="user")
        return checkpoint

    def complete_checkpoint(self, tinker_path: str) -> CheckpointRecord:
        try:
            checkpoint = self.checkpoints[tinker_path]
        except KeyError:
            raise TinkerError(f"checkpoint {tinker_path!r} does not exist", category="server") from None
        checkpoint.status = "ready"
        return checkpoint

    def fail_checkpoint_for_request(self, request_id: str) -> None:
        sequence = next(
            ((model_id, seq_id) for (model_id, seq_id), (_fingerprint, stored_request_id) in self.model_request_keys.items() if stored_request_id == request_id),
            None,
        )
        if sequence is None:
            return
        model_id, seq_id = sequence
        for checkpoint in self.checkpoints.values():
            if checkpoint.model_id == model_id and checkpoint.seq_id == seq_id and checkpoint.status == "pending":
                checkpoint.status = "failed"
