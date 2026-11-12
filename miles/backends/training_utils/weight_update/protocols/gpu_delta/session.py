"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from miles.utils.gpu_delta.publication import configured_skip_payload_hash


@dataclass(frozen=True)
class ReceiverPool:
    """Current receiver identities; independent of the publication contract."""

    participants: tuple[tuple[dict, ...], ...]
    engine_ids: tuple[str, ...]


def receiver_pool(descriptions: Sequence[dict]) -> ReceiverPool:
    participants, engine_ids, rank_ids = [], [], set()
    for description in descriptions:
        if not description["success"]:
            raise RuntimeError(f"GPU-delta admission failed: {description['message']}")
        identities = tuple(row["identity"] for row in description["participants"])
        if not identities:
            raise ValueError("Missing or duplicate original receiver identities")
        engine_id = identities[0]["engine_id"]
        for identity in identities:
            if identity["rank_id"] in rank_ids or identity["engine_id"] != engine_id:
                raise ValueError("Missing or duplicate original receiver identities")
            rank_ids.add(identity["rank_id"])
        participants.append(identities)
        engine_ids.append(engine_id)
    if not participants or len(set(engine_ids)) != len(engine_ids):
        raise ValueError("Missing or duplicate original receiver identities")
    return ReceiverPool(tuple(participants), tuple(engine_ids))


async def activate_publication(
    clients,
    pool: ReceiverPool,
    publication,
    session_id: str | None = None,
    weight_version: str | None = None,
    fault_tolerant=False,
    request_timeout=None,
):
    """Activate engines independently; settle all before advancing the trainer.

    Each engine owns preparation, all-rank apply acknowledgement and resume.
    An RPC failure settles every other engine request before returning an error.
    A failed or ambiguous RPC is returned to the FT controller for quarantine
    and replacement. A lost reply never triggers an XOR retry. Healthy engines
    retain their completed activation independently.
    """
    started = time.monotonic()
    session_id = session_id or uuid.uuid4().hex
    if len(clients) != len(pool.engine_ids) or len(clients) != len(pool.participants):
        raise ValueError("GPU-delta client and pool sizes differ")
    responses = await asyncio.gather(
        *[
            activate_engine(client, publication, participants, session_id, weight_version, request_timeout)
            for client, participants in zip(clients, pool.participants, strict=True)
        ],
        return_exceptions=True,
    )
    failures = {
        engine_id: f"{type(response).__name__}: {response}"
        for engine_id, response in zip(pool.engine_ids, responses, strict=True)
        if isinstance(response, BaseException)
    }
    if failures and not fault_tolerant:
        raise RuntimeError(f"GPU-delta activation requires receiver recovery: {failures}")
    completed = [
        (engine_id, response)
        for engine_id, response in zip(pool.engine_ids, responses, strict=True)
        if not isinstance(response, BaseException)
    ]
    return {
        "coordinator_timings": {"activation_s": time.monotonic() - started},
        "engine_timings": [{"engine_id": engine_id, **response["timings"]} for engine_id, response in completed],
        "session_id": session_id,
        "receipts": [receipt for _, response in completed for receipt in response["applied_participants"]],
        "resumed_receipts": [receipt for _, response in completed for receipt in response["participants"]],
        "succeeded_engine_ids": [engine_id for engine_id, _ in completed],
        "failed_engines": failures,
        "plan_digest": publication["plan_digest"],
    }


async def activate_engine(client, publication, participants, session_id, weight_version, timeout):
    """Activate one engine; its receipts must prove this session's completed activation."""
    request = {
        key: publication[key]
        for key in ("manifest_path", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
    }
    # Serving reports the actor's model version, while publication/receipt
    # versions retain their consecutive XOR-stream identity.
    request.update(session_id=session_id, weight_version=weight_version)
    request["validate_payload_hash"] = not configured_skip_payload_hash()
    response = await asyncio.wait_for(
        client.update_weights_from_gpu_delta(**request, participants=participants), timeout
    )
    if not response["success"]:
        raise RuntimeError(f"GPU-delta RPC failed: {response['message']}")
    check_resumed(response["participants"], participants, request)
    return response


def check_resumed(rows, participants, publication):
    """Prove completed activation; all other outcomes require receiver replacement."""
    _check_identities(rows, participants)
    for receipt in rows:
        if receipt["state"] != "RESUMED" or any(
            receipt[key] != publication[key]
            for key in ("session_id", "stream_id", "manifest_sha256", "target_version", "plan_digest")
        ):
            raise RuntimeError("GPU-delta receiver has no matching completed activation")


def _check_identities(rows, participants):
    actual = sorted((row["identity"] for row in rows), key=lambda identity: identity["rank_id"])
    expected = sorted(participants, key=lambda identity: identity["rank_id"])
    if actual != expected:
        raise RuntimeError("GPU-delta receiver incarnation changed during activation")
