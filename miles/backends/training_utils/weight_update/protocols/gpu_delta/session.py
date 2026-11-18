"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

# Every descriptor carries its writer's launch-validated payload-hash policy.
_VALIDATE_PAYLOAD_HASH = {"sha256": True, "none": False}


@dataclass(frozen=True)
class EngineIdentities:
    """Current receiver incarnations per engine; independent of the publication contract."""

    participants: tuple[tuple[dict, ...], ...]
    engine_ids: tuple[str, ...]


def engine_identities(descriptions: Sequence[dict]) -> EngineIdentities:
    participants, engine_ids, rank_ids = [], [], set()
    for description in descriptions:
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
    return EngineIdentities(tuple(participants), tuple(engine_ids))


async def activate_publication(
    clients,
    identities: EngineIdentities,
    publication,
    session_id: str | None = None,
    weight_version: str | None = None,
    fault_tolerant=False,
    request_timeout=None,
    pause_mode: str | None = None,
):
    """Activate engines independently; settle all before advancing the trainer.

    Each engine owns preparation, all-rank apply acknowledgement and resume.
    An RPC failure settles every other engine request before returning an error.
    A failed or ambiguous RPC is returned to the FT controller for quarantine
    and replacement. A lost reply never triggers an XOR retry. Healthy engines
    retain their completed activation independently. Only a rollout sync
    passes ``pause_mode`` (see ``activate_engine``).
    """
    started = time.monotonic()
    session_id = session_id or uuid.uuid4().hex
    if len(clients) != len(identities.engine_ids) or len(clients) != len(identities.participants):
        raise ValueError("GPU-delta client and engine identity counts differ")
    responses = await asyncio.gather(
        *[
            activate_engine(
                client, publication, participants, session_id, weight_version, request_timeout, pause_mode=pause_mode
            )
            for client, participants in zip(clients, identities.participants, strict=True)
        ],
        return_exceptions=True,
    )
    failures = {
        # format_exception_only keeps the client's note with the receiver's 409 body (its typed message).
        engine_id: "".join(traceback.format_exception_only(response)).strip()
        for engine_id, response in zip(identities.engine_ids, responses, strict=True)
        if isinstance(response, BaseException)
    }
    if failures and not fault_tolerant:
        raise RuntimeError(f"GPU-delta activation requires receiver recovery: {failures}")
    completed = [
        (engine_id, response)
        for engine_id, response in zip(identities.engine_ids, responses, strict=True)
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


async def activate_engine(client, publication, participants, session_id, weight_version, timeout, pause_mode=None):
    """Activate one engine; its receipts must prove this session's completed activation.

    A rollout sync passes its ``--pause-generation-mode`` as ``pause_mode``, which also decides the cache flush.
    Eval pins and replacement recovery leave it unset: the receiver's defaults retract and flush, so no KV or
    prefix cache computed with older weights survives their activation.
    """
    request = {
        key: publication[key]
        for key in ("manifest_path", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
    }
    # Serving reports the actor's model version, while publication/receipt
    # versions retain their consecutive XOR-stream identity.
    request.update(session_id=session_id, participants=participants)
    if weight_version is not None:
        request["weight_version"] = weight_version
    request["validate_payload_hash"] = _VALIDATE_PAYLOAD_HASH[publication["payload_checksum_format"]]
    if pause_mode is not None:
        request["pause_mode"] = pause_mode
    response = await asyncio.wait_for(client._make_request("update_weights_from_gpu_delta", request), timeout)
    resumed = response["participants"]
    actual = sorted((row["identity"] for row in resumed), key=lambda identity: identity["rank_id"])
    expected = sorted(participants, key=lambda identity: identity["rank_id"])
    if actual != expected:
        raise RuntimeError("GPU-delta receiver incarnation changed during activation")
    for receipt in resumed:
        if receipt["state"] != "RESUMED" or any(
            receipt[key] != request[key]
            for key in ("session_id", "stream_id", "manifest_sha256", "target_version", "plan_digest")
        ):
            raise RuntimeError("GPU-delta receiver has no matching completed activation")
    return response
