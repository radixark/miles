"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from miles.backends.training_utils.weight_update.protocols.gpu_delta.client import get_gpu_delta_info

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
    An unresolved RPC is returned to the FT controller for quarantine
    and replacement. Lost replies are reconciled per engine. Healthy engines
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


def committed_endpoint(description, participants, plan_digest):
    """Read one settled version only when every original rank agrees."""
    rows = description["participants"]
    if sorted((row["identity"] for row in rows), key=lambda rank: rank["rank_id"]) != sorted(
        participants, key=lambda rank: rank["rank_id"]
    ):
        raise RuntimeError("GPU-delta receiver incarnation changed")
    if any(row["state"] not in {"IDLE", "CLEARED", "RESUMED"} or row["plan_digest"] != plan_digest for row in rows):
        raise RuntimeError("GPU-delta receiver has unsettled weights or a different checkpoint")
    endpoints = {(row["stream_id"], row["version"]) for row in rows}
    if len(endpoints) != 1:
        raise RuntimeError("GPU-delta receiver ranks disagree about committed weights")
    return endpoints.pop()


async def activate_engine(
    client, publication, participants, session_id, weight_version, timeout, pause_mode=None, reverse=False
):
    """Activate one engine; its receipts must prove this session's completed activation.

    A rollout sync passes its ``--pause-generation-mode`` as ``pause_mode``, which also decides the cache flush.
    Eval pins and replacement recovery leave it unset: the receiver's defaults retract and flush, so no KV or
    prefix cache computed with older weights survives their activation.
    """
    request = {
        key: publication[key]
        for key in (
            "manifest_path",
            "manifest_sha256",
            "stream_id",
            "base_stream_id",
            "base_version",
            "target_version",
            "plan_digest",
        )
    }
    # Serving reports the actor's model version, while publication/receipt
    # versions retain their consecutive XOR-stream identity.
    request.update(session_id=session_id, participants=participants)
    if reverse:
        request["reverse"] = True
    base = (request["base_stream_id"], request["base_version"])
    target = (request["stream_id"], request["target_version"])
    source, destination = (target, base) if reverse else (base, target)
    if weight_version is not None:
        request["weight_version"] = weight_version
    request["validate_payload_hash"] = _VALIDATE_PAYLOAD_HASH[publication["payload_checksum_format"]]
    if pause_mode is not None:
        request["pause_mode"] = pause_mode
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(client._make_request("update_weights_from_gpu_delta", request), timeout)
            break
        except (TimeoutError, httpx.TransportError, httpx.HTTPStatusError) as error:
            rejected = isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500
            if rejected and not (attempt and error.response.status_code == 409):
                raise
            # The tokenizer serializes this read with the complete update, including resume.
            info = await asyncio.wait_for(
                get_gpu_delta_info(client, participants[0]["engine_id"], session_id), timeout
            )
            current = committed_endpoint(info, participants, request["plan_digest"])
            if current == destination and info.get("activation") is not None:
                response = info["activation"]
                break
            if current != source or attempt or rejected:
                raise
    resumed = response["participants"]
    actual = sorted((row["identity"] for row in resumed), key=lambda identity: identity["rank_id"])
    expected = sorted(participants, key=lambda identity: identity["rank_id"])
    if actual != expected:
        raise RuntimeError("GPU-delta receiver incarnation changed during activation")
    for receipt in resumed:
        if (
            receipt["state"] != "RESUMED"
            or (receipt["committed_stream_id"], receipt["version"]) != destination
            or receipt["reverse"] != reverse
            or any(
                receipt[key] != request[key]
                for key in (
                    "session_id",
                    "stream_id",
                    "base_stream_id",
                    "manifest_sha256",
                    "base_version",
                    "target_version",
                    "plan_digest",
                )
            )
        ):
            raise RuntimeError("GPU-delta receiver has no matching completed activation")
    return response
