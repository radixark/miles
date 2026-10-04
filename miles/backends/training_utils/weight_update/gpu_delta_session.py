"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miles.utils.gpu_delta_publication import CODEC, canonical_json, sha256


def merge_plans(descriptions: Sequence[dict], codec: str = CODEC) -> tuple[list[dict], list[dict], str]:
    """Merge canonical views, never receiver-specific physical layout maps."""
    if codec != CODEC:
        raise ValueError(f"Unsupported GPU-delta codec: {codec}")
    entries, identities = {}, []
    for description in descriptions:
        if description.get("success") is not True:
            raise RuntimeError(f"GPU-delta describe failed: {description.get('message')}")
        for participant in description["participants"]:
            if participant["plan"].get("codec") != codec:
                raise ValueError("Sender and receiver GPU-delta codecs differ")
            identities.append(participant["identity"])
            for tensor in participant["plan"]["tensors"]:
                name = tensor["name"]
                spec = {key: tensor[key] for key in ("name", "dtype", "shape", "encoding")}
                entry = entries.setdefault(name, spec | {"views": {}})
                if any(entry[key] != value for key, value in spec.items()):
                    raise ValueError(f"Receiver canonical plans differ for {name}")
                for view in tensor["views"]:
                    previous = entry["views"].setdefault(view["id"], view)
                    if previous != view:
                        raise ValueError(f"Receiver view IDs conflict for {name}")
    if not identities or len({x["rank_id"] for x in identities}) != len(identities):
        raise ValueError("Missing or duplicate original receiver identities")
    plan = [entry | {"views": sorted(entry["views"].values(), key=lambda v: v["id"])} for entry in entries.values()]
    plan.sort(key=lambda x: x["name"])
    return plan, identities, sha256(canonical_json(plan))


@dataclass(frozen=True)
class ReceiverCohort:
    """Negotiated immutable engine membership and plan for one update stream."""

    plan: list[dict]
    identities: tuple[dict, ...]
    participants: tuple[tuple[dict, ...], ...]
    engine_ids: tuple[str, ...]
    plan_digest: str
    codec: str
    engine_host_tensor_names: tuple[dict[str, list[str]], ...]


def negotiate_cohort(descriptions: Sequence[dict], codec: str = CODEC) -> ReceiverCohort:
    plan, identities, digest = merge_plans(descriptions, codec=codec)
    participants = tuple(tuple(dict(p["identity"]) for p in d["participants"]) for d in descriptions)
    if any(not group or len({p["engine_id"] for p in group}) != 1 for group in participants):
        raise ValueError("Missing or mixed engine participants")
    engine_ids = tuple(group[0]["engine_id"] for group in participants)
    if len(set(engine_ids)) != len(engine_ids):
        raise ValueError("Duplicate engine endpoints")
    host_names: dict[str, set[str]] = {}
    host_owners = {}
    for description in descriptions:
        for participant in description["participants"]:
            host_id = participant["identity"].get("host_cache_id")
            if not isinstance(host_id, str) or not host_id:
                raise ValueError("Receiver must advertise its engine-host cache identity")
            engine_id = participant["identity"]["engine_id"]
            if host_owners.setdefault(host_id, engine_id) != engine_id:
                raise ValueError("Independent engines must not share a host arena")
            host_names.setdefault(host_id, set()).update(tensor["name"] for tensor in participant["plan"]["tensors"])
    host_tensor_names = {host_id: sorted(names) for host_id, names in sorted(host_names.items())}
    engine_host_names = tuple(
        {host_id: host_tensor_names[host_id] for host_id in sorted({p["host_cache_id"] for p in group})}
        for group in participants
    )
    return ReceiverCohort(plan, tuple(identities), participants, engine_ids, digest, codec, engine_host_names)


def validate_receipts(response: Mapping, expected: Sequence[dict], states: set[str], session_id: str, publication: dict):
    if response.get("success") is not True:
        raise RuntimeError(f"GPU-delta RPC rejected: {response.get('message')}")
    receipts = response.get("participants", [])
    identities = [r.get("identity") for r in receipts]
    if sorted(canonical_json(x) for x in identities) != sorted(canonical_json(x) for x in expected):
        raise RuntimeError("GPU-delta receipt participant set differs from the original cohort")
    for receipt in receipts:
        if receipt.get("state") not in states or receipt.get("session_id") != session_id:
            raise RuntimeError("GPU-delta receipt state/session mismatch")
        for key in ("manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest"):
            if receipt.get(key) != publication[key]:
                raise RuntimeError(f"GPU-delta receipt {key} mismatch")
    return receipts


async def activate_publication(clients, cohort: ReceiverCohort, publication, session_id: str | None = None):
    """Activate engines independently; settle all before advancing the trainer.

    An engine may resume while another prepares or remains failed/paused. Only
    that engine's original ranks certify its resume. After apply dispatch there
    is no automatic abort, resume or XOR retry. An RPC failure drains every
    other engine coroutine before returning an error to the trainer. External
    cancellation leaves an incomplete operation and does not authorize recovery.
    """
    started = time.monotonic()
    session_id = session_id or uuid.uuid4().hex
    if publication["codec"] != cohort.codec or publication["plan_digest"] != cohort.plan_digest:
        raise ValueError("Publication differs from the negotiated receiver plan/codec")
    if len(clients) != len(cohort.engine_ids):
        raise ValueError("Missing engine endpoints")
    results = await asyncio.gather(
        *[
            _activate_engine(client, engine_id, participants, host_names, publication, session_id)
            for client, engine_id, participants, host_names in zip(
                clients, cohort.engine_ids, cohort.participants, cohort.engine_host_tensor_names, strict=True
            )
        ],
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return {
        "coordinator_timings": {"activation_s": time.monotonic() - started},
        "engine_timings": [result["timings"] for result in results],
        "session_id": session_id,
        "receipts": [receipt for result in results for receipt in result["receipts"]],
        "resumed_receipts": [receipt for result in results for receipt in result["resumed_receipts"]],
        "plan_digest": cohort.plan_digest,
    }


async def _activate_engine(client, engine_id, participants, host_names, publication, session_id):
    started = time.monotonic()
    common = {
        key: publication[key]
        for key in ("manifest_path", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
    }
    try:
        preparation = await client.prepare_weights_from_delta(
            **common,
            session_id=session_id,
            engine_id=engine_id,
            participants=participants,
            host_tensor_names=host_names,
        )
        validate_receipts(
            preparation, participants, states={"PREPARING", "PREPARED"}, session_id=session_id, publication=publication
        )
        await _wait_state(client, participants, session_id=session_id, publication=publication)
    except Exception:
        # No pause or mutation was requested on this engine. Its uncertain
        # prepare reply does not authorize aborting another engine's lease.
        await asyncio.gather(client.abort_weights_from_delta(session_id=session_id), return_exceptions=True)
        raise
    prepared_at = time.monotonic()
    applied = await client.update_weights_from_delta(session_id=session_id)
    receipts = validate_receipts(applied, participants, states={"APPLIED"}, session_id=session_id, publication=publication)
    certificate = [receipt["certificate"] for receipt in receipts]
    applied_at = time.monotonic()
    resumed = await client.resume_weights_from_delta(session_id=session_id, receipts=certificate)
    resumed_receipts = validate_receipts(
        resumed, participants, states={"RESUMED"}, session_id=session_id, publication=publication
    )
    resumed_at = time.monotonic()
    return {
        "timings": {
            "engine_id": engine_id,
            "prepare_s": prepared_at - started,
            "apply_s": applied_at - prepared_at,
            "resume_s": resumed_at - applied_at,
            "activation_s": resumed_at - started,
        },
        "receipts": receipts,
        "resumed_receipts": resumed_receipts,
    }


async def _wait_state(client, participants, session_id, publication, timeout=1800):
    async def poll():
        while True:
            response = await client.get_weights_delta_status(session_id=session_id)
            receipts = validate_receipts(
                response, participants, states={"PREPARING", "PREPARED"}, session_id=session_id, publication=publication
            )
            if all(receipt["state"] == "PREPARED" for receipt in receipts):
                return receipts
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(poll(), timeout=timeout)
