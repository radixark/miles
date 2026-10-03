"""Coordinator barriers for direct GPU deltas, independent of payload encoding."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miles.utils.gpu_delta_publication import CODEC, canonical_json, sha256


def merge_plans(descriptions: Sequence[dict], *, codec: str = CODEC) -> tuple[list[dict], list[dict], str]:
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
    host_tensor_names: dict[str, list[str]]


def negotiate_cohort(descriptions: Sequence[dict], *, codec: str = CODEC) -> ReceiverCohort:
    plan, identities, digest = merge_plans(descriptions, codec=codec)
    participants = tuple(tuple(dict(p["identity"]) for p in d["participants"]) for d in descriptions)
    if any(not group or len({p["engine_id"] for p in group}) != 1 for group in participants):
        raise ValueError("Missing or mixed engine participants")
    engine_ids = tuple(group[0]["engine_id"] for group in participants)
    if len(set(engine_ids)) != len(engine_ids):
        raise ValueError("Duplicate engine endpoints")
    host_names: dict[str, set[str]] = {}
    for description in descriptions:
        for participant in description["participants"]:
            host_id = participant["identity"].get("host_cache_id")
            if not isinstance(host_id, str) or not host_id:
                raise ValueError("Receiver must advertise its shared host-cache identity")
            host_names.setdefault(host_id, set()).update(tensor["name"] for tensor in participant["plan"]["tensors"])
    host_tensor_names = {host_id: sorted(names) for host_id, names in sorted(host_names.items())}
    return ReceiverCohort(plan, tuple(identities), participants, engine_ids, digest, codec, host_tensor_names)


def validate_receipts(response: Mapping, expected: list[dict], *, state: str, session_id: str, publication: dict):
    if response.get("success") is not True:
        raise RuntimeError(f"GPU-delta {state} rejected: {response.get('message')}")
    receipts = response.get("participants", [])
    identities = [r.get("identity") for r in receipts]
    if sorted(canonical_json(x) for x in identities) != sorted(canonical_json(x) for x in expected):
        raise RuntimeError("GPU-delta receipt participant set differs from the original cohort")
    for receipt in receipts:
        if receipt.get("state") != state or receipt.get("session_id") != session_id:
            raise RuntimeError("GPU-delta receipt state/session mismatch")
        for key in ("manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest"):
            if receipt.get(key) != publication[key]:
                raise RuntimeError(f"GPU-delta receipt {key} mismatch")
    return receipts


async def activate_publication(clients, cohort: ReceiverCohort, publication, *, session_id: str | None = None):
    """Prepare while serving, then locally pause/apply and globally certify resume.

    One coordinator owns the original engines throughout this operation; competing
    updates or engine administration are unsupported. Every fanout settles before
    the next phase. After apply starts, failures are terminal: never abort, resume
    without all APPLIED receipts, or blindly replay an XOR publication.
    """
    session_id = session_id or uuid.uuid4().hex
    if publication["codec"] != cohort.codec or publication["plan_digest"] != cohort.plan_digest:
        raise ValueError("Publication differs from the negotiated receiver plan/codec")
    if len(clients) != len(cohort.engine_ids):
        raise ValueError("Missing engine endpoints")
    expected = cohort.participants
    common = {
        key: publication[key]
        for key in ("manifest_path", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
    }
    preparations = await asyncio.gather(
        *[
            client.prepare_weights_from_delta(
                **common,
                session_id=session_id,
                engine_id=engine_id,
                participants=participants,
                cohort=cohort.identities,
                host_tensor_names=cohort.host_tensor_names,
            )
            for client, engine_id, participants in zip(clients, cohort.engine_ids, expected, strict=True)
        ],
        return_exceptions=True,
    )
    try:
        _validate_progress(
            preparations, expected, states={"PREPARING", "PREPARED"}, session_id=session_id, publication=publication
        )
        await _wait_state(
            clients, expected, state="PREPARED", pending="PREPARING", session_id=session_id, publication=publication
        )
    except Exception:
        # No pause or mutation was requested. Abort every endpoint, including an
        # uncertain prepare reply; the session ID names the only possible lease.
        await asyncio.gather(
            *[c.abort_weights_from_delta(session_id=session_id) for c in clients], return_exceptions=True
        )
        raise
    applied = await asyncio.gather(
        *[c.update_weights_from_delta(session_id=session_id) for c in clients],
        return_exceptions=True,
    )
    receipts = _validate_phase(applied, expected, state="APPLIED", session_id=session_id, publication=publication)
    # Keep results/timings in the returned evidence, not the all-rank certificate
    # copied to every scheduler. SGLang constructs this from its APPLIED state.
    certificate = [receipt["certificate"] for receipt in receipts]
    resumed = await asyncio.gather(
        *[c.resume_weights_from_delta(session_id=session_id, receipts=certificate) for c in clients],
        return_exceptions=True,
    )
    resumed_receipts = _validate_phase(
        resumed, expected, state="RESUMED", session_id=session_id, publication=publication
    )
    return {
        "session_id": session_id,
        "receipts": receipts,
        "resumed_receipts": resumed_receipts,
        "plan_digest": cohort.plan_digest,
    }


def _raise_rpc_errors(results):
    for result in results:
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"GPU-delta RPC rejected: {result.get('message')}")


def _validate_phase(results, expected, *, state, session_id, publication):
    _raise_rpc_errors(results)
    return [
        receipt
        for result, participants in zip(results, expected, strict=True)
        for receipt in validate_receipts(
            result, participants, state=state, session_id=session_id, publication=publication
        )
    ]


async def _wait_state(clients, expected, *, state, pending, session_id, publication, timeout=1800):
    # Preparation starts background work. Poll each original rank until its
    # immutable inputs are ready, before asking any engine to pause and apply.
    async def poll():
        while True:
            statuses = await asyncio.gather(
                *[c.get_weights_delta_status(session_id=session_id) for c in clients], return_exceptions=True
            )
            receipts = _validate_progress(
                statuses, expected, states={pending, state}, session_id=session_id, publication=publication
            )
            if all(receipt["state"] == state for receipt in receipts):
                return receipts
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(poll(), timeout=timeout)


def _validate_progress(results, expected, *, states, session_id, publication):
    _raise_rpc_errors(results)
    receipts = []
    for response, participants in zip(results, expected, strict=True):
        if response.get("success") is not True:
            raise RuntimeError("GPU-delta progress request failed")
        actual = [r.get("identity") for r in response.get("participants", [])]
        if sorted(map(canonical_json, actual)) != sorted(map(canonical_json, participants)):
            raise RuntimeError("Original receiver identity changed during preparation")
        for receipt in response["participants"]:
            if receipt.get("state") not in states:
                raise RuntimeError(f"Unexpected gpu-delta progress state: {receipt.get('state')}")
            validate_receipts(
                {"success": True, "participants": [receipt]},
                [receipt["identity"]],
                state=receipt["state"],
                session_id=session_id,
                publication=publication,
            )
            receipts.append(receipt)
    return receipts
