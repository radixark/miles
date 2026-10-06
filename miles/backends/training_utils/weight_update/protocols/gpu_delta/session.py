"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from miles.utils.gpu_delta.publication import canonical_json, sha256


def merge_plans(descriptions: Sequence[dict]) -> tuple[list[dict], list[dict], str]:
    """Merge canonical views, never receiver-specific physical layout maps."""
    entries, identities = {}, []
    for description in descriptions:
        if description.get("success") is not True:
            raise RuntimeError(f"GPU-delta describe failed: {description.get('message')}")
        for participant in description["participants"]:
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
    engine_host_tensor_names: tuple[dict[str, list[str]], ...]


def negotiate_cohort(descriptions: Sequence[dict]) -> ReceiverCohort:
    plan, identities, digest = merge_plans(descriptions)
    participants = tuple(tuple(p["identity"] for p in d["participants"]) for d in descriptions)
    engine_ids = tuple(group[0]["engine_id"] for group in participants)
    engine_host_names = []
    for description in descriptions:
        host_names: dict[str, set[str]] = {}
        for participant in description["participants"]:
            host_id = participant["identity"]["host_cache_id"]
            host_names.setdefault(host_id, set()).update(tensor["name"] for tensor in participant["plan"]["tensors"])
        engine_host_names.append({host_id: sorted(names) for host_id, names in host_names.items()})
    return ReceiverCohort(plan, tuple(identities), participants, engine_ids, digest, tuple(engine_host_names))


def _receipts(response):
    if not response["success"]:
        raise RuntimeError(f"GPU-delta RPC failed: {response['message']}")
    return response["participants"]


async def activate_publication(clients, cohort: ReceiverCohort, publication, session_id: str | None = None):
    """Activate engines independently; settle all before advancing the trainer.

    An engine may resume while another prepares or remains failed/paused. Only
    that engine's successful apply reply authorizes its resume. After apply dispatch there
    is no automatic abort, resume or XOR retry. An RPC failure drains every
    other engine coroutine before returning an error to the trainer. External
    cancellation leaves an incomplete operation and does not authorize recovery.
    """
    started = time.monotonic()
    session_id = session_id or uuid.uuid4().hex
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
            participants=participants,
            host_tensor_names=host_names,
        )
        _receipts(preparation)
        await _wait_prepared(client, session_id=session_id)
    except Exception:
        # No pause or mutation was requested on this engine. Its uncertain
        # prepare reply does not authorize aborting another engine's lease.
        await asyncio.gather(client.abort_weights_from_delta(session_id=session_id), return_exceptions=True)
        raise
    prepared_at = time.monotonic()
    applied = await client.update_weights_from_delta(session_id=session_id)
    receipts = _receipts(applied)
    applied_at = time.monotonic()
    resumed = await client.resume_weights_from_delta(session_id=session_id)
    resumed_receipts = _receipts(resumed)
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


async def _wait_prepared(client, session_id, timeout=1800):
    async def poll():
        while True:
            response = await client.get_weights_delta_status(session_id=session_id)
            receipts = _receipts(response)
            if all(receipt["state"] == "PREPARED" for receipt in receipts):
                return receipts
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(poll(), timeout=timeout)
