"""Independent engine activation of one canonical GPU-delta publication."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from miles.utils.gpu_delta.publication import canonical_json, configured_skip_payload_hash, sha256


def merge_plans(descriptions: Sequence[dict]) -> tuple[list[dict], str]:
    """Merge canonical views, never receiver-specific physical layout maps."""
    entries, rank_ids = {}, set()
    for description in descriptions:
        if not description["success"]:
            raise RuntimeError(f"GPU-delta describe failed: {description['message']}")
        for participant in description["participants"]:
            rank_id = participant["identity"]["rank_id"]
            if rank_id in rank_ids:
                raise ValueError("Missing or duplicate original receiver identities")
            rank_ids.add(rank_id)
            for tensor in participant["plan"]["tensors"]:
                name = tensor["name"]
                spec = {key: tensor[key] for key in ("name", "dtype", "shape", "encoding")}
                spec.update({key: tensor[key] for key in ("format", "checkpoint_dtype") if key in tensor})
                entry = entries.setdefault(name, spec | {"views": {}})
                if {key: value for key, value in entry.items() if key != "views"} != spec:
                    raise ValueError(f"Receiver canonical plans differ for {name}")
                for view in tensor["views"]:
                    previous = entry["views"].setdefault(view["id"], view)
                    if previous != view:
                        raise ValueError(f"Receiver view IDs conflict for {name}")
    if not rank_ids:
        raise ValueError("Missing or duplicate original receiver identities")
    plan = [entry | {"views": sorted(entry["views"].values(), key=lambda v: v["id"])} for entry in entries.values()]
    plan.sort(key=lambda x: x["name"])
    return plan, sha256(canonical_json(plan))


@dataclass(frozen=True)
class ReceiverCohort:
    """Negotiated immutable engine membership and plan for one update stream."""

    plan: list[dict]
    participants: tuple[tuple[dict, ...], ...]
    engine_ids: tuple[str, ...]
    plan_digest: str


def negotiate_cohort(descriptions: Sequence[dict]) -> ReceiverCohort:
    plan, digest = merge_plans(descriptions)
    participants = tuple(tuple(p["identity"] for p in d["participants"]) for d in descriptions)
    engine_ids = tuple(group[0]["engine_id"] for group in participants)
    return ReceiverCohort(plan, participants, engine_ids, digest)


async def activate_publication(
    clients, cohort: ReceiverCohort, publication, session_id: str | None = None, weight_version: str | None = None
):
    """Activate engines independently; settle all before advancing the trainer.

    Each engine owns preparation, all-rank apply acknowledgement and resume.
    An RPC failure settles every other engine request before returning an error.
    A lost response may hide a completed update: the client never aborts or
    retries it. External cancellation also leaves an uncertain operation.
    """
    started = time.monotonic()
    session_id = session_id or uuid.uuid4().hex
    if len(clients) != len(cohort.engine_ids) or len(clients) != len(cohort.participants):
        raise ValueError("GPU-delta client and cohort sizes differ")
    common = {
        key: publication[key]
        for key in ("manifest_path", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
    }
    # Serving reports the actor's model version, while publication/receipt
    # versions retain their consecutive XOR-stream identity.
    common.update(
        session_id=session_id, weight_version=weight_version, validate_payload_hash=not configured_skip_payload_hash()
    )
    responses = await asyncio.gather(
        *[
            client.update_weights_from_gpu_delta(**common, participants=participants)
            for client, participants in zip(clients, cohort.participants, strict=True)
        ],
        return_exceptions=True,
    )
    for response in responses:
        if isinstance(response, BaseException):
            raise response
        if not response["success"]:
            raise RuntimeError(f"GPU-delta RPC failed: {response['message']}")
    return {
        "coordinator_timings": {"activation_s": time.monotonic() - started},
        "engine_timings": [
            {"engine_id": engine_id, **response["timings"]}
            for engine_id, response in zip(cohort.engine_ids, responses, strict=True)
        ],
        "session_id": session_id,
        "receipts": [receipt for response in responses for receipt in response["applied_participants"]],
        "resumed_receipts": [receipt for response in responses for receipt in response["participants"]],
        "plan_digest": cohort.plan_digest,
    }
