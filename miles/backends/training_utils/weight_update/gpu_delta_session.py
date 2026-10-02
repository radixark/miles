"""Coordinator barriers for direct GPU deltas, independent of payload encoding."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence

from miles.utils.gpu_delta_publication import canonical_json, sha256


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
        for key in ("manifest_sha256", "stream_id", "base_version", "target_version"):
            if receipt.get(key) != publication[key]:
                raise RuntimeError(f"GPU-delta receipt {key} mismatch")
    return receipts


async def activate_publication(clients, descriptions, publication, *, session_id: str | None = None):
    """Prepare everyone while serving, then apply/commit/resume with exact cohorts.

    Every fanout settles before the next phase. After pause/apply starts, failures
    remain fail-closed; reconnection must not blindly replay an XOR publication.
    """
    session_id = session_id or uuid.uuid4().hex
    plan, cohort, plan_digest = merge_plans(descriptions)
    if publication["plan_digest"] != plan_digest:
        raise ValueError("Publication differs from the negotiated receiver plan")
    expected = [[p["identity"] for p in d["participants"]] for d in descriptions]
    engine_ids = [identities[0]["engine_id"] for identities in expected]
    if len(set(engine_ids)) != len(clients) or len(clients) != len(descriptions):
        raise ValueError("Duplicate/missing engine endpoints")
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
                cohort=cohort,
                expected_engines=engine_ids,
            )
            for client, engine_id, participants in zip(clients, engine_ids, expected, strict=True)
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
    pauses = await asyncio.gather(*[c.pause_generation(mode="retract") for c in clients], return_exceptions=True)
    _raise_rpc_errors(pauses)
    quiesced = await _wait_state(
        clients, expected, state="QUIESCED", pending="PREPARED", session_id=session_id, publication=publication
    )
    applied = await asyncio.gather(
        *[
            c.update_weights_from_delta(session_id=session_id, participants=p, receipts=quiesced)
            for c, p in zip(clients, expected, strict=True)
        ],
        return_exceptions=True,
    )
    receipts = _validate_phase(applied, expected, state="APPLIED", session_id=session_id, publication=publication)
    committed = await asyncio.gather(
        *[c.commit_weights_from_delta(session_id=session_id, receipts=receipts) for c in clients],
        return_exceptions=True,
    )
    certificate = _validate_phase(
        committed, expected, state="COMMITTED", session_id=session_id, publication=publication
    )
    resumed = await asyncio.gather(
        *[c.continue_generation(delta_session_id=session_id, delta_commit_receipts=certificate) for c in clients],
        return_exceptions=True,
    )
    resumed_receipts = _validate_phase(
        resumed, expected, state="RESUMED", session_id=session_id, publication=publication
    )
    return {
        "session_id": session_id,
        "receipts": certificate,
        "resumed_receipts": resumed_receipts,
        "plan_digest": plan_digest,
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
    # HTTP preparation/pause replies acknowledge enqueueing. Status comes from
    # each original rank and observes completion of its prepare/reader fence.
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
            raise RuntimeError("Original receiver identity changed during preparation/quiescence")
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
