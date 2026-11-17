"""GPU-delta receiver controls, sent through an engine client's request helper.

The generic ``SGLangApiClient`` carries no GPU-delta route. Its ``_make_request`` posts each control here; the manual
benchmark's direct client provides the same helper.
"""


def info_request(engine_id):
    """``get_gpu_delta_info`` as request-helper arguments; a rollout cell updater submits them as they are."""
    return {"endpoint": "get_gpu_delta_info", "payload": {"engine_id": engine_id}}


async def get_gpu_delta_info(client, engine_id):
    return await client._make_request(**info_request(engine_id))


async def update_weights_from_gpu_delta(
    client,
    session_id,
    manifest_path,
    manifest_sha256,
    stream_id,
    base_version,
    target_version,
    plan_digest,
    participants,
    release_state: bool | None = None,
    pause_mode: str | None = None,
    abort_all_requests: bool | None = None,
    weight_version: str | None = None,
    validate_payload_hash: bool | None = None,
):
    # Unset options stay out of the payload: the server's request schema owns their defaults.
    options = {
        "release_state": release_state,
        "pause_mode": pause_mode,
        "abort_all_requests": abort_all_requests,
        "weight_version": weight_version,
        "validate_payload_hash": validate_payload_hash,
    }
    payload = {
        "session_id": session_id,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "stream_id": stream_id,
        "base_version": base_version,
        "target_version": target_version,
        "plan_digest": plan_digest,
        "participants": participants,
        **{key: value for key, value in options.items() if value is not None},
    }
    return await client._make_request("update_weights_from_gpu_delta", payload)


async def clear_gpu_delta_state(client):
    return await client._make_request("clear_gpu_delta_state", {})
