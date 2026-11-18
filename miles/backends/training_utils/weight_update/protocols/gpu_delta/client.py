"""GPU-delta receiver controls, sent through an engine client's request helper.

The generic ``SGLangApiClient`` carries no GPU-delta route. Its ``_make_request`` posts each control here; the manual
benchmark's direct client provides the same helper.
"""


def info_request(engine_id):
    """``get_gpu_delta_info`` as request-helper arguments; a rollout cell updater submits them as they are."""
    return {"endpoint": "get_gpu_delta_info", "payload": {"engine_id": engine_id}}


async def get_gpu_delta_info(client, engine_id):
    return await client._make_request(**info_request(engine_id))


async def clear_gpu_delta_state(client):
    return await client._make_request("clear_gpu_delta_state", {})
