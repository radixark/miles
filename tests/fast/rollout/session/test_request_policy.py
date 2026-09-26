import pytest

from miles.rollout.session.request_policy import RolloutRequestContext, prepare_rollout_request


@pytest.mark.asyncio
async def test_prepare_rollout_request_applies_async_hook():
    async def hook(hook_args, context, request):
        assert hook_args == {"minimum_version": 7}
        assert context.session_id == "session-1"
        request["payload"]["weight_version"] = {"min_version": hook_args["minimum_version"]}
        return {"max_attempts": 3, "retry_interval": 0.25}

    request = await prepare_rollout_request(
        hook,
        {"minimum_version": 7},
        RolloutRequestContext(session_id="session-1"),
        payload={},
        headers={},
    )

    assert request["payload"]["weight_version"] == {"min_version": 7}
    assert request["max_attempts"] == 3
    assert request["retry_interval"] == 0.25


@pytest.mark.asyncio
async def test_prepare_rollout_request_rejects_invalid_retry_policy():
    def hook(_hook_args, _context, _request):
        return {"max_attempts": 0}

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        await prepare_rollout_request(
            hook,
            {},
            RolloutRequestContext(session_id="session-1"),
            payload={},
            headers={},
        )
