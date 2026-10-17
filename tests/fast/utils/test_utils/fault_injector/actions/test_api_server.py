import re

import httpx
import pytest
from pydantic import ValidationError
from tests.fast.utils.test_utils.fault_injector.fakes import _ApiServer

from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction, StopProcessAction
from miles.utils.test_utils.fault_injector.actions.remote import API_SERVER_TIMEOUT_SECONDS, ApiServerFaultAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import ObservedFaultHookTarget

_ACTION = ApiServerFaultAction(base_url="http://api:9000", cell_id="rollout-0", rank=1, inner=KillProcessAction())


async def _run(action: ApiServerFaultAction = _ACTION) -> None:
    await action(context=FaultHookContext(rollout_id=3, weight_version=7), resources=FaultHookResources())


def _posted_command(server: _ApiServer) -> FaultHookCommand:
    [post] = [request for request in server.requests if request.method == "POST"]
    return FaultHookCommand.model_validate_json(post.content)


class TestApiServerFaultActionRequests:
    async def test_the_target_is_observed_before_the_hook_is_set_on_that_identity(
        self, api_server: _ApiServer
    ) -> None:
        """The remote fault must first read the rank's incarnation, then SET the inner action on exactly it."""
        await _run()

        get, post = api_server.requests
        assert (get.method, str(get.url)) == ("GET", "http://api:9000/api/v1/cells/rollout-0/fault-target?rank=1")
        assert (post.method, str(post.url)) == ("POST", "http://api:9000/api/v1/cells/rollout-0/fault-hook")
        assert post.headers["content-type"] == "application/json"
        command = _posted_command(api_server)
        assert command.operation == FaultHookOperation.SET
        assert command.request.action == KillProcessAction()
        assert command.request.target == ObservedFaultHookTarget(cell_id="rollout-0", rank=1, workers_hash="hash-a")

    async def test_the_remote_request_fires_on_receipt_and_matches_any_context(self, api_server: _ApiServer) -> None:
        """The remote SET must be immediate and unfiltered, so the receiver acts the moment it arrives."""
        await _run()

        request = _posted_command(api_server).request
        assert request.hook_name is None
        assert (request.rollout_id, request.attempt, request.weight_version) == (None, None, None)
        assert (request.delay_ms, request.lifetime_seconds) == (0.0, None)

    async def test_each_execution_sets_a_fresh_request_id(self, api_server: _ApiServer) -> None:
        """Two remote faults must never collide as a duplicate ID on the receiver."""
        await _run()
        await _run(_ACTION.model_copy(update={"inner": StopProcessAction()}))

        ids = [
            FaultHookCommand.model_validate_json(request.content).request.request_id
            for request in api_server.requests
            if request.method == "POST"
        ]
        assert len(set(ids)) == 2
        assert all(re.fullmatch(r"api_server_fault_[0-9a-f]{32}", request_id) for request_id in ids)

    async def test_the_client_is_bounded_by_the_api_timeout(self, api_server: _ApiServer) -> None:
        """A hung api server must not park the hook forever."""
        await _run()

        assert api_server.client_timeouts == [API_SERVER_TIMEOUT_SECONDS]


class TestApiServerFaultActionFailures:
    @pytest.mark.parametrize("status", [404, 412, 500, 504])
    async def test_a_failed_observation_sets_nothing(self, api_server: _ApiServer, status: int) -> None:
        """Without an observed identity the fault must not be sent to whatever answers next."""
        api_server.get_status = status

        with pytest.raises(httpx.HTTPStatusError):
            await _run()

        assert [request.method for request in api_server.requests] == ["GET"]

    @pytest.mark.parametrize(
        "target",
        [
            {"kind": "observed", "cell_id": "rollout-0", "rank": 1},
            {"kind": "declared", "cell_id": "rollout-0", "rank": 1},
        ],
    )
    async def test_an_observation_without_an_incarnation_sets_nothing(
        self, api_server: _ApiServer, target: dict[str, object]
    ) -> None:
        """A target answer without a worker identity must be rejected before any SET."""
        api_server.target = target

        with pytest.raises(ValidationError):
            await _run()

        assert [request.method for request in api_server.requests] == ["GET"]

    @pytest.mark.parametrize("status", [400, 404, 409, 412, 422])
    async def test_a_refused_set_is_raised(self, api_server: _ApiServer, status: int) -> None:
        """A client error means the fault definitely did not land and must fail the hook."""
        api_server.post_status = status

        with pytest.raises(httpx.HTTPStatusError):
            await _run()

    @pytest.mark.parametrize("status", [500, 502, 504])
    async def test_a_server_error_leaves_the_outcome_to_the_evidence(
        self, api_server: _ApiServer, status: int
    ) -> None:
        """A 5xx may hide a fault that did land, so the hook must not fail on it."""
        api_server.post_status = status

        await _run()

        assert [request.method for request in api_server.requests] == ["GET", "POST"]

    async def test_a_transport_timeout_is_raised(self, api_server: _ApiServer) -> None:
        """A transport timeout must surface instead of being mistaken for a delivered fault."""
        api_server.post_error = httpx.ReadTimeout("no answer")

        with pytest.raises(httpx.ReadTimeout):
            await _run()


class TestApiServerFaultActionModel:
    def test_the_inner_action_cannot_be_a_cell_or_remote_action(self) -> None:
        """Only a process fault may run on the receiving worker."""
        for inner in ({"kind": "stop_cell", "cell_id": "c"}, _ACTION.model_dump(mode="json")):
            with pytest.raises(ValidationError):
                ApiServerFaultAction.model_validate({**_ACTION.model_dump(mode="json"), "inner": inner})

    def test_a_negative_rank_is_rejected(self) -> None:
        """A negative rank would select a worker by accident on the receiver."""
        with pytest.raises(ValidationError):
            ApiServerFaultAction.model_validate({**_ACTION.model_dump(mode="json"), "rank": -1})
