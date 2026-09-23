import random

import httpx
import pytest
from tests.fast.utils.soak.soak_fakes import (
    _at,
    _cell,
    _cell_target,
    _FakeCellApi,
    _fault_target,
    _observation,
    _patch_http,
    _raising_hook_transport,
    _with_fault_target,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions import inject_fault as inject_fault_module
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, ObservedCellFaultKind

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.test_utils.fault_injector.actions.process import ExitProcessAction, KillProcessAction
from miles.utils.test_utils.fault_injector.controller import FaultHookOperation
from miles.utils.workers.naming import compute_cell_id

_BASE_URL = "http://api:18080"
_ACTOR_0 = compute_cell_id(pool_id="actor", cell_index=0)


def _form() -> InjectFaultForm:
    return InjectFaultForm(base_url=_BASE_URL, action=KillProcessAction())


def _create(
    form: InjectFaultForm, target: CellTarget, *, events: tuple[SoakEvent, ...] | list[SoakEvent] = ()
) -> SoakActionRequest | None:
    return form.maybe_create_request(
        target=target, observation=_observation([target], at=_at(0)), events=list(events), rng=random.Random(0)
    )


async def _execute(form: InjectFaultForm, request: SoakActionRequest) -> list[SoakActionEvidence]:
    reported: list[SoakActionEvidence] = []
    await form.execute(request, report_applied=reported.append)
    return reported


class TestInjectFaultFormName:
    def test_the_name_encodes_the_action(self) -> None:
        """Forms injecting different actions get distinct names so find_form can tell them apart."""
        assert _form().name == "inject_fault:kill_process"
        assert InjectFaultForm(base_url=_BASE_URL, action=ExitProcessAction()).name == "inject_fault:exit_process"


class TestInjectFaultFormRequest:
    def test_a_target_without_a_fault_target_is_declined(self) -> None:
        """Without an observed rank-0 worker there is nothing to address."""
        assert _create(_form(), _cell_target()) is None

    def test_a_fault_target_of_an_older_incarnation_is_declined(self) -> None:
        """A fault target read before the cell was replaced would hit the wrong process."""
        target = _cell_target(incarnation="new").model_copy(
            update={"fault_target": _fault_target(_ACTOR_0, workers_hash="old")}
        )

        assert _create(_form(), target) is None

    def test_a_request_names_the_observed_fault_target(self) -> None:
        """The request addresses exactly the worker observed for the current incarnation."""
        target = _with_fault_target(_cell_target())

        request = _create(_form(), target)

        assert (request.target, request.form_name) == (target, _form().name)
        assert request.details == InjectFaultDetails(fault_target=target.fault_target)


class TestInjectFaultFormExecute:
    async def test_the_fault_is_set_on_the_target_and_a_missing_cell_is_the_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SET command naming this request is posted and a vanished cell is reported as applied."""
        api = _FakeCellApi([])
        _patch_http(monkeypatch, api)
        form = _form()
        request = _create(form, _with_fault_target(_cell_target()))

        [evidence] = await _execute(form, request)

        [(cell_id, command)] = api.hook_posts
        assert cell_id == _ACTOR_0
        assert command.operation is FaultHookOperation.SET
        assert command.request.request_id == request.request_id
        assert command.request.action == KillProcessAction()
        assert command.request.target == request.details.fault_target
        assert evidence == ObservedCellFault(
            request_id=request.request_id,
            target=request.details.fault_target,
            action=KillProcessAction(),
            observed=ObservedCellFaultKind.MISSING,
        )

    async def test_a_new_incarnation_is_reported_as_a_replacement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A changed workers hash is the effect and carries the new incarnation."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor", workers_hash="inc-b")])
        _patch_http(monkeypatch, api)
        form = _form()

        [evidence] = await _execute(form, _create(form, _with_fault_target(_cell_target())))

        assert (evidence.observed, evidence.observed_workers_hash) == (ObservedCellFaultKind.REPLACED, "inc-b")

    async def test_an_unhealthy_cell_of_the_same_incarnation_is_the_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cell that stopped being healthy has visibly been hit."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor", workers_hash="inc-a", healthy=TriState.FALSE)])
        _patch_http(monkeypatch, api)
        form = _form()

        [evidence] = await _execute(form, _create(form, _with_fault_target(_cell_target())))

        assert evidence.observed is ObservedCellFaultKind.UNHEALTHY

    async def test_unreadable_or_unchanged_cells_are_not_an_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Transport errors, server errors and a healthy unchanged cell keep waiting for the real effect."""
        api = _FakeCellApi([])
        api.cell_replies[_ACTOR_0] = [
            httpx.ConnectError("down"),
            503,
            _cell(_ACTOR_0, cell_type="actor", workers_hash="inc-a"),
            404,
        ]
        _patch_http(monkeypatch, api)
        form = _form()

        [evidence] = await _execute(form, _create(form, _with_fault_target(_cell_target())))

        assert evidence.observed is ObservedCellFaultKind.MISSING
        assert api.cell_replies[_ACTOR_0] == [404]

    @pytest.mark.parametrize("post", [500, "transport"])
    async def test_an_unknown_submission_outcome_still_watches_for_the_effect(
        self, monkeypatch: pytest.MonkeyPatch, post: int | str
    ) -> None:
        """A server error or lost reply may still have fired the fault, so the effect decides."""
        api = _FakeCellApi([])
        if isinstance(post, int):
            api.hook_status = post
        _patch_http(monkeypatch, _raising_hook_transport(api) if post == "transport" else api)
        form = _form()

        [evidence] = await _execute(form, _create(form, _with_fault_target(_cell_target())))

        assert evidence.observed is ObservedCellFaultKind.MISSING

    async def test_a_rejected_submission_fails_without_an_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A client error means the fault was refused, so nothing is reported applied."""
        api = _FakeCellApi([])
        api.hook_status = 409
        _patch_http(monkeypatch, api)
        form = _form()
        request = _create(form, _with_fault_target(_cell_target()))
        reported: list[SoakActionEvidence] = []

        with pytest.raises(httpx.HTTPStatusError):
            await form.execute(request, report_applied=reported.append)
        assert reported == []

    async def test_an_effect_that_never_appears_times_out_without_an_applied_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unchanged healthy cell after the bound is a failed injection, not a silent success."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor", workers_hash="inc-a")])
        _patch_http(monkeypatch, api)
        monkeypatch.setattr(inject_fault_module, "EFFECT_TIMEOUT_SECONDS", 0.05)
        form = _form()
        request = _create(form, _with_fault_target(_cell_target()))
        reported: list[SoakActionEvidence] = []

        with pytest.raises(TimeoutError):
            await form.execute(request, report_applied=reported.append)
        assert reported == []

    async def test_a_request_whose_fault_target_is_not_its_target_is_refused_before_posting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request addressing another incarnation must never reach the fault hook."""
        api = _FakeCellApi([])
        _patch_http(monkeypatch, api)
        form = _form()
        request = _create(form, _with_fault_target(_cell_target()))
        forged = request.model_copy(update={"target": _cell_target(incarnation="inc-b")})

        with pytest.raises(AssertionError):
            await _execute(form, forged)
        assert api.hook_posts == []
