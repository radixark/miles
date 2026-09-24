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
    _requested,
    _with_fault_target,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions import inject_fault as inject_fault_module
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, ObservedCellFaultKind

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.test_utils.fault_injector.actions.remote import ApiServerFaultAction
from miles.utils.test_utils.fault_injector.controller import FaultHookOperation
from miles.utils.test_utils.fault_injector.models import FaultHookName
from miles.utils.workers.naming import compute_cell_id

_BASE_URL = "http://api:18080"
_ACTOR_0 = compute_cell_id(pool_id="actor", cell_index=0)
_ROLLOUT_0 = compute_cell_id(pool_id="rollout", cell_index=0)
_HOOK = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND


def _form(**kwargs: object) -> InjectFaultForm:
    return InjectFaultForm(base_url=_BASE_URL, action=KillProcessAction(), **kwargs)


def _create(
    form: InjectFaultForm,
    target: CellTarget,
    *,
    trainers: tuple[CellTarget, ...] | list[CellTarget] = (),
    events: tuple[SoakEvent, ...] | list[SoakEvent] = (),
) -> SoakActionRequest | None:
    return form.maybe_create_request(
        target=target,
        observation=_observation([target, *trainers], at=_at(0)),
        events=list(events),
        rng=random.Random(0),
    )


async def _execute(form: InjectFaultForm, request: SoakActionRequest) -> list[SoakActionEvidence]:
    reported: list[SoakActionEvidence] = []
    await form.execute(request, report_applied=reported.append)
    return reported


class TestInjectFaultFormName:
    def test_the_name_encodes_action_hook_delay_and_trainer_route(self) -> None:
        """Forms differing in hook, delay or route get distinct names so find_form can tell them apart."""
        assert _form().name == "inject_fault:kill_process"
        assert _form(hook_name=_HOOK, max_delay_ms=1000).name == f"inject_fault:kill_process:{_HOOK.value}:1000ms"
        assert (
            _form(hook_name=_HOOK, max_delay_ms=0, through_trainer_hook=True).name
            == f"inject_fault:kill_process:{_HOOK.value}:0ms:through_trainer"
        )

    def test_a_trainer_routed_form_also_needs_trainer_fault_targets(self) -> None:
        """Routing through a trainer hook requires observing trainer fault targets too."""
        assert _form().fault_target_cell_types("rollout") == frozenset({"rollout"})
        assert _form(through_trainer_hook=True).fault_target_cell_types("rollout") == frozenset({"rollout", "actor"})


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

    def test_a_direct_request_hooks_the_fault_target_itself(self) -> None:
        """Without trainer routing the hook and fault targets are the same worker."""
        target = _with_fault_target(_cell_target())

        request = _create(_form(hook_name=_HOOK, max_delay_ms=500), target)

        assert (request.target, request.form_name) == (target, _form(hook_name=_HOOK, max_delay_ms=500).name)
        assert isinstance(request.details, InjectFaultDetails)
        assert request.details.fault_target == request.details.hook_target == target.fault_target
        assert request.details.hook_name == _HOOK
        assert 0 <= request.details.delay_ms <= 500

    def test_a_form_without_delay_never_delays(self) -> None:
        """A zero maximum delay yields an immediate fault."""
        assert _create(_form(), _with_fault_target(_cell_target())).details.delay_ms == 0

    def test_a_trainer_routed_request_hooks_a_live_untouched_trainer(self) -> None:
        """The rollout fault fires from a live trainer that no earlier request has harmed."""
        rollout = _with_fault_target(_cell_target(kind="rollout"))
        harmed = _with_fault_target(_cell_target(cell_index=0))
        dead = _with_fault_target(_cell_target(cell_index=1, alive=False))
        untouched = _with_fault_target(_cell_target(cell_index=2))
        earlier = SoakActionRequest(
            target=harmed,
            form_name="x",
            details=InjectFaultDetails(fault_target=harmed.fault_target, hook_target=harmed.fault_target),
        )

        request = _create(
            _form(through_trainer_hook=True),
            rollout,
            trainers=[harmed, dead, untouched],
            events=[_requested(earlier, at=_at(0))],
        )

        assert request.details.fault_target == rollout.fault_target
        assert request.details.hook_target == untouched.fault_target

    def test_a_replaced_trainer_is_eligible_again(self) -> None:
        """A harmed trainer's new incarnation is untouched and may carry the hook."""
        rollout = _with_fault_target(_cell_target(kind="rollout"))
        old = _with_fault_target(_cell_target(incarnation="old"))
        new = _with_fault_target(_cell_target(incarnation="new"))
        earlier = SoakActionRequest(
            target=old,
            form_name="x",
            details=InjectFaultDetails(fault_target=old.fault_target, hook_target=old.fault_target),
        )

        request = _create(
            _form(through_trainer_hook=True), rollout, trainers=[new], events=[_requested(earlier, at=_at(0))]
        )

        assert request.details.hook_target == new.fault_target

    def test_a_trainer_routed_request_without_a_candidate_is_declined(self) -> None:
        """No live trainer with a current fault target means no request."""
        rollout = _with_fault_target(_cell_target(kind="rollout"))
        stale = _cell_target(incarnation="new").model_copy(
            update={"fault_target": _fault_target(_ACTOR_0, workers_hash="old")}
        )

        assert _create(_form(through_trainer_hook=True), rollout, trainers=[stale]) is None


class TestInjectFaultFormExecute:
    async def test_the_fault_is_set_on_the_target_and_a_missing_cell_is_the_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SET command naming this request is posted and a vanished cell is reported as applied."""
        api = _FakeCellApi([])
        _patch_http(monkeypatch, api)
        form = _form(hook_name=_HOOK, lifetime_seconds=60.0)
        request = _create(form, _with_fault_target(_cell_target()))

        [evidence] = await _execute(form, request)

        [(cell_id, command)] = api.hook_posts
        assert cell_id == _ACTOR_0
        assert command.operation is FaultHookOperation.SET
        assert command.request.request_id == request.request_id
        assert command.request.action == KillProcessAction()
        assert command.request.target == request.details.hook_target
        assert (command.request.hook_name, command.request.lifetime_seconds) == (_HOOK, 60.0)
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

    async def test_a_trainer_routed_fault_is_posted_to_the_trainer_and_watched_on_the_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trainer hook forwards the action to the rollout worker whose disappearance is the effect."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor")])
        _patch_http(monkeypatch, api)
        form = _form(through_trainer_hook=True)
        rollout = _with_fault_target(_cell_target(kind="rollout"))
        request = _create(form, rollout, trainers=[_with_fault_target(_cell_target())])

        [evidence] = await _execute(form, request)

        [(cell_id, command)] = api.hook_posts
        assert cell_id == _ACTOR_0
        assert command.request.action == ApiServerFaultAction(
            base_url=_BASE_URL, cell_id=_ROLLOUT_0, rank=0, inner=KillProcessAction()
        )
        assert evidence.target == rollout.fault_target
        assert evidence.observed is ObservedCellFaultKind.MISSING


# ============================ hook triggers ============================


class TestHookTriggeredRequests:
    def test_the_drawn_delay_is_reproducible_from_the_seed_and_bounded(self) -> None:
        """A soak replayed with its seed must draw the same delays, each within the form's maximum."""
        form = _form(hook_name=_HOOK, max_delay_ms=1000)
        target = _with_fault_target(_cell_target())

        def _delays(seed: int) -> list[float]:
            rng = random.Random(seed)
            observation = _observation([target], at=_at(0))
            return [
                form.maybe_create_request(target=target, observation=observation, events=[], rng=rng).details.delay_ms
                for _ in range(5)
            ]

        assert _delays(3) == _delays(3)
        assert all(0 <= delay <= 1000 for delay in _delays(3))

    async def test_the_drawn_delay_and_hook_reach_the_set_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The worker enforces the delay it receives, so the drawn one must be the one posted."""
        api = _FakeCellApi([])
        _patch_http(monkeypatch, api)
        form = _form(hook_name=_HOOK, max_delay_ms=1000)
        request = _create(form, _with_fault_target(_cell_target()))

        await _execute(form, request)

        [(_cell_id, command)] = api.hook_posts
        assert command.request.delay_ms == request.details.delay_ms > 0
        assert command.request.hook_name == _HOOK

    def test_a_trainer_target_never_carries_its_own_routed_hook(self) -> None:
        """Routing a trainer's fault through itself would make the victim fire its own fault."""
        victim = _with_fault_target(_cell_target(cell_index=0))
        other = _with_fault_target(_cell_target(cell_index=1))
        form = _form(through_trainer_hook=True)

        assert _create(form, victim, trainers=[other]).details.hook_target == other.fault_target
        assert _create(form, victim) is None

    def test_the_routing_trainer_is_drawn_from_the_seed(self) -> None:
        """The same seed must pick the same trainer among several eligible ones."""
        rollout = _with_fault_target(_cell_target(kind="rollout"))
        trainers = [_with_fault_target(_cell_target(cell_index=index)) for index in range(4)]
        observation = _observation([rollout, *trainers], at=_at(0))
        form = _form(through_trainer_hook=True)

        def _pick(seed: int) -> object:
            return form.maybe_create_request(
                target=rollout, observation=observation, events=[], rng=random.Random(seed)
            ).details.hook_target

        assert _pick(11) == _pick(11)
        assert _pick(11) in [trainer.fault_target for trainer in trainers]
