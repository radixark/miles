import random
import threading
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from tests.fast.utils.soak.utils import (
    StubFaultForm,
    api_server_fault_forms,
    cell,
    fixed_fault_forms,
    intervals,
    mock_response,
    patched_requests,
    typed_cell,
)
from tests.utils.soak import core, fault_forms, state, views


@pytest.mark.parametrize("fails", [False, True])
def test_requests_are_recorded_before_execution_and_failures_keep_the_same_identity(fails: bool) -> None:
    """A lost response keeps the selected target and cannot become a successful injection."""
    log = state.EventLog()
    target = typed_cell("actor-0", "actor")
    log.observe([target])

    def inject(cell: dict, rng: random.Random) -> None:
        requests = [event for event in log.events if isinstance(event, state.SoakActionRequestedEvent)]
        assert len(requests) == 1
        assert requests[0].request.target == target
        assert not [event for event in log.events if isinstance(event, state.SoakActionResultEvent)]
        cell["metadata"]["name"] = "mutated"
        if fails:
            raise RuntimeError("response lost")

    core._execute_action(
        action=state.SoakActionRequest(target=target, form_name="fault", harms_cell=True),
        forms={"actor": [StubFaultForm("fault", inject)]},
        rng=random.Random(0),
        event_log=log,
    )

    requests = [event for event in log.events if isinstance(event, state.SoakActionRequestedEvent)]
    results = [event for event in log.events if isinstance(event, state.SoakActionResultEvent)]
    assert len(results) == 1
    assert results[0].request_id == requests[0].request.request_id
    assert requests[0].request.target["metadata"]["name"] == "actor-0"
    assert results[0].returned is not fails
    assert results[0].error == ("RuntimeError('response lost')" if fails else None)
    assert not [event for event in log.events if isinstance(event, state.InjectionEvent)]
    assert views.compute_num_injections(log.events) == int(not fails)
    assert views.compute_forms_drawn_without_success(log.events) == ([("actor", "fault")] if fails else [])


def test_recorded_deadlines_survive_rebuilding_the_scheduler() -> None:
    """A recorded future deadline is not redrawn by polling or recreating the scheduler."""
    log = state.EventLog()
    cells = [typed_cell(f"actor-{i}", "actor") for i in range(2)]
    log.observe(cells)
    cells.clear()
    log.note_schedule(state.SoakScheduleEvent(due_of_type={"actor": 10.0}))
    forms = {"actor": [StubFaultForm("fault", _do_nothing)]}
    scheduler = core.SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1.0}, forms=forms)

    assert scheduler.choose(events=log.events, now=9.0) is None
    with patch.object(core, "_compute_next_injection_time", return_value=20.0):
        request = scheduler.choose(events=log.events, now=10.0)
    assert request is not None
    assert request.next_due_at == 20.0
    log.note_action_requested(request)

    rebuilt = core.SoakActionScheduler(rng=random.Random(100), mean_intervals={"actor": 1.0}, forms=forms)
    assert rebuilt.choose(events=log.events, now=19.0) is None
    assert rebuilt.choose(events=log.events, now=20.0) is not None


def _run_injection_loop(
    *,
    fake_get,
    fake_post=None,
    cell_types: tuple[str, ...] = ("actor", "rollout"),
    event_log: state.EventLog | None = None,
    cell_fault_forms: fault_forms.CellFaultForms | None = None,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    stop_event: threading.Event,
) -> None:
    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        if fake_post is not None:
            mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(cell_types, 1e-9),
            stop_event=stop_event,
            event_log=event_log or state.EventLog(),
            cell_fault_forms=cell_fault_forms or api_server_fault_forms(),
            get_virtual_cells=get_virtual_cells,
            injection_enabled=injection_enabled,
            poll_interval_seconds=1e-6,
        )


def test_virtual_cells_use_the_regular_targeted_injection_path() -> None:
    """Synthetic replicas satisfy the ordinary scheduler without a real FT cell."""
    injected: list[str] = []
    stop_event = threading.Event()
    virtual_cells = [
        typed_cell("virtual-0", "virtual"),
        typed_cell("virtual-1", "virtual"),
    ]

    def inject(target: dict, _rng: random.Random) -> None:
        injected.append(target["metadata"]["name"])
        stop_event.set()

    def fake_get(url: str, timeout: float) -> MagicMock:
        return mock_response({"items": []})

    _run_injection_loop(
        fake_get=fake_get,
        cell_types=("virtual",),
        cell_fault_forms={"virtual": [StubFaultForm("virtual-fault", inject)]},
        get_virtual_cells=lambda: virtual_cells,
        stop_event=stop_event,
    )

    assert len(injected) == 1
    assert injected[0] in {"virtual-0", "virtual-1"}


def test_disabled_injection_still_observes_cells_without_injecting() -> None:
    """A closing scenario keeps recovery evidence while admitting no new fault."""
    injected: list[str] = []
    event_log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 3:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url)
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        event_log=event_log,
        injection_enabled=lambda: False,
        stop_event=stop_event,
    )

    assert injected == []
    assert event_log.events


def _run_typed_injection_loop(cells: list[dict], *, cell_types: tuple[str, ...], num_polls: int = 8) -> list[str]:
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= num_polls:
            stop_event.set()
        return mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=cell_types, stop_event=stop_event)

    return injected


def test_a_stop_that_arrives_while_listing_buys_no_further_injection() -> None:
    """A fault injected on the way out is one nothing is left polling to see recover."""
    injected: list[str] = []
    stop_event = threading.Event()

    def fake_get(url: str, timeout: float) -> MagicMock:
        stop_event.set()
        return mock_response({"items": [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url)
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, stop_event=stop_event)

    assert injected == []


def test_injection_can_be_restricted_to_one_kind_of_cell() -> None:
    """Rollout and trainer cells share one api server, so a run targets one kind at a time."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected
    assert all(name.startswith("rollout-") for name in injected), injected


def test_the_live_replica_count_only_considers_the_targeted_kind() -> None:
    """A single rollout cell must not be killed just because trainer cells are also alive."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected == []


def test_a_mixed_run_sees_every_targeted_kind() -> None:
    """A mixed-ft soak schedules both kinds, and must be able to crash either one."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert injected


def test_a_mixed_run_still_keeps_one_replica_of_each_kind() -> None:
    """Counting kinds together would let the trainer cells license killing the last engine."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert all(name.startswith("actor-") for name in injected), injected


class TestFaultInjectionLoopErrorHandling:
    def test_list_cells_failure_is_retried_and_does_not_stop_the_loop(self) -> None:
        """A transient api-server outage must cost one poll, not the rest of the soak."""
        cells = [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]
        log = state.EventLog()
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] == 1:
                raise RuntimeError("api server unreachable")
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("actor",),
            event_log=log,
            stop_event=stop_event,
        )

        assert injected, injected
        assert views.compute_num_injections(log.events, cell_type="actor") == len(injected)

    def test_failed_fault_post_is_not_counted_and_is_retried(self) -> None:
        """A rejected inject-fault call must leave the soak free to try again, and must not inflate the tally."""
        cells = [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")]
        log = state.EventLog()
        attempts: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 5:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            attempts.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            if len(attempts) == 1:
                raise RuntimeError("inject-fault refused")
            stop_event.set()
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("rollout",),
            event_log=log,
            stop_event=stop_event,
        )

        assert len(attempts) == 2, attempts
        assert views.compute_num_injections(log.events, cell_type="rollout") == 1


class TestMixedInjectionSelection:
    def test_mixed_run_injects_rollout_when_only_rollout_has_a_spare(self) -> None:
        """The mirror of the trainer case: mixed selection must not be hard-coded to actor cells."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected
        assert all(name.startswith("rollout-engine-") for name in injected), injected


def test_the_loop_injects_through_the_forms_of_the_cell_it_picked() -> None:
    """A pod deletion drawn by the loop must reach kubectl, not the api server's inject-fault route."""
    drawn: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=fixed_fault_forms(
                [
                    StubFaultForm(
                        fault_forms.DELETE_POD_FORM_NAME,
                        lambda cell, rng: drawn.append(fault_forms.DELETE_POD_FORM_NAME),
                    )
                ]
            ),
            poll_interval_seconds=1e-6,
        )

        assert drawn, drawn
        assert set(drawn) == {fault_forms.DELETE_POD_FORM_NAME}, drawn
        mock_requests.post.assert_not_called()


def test_the_loop_draws_a_form_that_has_never_worked_before_repeating_a_proven_one() -> None:
    """Uniform sampling can leave the rarest fault untried for a whole soak, which is the one worth trying."""
    drawn: list[str] = []
    log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 10:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(4)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [StubFaultForm(name, lambda cell, rng, n=name: drawn.append(n)) for name in ("a", "b", "c")]
            ),
            poll_interval_seconds=1e-6,
        )

    assert set(drawn[:3]) == {"a", "b", "c"}, drawn


def test_a_form_that_always_refuses_keeps_being_drawn_so_the_soak_can_see_it() -> None:
    """A form that rides on the ones that did work would end the run green while never having fired."""
    log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 8:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [StubFaultForm("works", _do_nothing), StubFaultForm("broken", _always_refuse)]
            ),
            poll_interval_seconds=1e-6,
        )

    assert views.compute_forms_drawn_without_success(log.events) == [("actor", "broken")]


def _always_refuse(cell: dict, rng: random.Random) -> None:
    raise RuntimeError("this form never works")


def _do_nothing(cell: dict, rng: random.Random) -> None:
    return None


@pytest.mark.parametrize("healthy,serving", [(True, True), (False, True), (True, False), (None, False)])
def test_injection_does_not_wait_for_health_or_serving(healthy: bool | None, serving: bool) -> None:
    """Weight updates and recovery do not postpone a due fault."""
    cells = [typed_cell(f"rollout-engine-{i}", "rollout", serving=serving) for i in range(2)]
    for item in cells:
        for condition in item["status"]["conditions"]:
            if condition["type"] == "Healthy":
                condition["status"] = "Unknown" if healthy is None else str(healthy)

    injected = _run_typed_injection_loop(cells, cell_types=("rollout",))

    assert injected


def test_an_unrecovered_cell_does_not_block_another_fault() -> None:
    """A second fault may overlap the recovery of the first."""
    injected = _run_typed_injection_loop(
        [typed_cell(f"actor-{i}", "actor", healthy=False) for i in range(3)],
        cell_types=("actor",),
    )

    assert len(injected) >= 2


@pytest.mark.parametrize("fails", [False, True])
def test_each_attempt_draws_a_new_deadline(fails: bool) -> None:
    """A failed response does not leave an overdue fault retrying every poll."""
    stop_event = threading.Event()
    attempts: list[str] = []
    polls = 0

    def fake_get(url: str, timeout: float) -> MagicMock:
        nonlocal polls
        polls += 1
        if polls == 5:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(2)]})

    def inject(target: dict, rng: random.Random) -> None:
        attempts.append(target["metadata"]["name"])
        if fails:
            raise RuntimeError("response lost")

    with patch.object(core, "_compute_next_injection_time", side_effect=[0.0, float("inf")]):
        _run_injection_loop(
            fake_get=fake_get,
            cell_types=("actor",),
            cell_fault_forms={"actor": [StubFaultForm("fault", inject)]},
            stop_event=stop_event,
        )

    assert len(attempts) == 1
