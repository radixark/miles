import random
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    StubFaultForm,
    api_server_fault_forms,
    cell,
    fixed_fault_forms,
    intervals,
    mock_response,
    patched_requests,
    typed_cell,
)


def _run_injection_loop(
    *,
    fake_get,
    fake_post=None,
    cell_types: tuple[str, ...] = ("actor", "rollout"),
    quiescent_polls_required: int = 1,
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
            quiescent_polls_required=quiescent_polls_required,
        )


def test_no_injection_before_the_quiescence_streak_is_long_enough() -> None:
    """The first reads after a disturbance can all be stale, so a short all-serving streak buys no kill."""
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 4:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, quiescent_polls_required=10, stop_event=stop_event)

    assert injected == []


def test_successive_injections_are_spaced_by_the_quiescence_gate() -> None:
    """An injection resets the streak, so the next kill of that kind waits out the gate again."""
    injection_polls: list[int] = []
    stop_event = threading.Event()
    required = 3
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 14:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injection_polls.append(polls["n"])
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=required,
        stop_event=stop_event,
    )

    assert len(injection_polls) >= 2, injection_polls
    gaps = [after - before for before, after in zip(injection_polls, injection_polls[1:], strict=False)]
    assert all(gap >= required for gap in gaps), injection_polls


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


def test_a_kind_with_a_dead_replica_is_not_quiescent() -> None:
    """A kill must be followed by an observed full recovery streak before that kind is due again."""
    injection_polls: list[int] = []
    stop_event = threading.Event()
    down = {"name": None, "polls_left": 0}
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if len(injection_polls) >= 2 or polls["n"] >= 100:
            stop_event.set()
        items = [cell(n, healthy=not (down["name"] == n and down["polls_left"] > 0)) for n in ("actor-0", "actor-1")]
        if down["polls_left"] > 0:
            down["polls_left"] -= 1
        return mock_response({"items": items})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injection_polls.append(polls["n"])
        down["name"], down["polls_left"] = url.rsplit("/cells/", 1)[1].split("/")[0], 3
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=2,
        stop_event=stop_event,
    )

    assert len(injection_polls) >= 2, injection_polls
    assert injection_polls[1] - injection_polls[0] >= 3 + 2, injection_polls


def test_a_vanished_replica_blocks_its_kind_even_when_the_survivors_serve() -> None:
    """A killed pod can disappear from the listing entirely, which must read as still recovering."""
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}
    all_names = ("actor-0", "actor-1", "actor-2")

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 20:
            stop_event.set()
        names = [n for n in all_names if n not in injected] if injected else list(all_names)
        return mock_response({"items": [cell(n, healthy=True) for n in names]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=2,
        stop_event=stop_event,
    )

    assert len(injected) == 1, injected


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


def test_a_failed_injection_forfeits_the_quiescence_streak() -> None:
    """A lost response does not prove a lost kill, so the next attempt must wait out the gate again."""
    attempt_polls: list[int] = []
    stop_event = threading.Event()
    required = 3
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 14:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        attempt_polls.append(polls["n"])
        if len(attempt_polls) == 1:
            raise RuntimeError("response lost after the kill may have landed")
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=required,
        stop_event=stop_event,
    )

    assert len(attempt_polls) >= 2, attempt_polls
    assert attempt_polls[1] - attempt_polls[0] >= required, attempt_polls


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
            quiescent_polls_required=1,
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
            quiescent_polls_required=1,
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
            quiescent_polls_required=1,
        )

    assert views.compute_forms_drawn_but_never_successful(log.events) == [("actor", "broken")]


def _always_refuse(cell: dict, rng: random.Random) -> None:
    raise RuntimeError("this form never works")


def _do_nothing(cell: dict, rng: random.Random) -> None:
    return None


class TestRolloutQuiescence:
    def test_an_engine_that_is_not_in_the_router_blocks_its_kind(self) -> None:
        """A relaunched engine reads Healthy long before it can answer, so its kind is still recovering."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_two_serving_engines_still_leave_one_of_them_injectable(self) -> None:
        """The quiescence rule must not block the case it was never meant to block."""
        injected = _run_typed_injection_loop(
            [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")],
            cell_types=("rollout",),
        )

        assert injected

    def test_a_trainer_cell_is_judged_by_liveness_alone(self) -> None:
        """Trainer cells carry no Serving condition, so requiring one would stop every trainer soak."""
        assert core._cell_can_serve(typed_cell("actor-0", "actor"))
