import random
import threading
from unittest.mock import MagicMock

from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    SERVING,
    StubFaultForm,
    api_server_fault_forms,
    cell,
    fixed_fault_forms,
    intervals,
    mock_response,
    patched_requests,
    staged,
    typed_cell,
)


def test_loop_never_kills_the_last_live_cell_under_stale_liveness() -> None:
    """Regression: a perpetually-stale 'all healthy' view yields at most one kill (2 cells)."""
    cell_names = ["actor-0", "actor-1"]
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        # Worst case: the injected cell's death is never detected (every cell always Healthy).
        return mock_response({"items": [cell(n, healthy=True) for n in cell_names]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-6),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    assert len(injected) == 1, f"expected at most one injection, got {injected}"


def test_loop_injects_again_after_an_injected_cell_recovers() -> None:
    """Polling tracks a cell's down->up cycle between injections, so a second injection follows."""
    cell_names = ["actor-0", "actor-1"]
    injected: list[str] = []
    stop_event = threading.Event()
    down = {"name": None, "polls_left": 0}
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if len(injected) >= 2 or polls["n"] >= 100:
            stop_event.set()
        items = [cell(n, healthy=not (down["name"] == n and down["polls_left"] > 0)) for n in cell_names]
        if down["polls_left"] > 0:
            down["polls_left"] -= 1
        return mock_response({"items": items})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        name = url.rsplit("/cells/", 1)[1].split("/")[0]
        injected.append(name)
        down["name"], down["polls_left"] = name, 3  # crashed cell reads unhealthy for a few polls, then heals
        return mock_response({})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-6),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    assert len(injected) >= 2, f"expected a second injection after recovery, got {injected}"


def _run_typed_injection_loop(cells: list[dict], *, cell_types: tuple[str, ...]) -> list[str]:
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(cell_types, 1e-6),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

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

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-9),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

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
    def test_list_cells_failure_is_retried_without_recording_recovery(self) -> None:
        """A transient outage after injection must preserve pending recovery debt and retry."""
        cells = [staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)]
        log = state.EventLog()
        injected: list[str] = []
        debt_around_failure: list[dict[str, int]] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] in {2, 3}:
                debt_around_failure.append(
                    views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout")
                )
            if polls["n"] == 2:
                raise RuntimeError("api server unreachable")
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            return mock_response({})

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            mock_requests.post.side_effect = fake_post
            core.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                event_log=log,
                cell_fault_forms=api_server_fault_forms(),
                poll_interval_seconds=1e-6,
            )

        assert len(injected) == 1, injected
        expected_debt: dict[str, int] = {injected[0]: 1}
        assert debt_around_failure == [expected_debt, expected_debt]
        assert views.compute_states_of_cell_name(log.events) == {
            "rollout-engine-0": [SERVING],
            "rollout-engine-1": [SERVING],
        }
        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
        assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == expected_debt

    def test_failed_fault_post_is_not_counted_and_is_retried(self) -> None:
        """A rejected inject-fault call must leave the soak free to try again, and must not inflate the tally."""
        cells = [staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)]
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

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            mock_requests.post.side_effect = fake_post
            core.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-6),
                stop_event=stop_event,
                event_log=log,
                cell_fault_forms=api_server_fault_forms(),
                poll_interval_seconds=1e-6,
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

        assert drawn == [fault_forms.DELETE_POD_FORM_NAME, fault_forms.DELETE_POD_FORM_NAME], drawn
        mock_requests.post.assert_not_called()


def test_the_loop_draws_a_form_that_has_never_worked_before_repeating_a_proven_one() -> None:
    """Uniform sampling can leave the rarest fault untried for a whole soak, which is the one worth trying."""
    drawn: list[str] = []
    log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
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


class TestRolloutSpareReadiness:
    def test_a_healthy_engine_that_is_not_in_the_router_is_not_a_spare(self) -> None:
        """Regression: a relaunched engine reads Healthy long before it can answer, so it is no replacement."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_two_serving_engines_still_leave_one_of_them_injectable(self) -> None:
        """The readiness rule must not block the case it was never meant to block."""
        injected = _run_typed_injection_loop(
            [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")],
            cell_types=("rollout",),
        )

        assert injected

    def test_a_trainer_cell_is_judged_by_liveness_alone(self) -> None:
        """Trainer cells carry no Serving condition, so requiring one would stop every trainer soak."""
        assert core._cell_can_serve(typed_cell("actor-0", "actor"))

    def test_an_engine_that_cannot_serve_yet_is_still_injectable(self) -> None:
        """Crashing an engine mid-relaunch is a real fault window, and only the replica count needs it to serve."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout"),
                typed_cell("rollout-engine-2", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected
