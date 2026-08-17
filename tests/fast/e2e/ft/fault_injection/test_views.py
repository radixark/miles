from tests.e2e.ft.conftest_ft.fault_injection import state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    PENDING,
    RUNNING_NOT_SERVING,
    SERVING,
    SUSPENDED,
    cell,
    log_of,
    names,
    note_injected,
    staged,
)


def test_a_fresh_log_counts_every_healthy_cell_as_alive() -> None:
    """With no outstanding injection the live set is just the healthy cells."""
    log = state.EventLog()
    cells = [cell("c0", healthy=True), cell("c1", healthy=False)]
    log.observe(cells)
    assert names(views.compute_genuinely_alive(log.events, cells)) == {"c0"}


def test_injected_cell_is_excluded_while_its_crash_is_still_undetected() -> None:
    """The api server's stale 'still healthy' view must not count a just-killed cell."""
    log = state.EventLog()
    cells = [cell("c0", healthy=True), cell("c1", healthy=True)]
    note_injected(log, "c0")
    log.observe(cells)  # c0 really dead but still reported Healthy
    assert names(views.compute_genuinely_alive(log.events, cells)) == {"c1"}


def test_injected_cell_counts_again_only_after_a_full_down_then_up_cycle() -> None:
    """A cell must be seen unhealthy and then healthy again before it rejoins the live set."""
    log = state.EventLog()
    healthy = [cell("c0", healthy=True), cell("c1", healthy=True)]
    down = [cell("c0", healthy=False), cell("c1", healthy=True)]
    note_injected(log, "c0")

    log.observe(healthy)  # stale-alive
    assert names(views.compute_genuinely_alive(log.events, healthy)) == {"c1"}
    log.observe(down)  # detected down
    assert names(views.compute_genuinely_alive(log.events, down)) == {"c1"}
    log.observe(healthy)  # healed
    assert names(views.compute_genuinely_alive(log.events, healthy)) == {"c0", "c1"}


def test_vanished_cell_counts_as_the_down_half_of_the_cycle() -> None:
    """A cell missing from the snapshot is treated as observed-down, then recovers when back."""
    log = state.EventLog()
    note_injected(log, "c0")
    log.observe([cell("c1", healthy=True)])  # c0 absent == down
    healthy = [cell("c0", healthy=True), cell("c1", healthy=True)]
    log.observe(healthy)
    assert names(views.compute_genuinely_alive(log.events, healthy)) == {"c0", "c1"}


def test_allows_overlapping_crashes_while_one_cell_stays_alive() -> None:
    """The live set guards >=1 live replica, not 1-crash-at-a-time: with 3 cells two may be down."""
    log = state.EventLog()
    cells = [cell("c0", healthy=True), cell("c1", healthy=True), cell("c2", healthy=True)]

    note_injected(log, "c0")
    log.observe(cells)
    assert names(views.compute_genuinely_alive(log.events, cells)) == {
        "c1",
        "c2",
    }  # 2 still alive -> a 2nd inject is allowed

    note_injected(log, "c1")
    log.observe(cells)
    assert names(views.compute_genuinely_alive(log.events, cells)) == {"c2"}  # now only 1 -> loop would skip


def test_observed_states_record_only_transitions() -> None:
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    log = log_of([SERVING, SERVING, SUSPENDED, SUSPENDED, SERVING])

    assert views.compute_states_of_cell_name(log.events) == {"rollout-engine-0": [SERVING, SUSPENDED, SERVING]}


def test_a_colocated_cell_recovers_through_a_gated_relaunch() -> None:
    """A relaunched engine stays gated until the next weight update window puts it back in the router."""
    log = log_of([SERVING, SUSPENDED, PENDING, RUNNING_NOT_SERVING, SERVING], inject_before={1: 1})

    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 1
    assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {}


def test_a_missed_suspended_sample_still_counts_as_a_recovery() -> None:
    """Suspension lasts only the resume delay, so a 2s poll can miss it entirely."""
    log = log_of([SERVING, PENDING, SERVING], inject_before={1: 1})

    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 1


def test_a_replacement_that_never_reaches_the_router_is_not_a_recovery() -> None:
    """Regression: a relaunched engine stuck at PendingWeights also reads Running, and must not pass."""
    log = log_of([SERVING, PENDING, RUNNING_NOT_SERVING], inject_before={1: 1})

    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
    assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {"rollout-engine-0": 1}


def test_a_cell_that_was_never_injected_witnesses_no_recovery() -> None:
    """Otherwise a run that injected nothing would still pass the gated assertion."""
    log = log_of([PENDING, SERVING])

    assert views.compute_num_injections(log.events, cell_type="rollout") == 0
    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0


def test_skipping_the_relaunch_phase_is_not_a_recovery() -> None:
    """A cell that never left Running was never replaced, so it witnesses no healing."""
    log = log_of([SERVING, RUNNING_NOT_SERVING, SERVING], inject_before={1: 1})

    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0


def test_each_accepted_injection_needs_its_own_completed_recovery() -> None:
    """Regression: a second crash accepted just before the run ends must not ride on the first heal."""
    log = log_of([SERVING, PENDING, SERVING, SERVING], inject_before={1: 1, 3: 1})

    assert views.compute_num_injections(log.events, cell_type="rollout") == 2
    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 1
    assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {"rollout-engine-0": 1}


def test_recoveries_of_another_cell_kind_do_not_count() -> None:
    """A mixed soak injects both kinds, and the rollout view must only see rollout cells."""
    log = state.EventLog()
    log.observe([staged("actor-0", SERVING, cell_type="actor")])
    note_injected(log, "actor-0")
    for cell_state in [PENDING, SERVING]:
        log.observe([staged("actor-0", cell_state, cell_type="actor")])

    assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
    assert views.compute_num_completed_recoveries(log.events, cell_type="actor") == 1


class TestRecoveryPairing:
    def test_another_cells_relaunch_cannot_complete_the_injected_cells_recovery(self) -> None:
        """A sibling engine's relaunch-and-serve cycle must not discharge the injected cell's debt."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)])
        note_injected(log, "rollout-engine-0")
        for sibling_state in [PENDING, SERVING]:
            log.observe([staged("rollout-engine-0", SERVING), staged("rollout-engine-1", sibling_state)])

        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
        assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {"rollout-engine-0": 1}

    def test_relaunch_observed_before_injection_does_not_count_as_recovery(self) -> None:
        """The cycle must be ordered injection then relaunch then serving, not merely present in the history."""
        log = log_of([SERVING, PENDING, SERVING], inject_before={2: 1})

        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
        assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {"rollout-engine-0": 1}


class TestOverlappingRecoveries:
    def test_a_cell_crashed_again_mid_relaunch_is_repaid_by_one_final_serve(self) -> None:
        """Regression: a dense soak crashes an engine before it re-serves, and pairing one-for-one went red."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING)])
        _note_injection(log)
        log.observe([staged("rollout-engine-0", PENDING)])
        log.observe([staged("rollout-engine-0", RUNNING_NOT_SERVING)])
        _note_injection(log)
        log.observe([staged("rollout-engine-0", PENDING)])
        log.observe([staged("rollout-engine-0", SERVING)])

        assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {}
        assert views.compute_num_completed_recoveries(log.events, cell_type="rollout") == 2

    def test_a_serve_that_predates_the_last_crash_does_not_discharge_it(self) -> None:
        """Otherwise the last crash of a soak is paid for by the recovery of the crash before it."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING)])
        _note_injection(log)
        log.observe([staged("rollout-engine-0", PENDING)])
        log.observe([staged("rollout-engine-0", SERVING)])
        _note_injection(log)

        assert views.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {"rollout-engine-0": 1}


def _note_injection(log: state.EventLog, *, cell_name: str = "rollout-engine-0") -> None:
    log.note_injection_attempt(cell_name=cell_name, form_name="inject_fault:sigkill", succeeded=True)
