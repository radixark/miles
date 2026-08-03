import dataclasses
import threading
from unittest.mock import MagicMock, patch

from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.e2e.ft.conftest_ft.fault_injection import RecoveryGate, cell_is_alive
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode


def _cell(name: str, *, healthy: bool, cell_type: str = "actor", phase: str = "Running") -> dict:
    status = "True" if healthy else "False"
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": cell_type}},
        "status": {"phase": phase, "conditions": [{"type": "Healthy", "status": status}]},
    }


def _by_name(*cells: dict) -> dict[str, dict]:
    return {c["metadata"]["name"]: c for c in cells}


def _names(cells: list[dict]) -> set[str]:
    return {c["metadata"]["name"] for c in cells}


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    """cell_is_alive reflects the Healthy condition status."""
    assert cell_is_alive(_cell("c", healthy=True))
    assert not cell_is_alive(_cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    """A cell with no Healthy condition is not considered alive."""
    assert not cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_fresh_gate_counts_every_healthy_cell_as_alive() -> None:
    """With no outstanding injection the live set is just the healthy cells."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=False)]
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c0"}


def test_injected_cell_is_excluded_while_its_crash_is_still_undetected() -> None:
    """The api server's stale 'still healthy' view must not count a just-killed cell."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    gate.note_injected("c0")
    gate.observe(_by_name(*cells))  # c0 really dead but still reported Healthy
    assert _names(gate.genuinely_alive(cells)) == {"c1"}


def test_injected_cell_counts_again_only_after_a_full_down_then_up_cycle() -> None:
    """A cell must be seen unhealthy and then healthy again before it rejoins the live set."""
    gate = RecoveryGate()
    healthy = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    down = [_cell("c0", healthy=False), _cell("c1", healthy=True)]
    gate.note_injected("c0")

    gate.observe(_by_name(*healthy))  # stale-alive
    assert _names(gate.genuinely_alive(healthy)) == {"c1"}
    gate.observe(_by_name(*down))  # detected down
    assert _names(gate.genuinely_alive(down)) == {"c1"}
    gate.observe(_by_name(*healthy))  # healed
    assert _names(gate.genuinely_alive(healthy)) == {"c0", "c1"}


def test_vanished_cell_counts_as_the_down_half_of_the_cycle() -> None:
    """A cell missing from the snapshot is treated as observed-down, then recovers when back."""
    gate = RecoveryGate()
    gate.note_injected("c0")
    gate.observe(_by_name(_cell("c1", healthy=True)))  # c0 absent == down
    healthy = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    gate.observe(_by_name(*healthy))
    assert _names(gate.genuinely_alive(healthy)) == {"c0", "c1"}


def test_allows_overlapping_crashes_while_one_cell_stays_alive() -> None:
    """The gate guards >=1 live replica, not 1-crash-at-a-time: with 3 cells two may be down."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=True), _cell("c2", healthy=True)]

    gate.note_injected("c0")
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c1", "c2"}  # 2 still alive -> a 2nd inject is allowed

    gate.note_injected("c1")
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c2"}  # now only 1 -> loop would skip


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


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
        return _mock_response({"items": [_cell(n, healthy=True) for n in cell_names]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds=1e-6,
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            cell_type=None,
            phase_history=fi.PhaseHistory(),
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
        items = [_cell(n, healthy=not (down["name"] == n and down["polls_left"] > 0)) for n in cell_names]
        if down["polls_left"] > 0:
            down["polls_left"] -= 1
        return _mock_response({"items": items})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        name = url.rsplit("/cells/", 1)[1].split("/")[0]
        injected.append(name)
        down["name"], down["polls_left"] = name, 3  # crashed cell reads unhealthy for a few polls, then heals
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds=1e-6,
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            cell_type=None,
            phase_history=fi.PhaseHistory(),
            poll_interval_seconds=1e-6,
        )

    assert len(injected) >= 2, f"expected a second injection after recovery, got {injected}"


def _typed_cell(name: str, cell_type: str, *, healthy: bool = True) -> dict:
    return _cell(name, healthy=healthy, cell_type=cell_type)


def _run_typed_injection_loop(cells: list[dict], *, cell_type: str | None) -> list[str]:
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return _mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds=1e-6,
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            cell_type=cell_type,
            phase_history=fi.PhaseHistory(),
            poll_interval_seconds=1e-6,
        )

    return injected


def test_injection_can_be_restricted_to_one_kind_of_cell() -> None:
    """Rollout and trainer cells share one api server, so a run targets one kind at a time."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
            _typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_type="rollout",
    )

    assert injected
    assert all(name.startswith("rollout-") for name in injected), injected


def test_the_live_replica_count_only_considers_the_targeted_kind() -> None:
    """A single rollout cell must not be killed just because trainer cells are also alive."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_type="rollout",
    )

    assert injected == []


def test_an_untyped_run_sees_every_cell() -> None:
    """A mixed-ft soak declares no cell type, and must be able to crash either kind."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
            _typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_type=None,
    )

    assert injected


def test_an_untyped_run_still_keeps_one_replica_of_each_kind() -> None:
    """Counting kinds together would let the trainer cells license killing the last engine."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_type=None,
    )

    assert all(name.startswith("actor-") for name in injected), injected


def _phased(name: str, phase: str, *, cell_type: str = "rollout") -> dict:
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": cell_type}},
        "status": {"phase": phase, "conditions": []},
    }


INJECT: str = "INJECT"


def _replay(script: list[str], *, cell_name: str = "rollout-engine-0", cell_type: str = "rollout") -> fi.PhaseHistory:
    history = fi.PhaseHistory()
    for step in script:
        if step == INJECT:
            history.note_injected(cell_name)
        else:
            history.observe([_phased(cell_name, step, cell_type=cell_type)])
    return history


def test_phase_history_records_only_transitions() -> None:
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    history = _replay(["Running", "Running", "Suspended", "Suspended", "Running"])

    assert history.phases_of_cell_name == {"rollout-engine-0": ["Running", "Suspended", "Running"]}


def test_a_colocated_cell_heals_through_pending() -> None:
    """A relaunched engine stays gated until the next weight update window activates it."""
    history = _replay(["Running", INJECT, "Suspended", "Pending", "Running"])

    assert [outcome.recovered for outcome in history.injection_outcomes()] == [True]


def test_a_missed_suspended_sample_still_counts_as_healed() -> None:
    """Suspension lasts only the resume delay, so a 2s poll can miss it entirely."""
    history = _replay(["Running", INJECT, "Pending", "Running"])

    assert [outcome.recovered for outcome in history.injection_outcomes()] == [True]


def test_a_cell_that_never_crashed_is_not_a_healing_witness() -> None:
    """Otherwise a run that injected nothing would still pass the gated assertion."""
    history = _replay(["Pending", "Running"])

    assert history.injection_outcomes() == []


def test_skipping_the_pending_phase_is_not_a_healing_witness() -> None:
    """A cell that never re-entered Pending was never replaced, so it witnesses no healing."""
    history = _replay(["Running", INJECT, "Suspended", "Running"])

    assert [(outcome.recovered, outcome.still_down) for outcome in history.injection_outcomes()] == [(False, False)]


def test_a_healing_that_predates_the_injection_does_not_pair_with_it() -> None:
    """A recovery from an earlier crash must not vouch for a later injection."""
    history = _replay(["Running", "Pending", "Running", INJECT, "Running"])

    assert [(outcome.recovered, outcome.still_down) for outcome in history.injection_outcomes()] == [(False, False)]


def test_one_healing_cannot_pair_with_two_injections() -> None:
    """The whole point of pairing: N injections need N recoveries, not one shared witness."""
    history = _replay(["Running", INJECT, "Pending", "Running", INJECT, "Running"])

    assert [outcome.recovered for outcome in history.injection_outcomes()] == [True, False]


def test_each_injection_pairs_with_its_own_later_healing() -> None:
    """Two crash->heal cycles on one cell are two witnessed recoveries."""
    history = _replay(["Running", INJECT, "Pending", "Running", INJECT, "Pending", "Running"])

    assert [outcome.recovered for outcome in history.injection_outcomes()] == [True, True]


def test_an_injection_whose_cell_is_still_down_at_the_end_is_flagged_as_unfinished() -> None:
    """Training can finish mid-recovery, which is not evidence that healing is broken."""
    history = _replay(["Running", INJECT, "Pending"])

    assert [(outcome.recovered, outcome.still_down) for outcome in history.injection_outcomes()] == [(False, True)]


def test_injection_outcomes_can_be_restricted_to_one_kind_of_cell() -> None:
    """A mixed soak injects both kinds, but the rollout witness only judges rollout cells."""
    history = fi.PhaseHistory()
    history.observe([_phased("actor-0", "Running", cell_type="actor"), _phased("rollout-engine-0", "Running")])
    history.note_injected("actor-0")
    history.note_injected("rollout-engine-0")

    assert [outcome.cell_name for outcome in history.injection_outcomes(cell_type="rollout")] == ["rollout-engine-0"]


def test_the_loop_records_every_accepted_injection_in_the_phase_history() -> None:
    """Pairing recoveries to injections needs the injections in the same history as the phases."""
    cells = [_typed_cell("rollout-engine-0", "rollout"), _typed_cell("rollout-engine-1", "rollout")]
    phase_history = fi.PhaseHistory()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return _mock_response({"items": cells})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = lambda url, json, timeout: _mock_response({})
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds=1e-6,
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            cell_type="rollout",
            phase_history=phase_history,
            poll_interval_seconds=1e-6,
        )

    outcomes = phase_history.injection_outcomes(cell_type="rollout")
    assert outcomes
    assert all(outcome.cell_name.startswith("rollout-engine-") for outcome in outcomes)


def test_stop_and_join_takes_one_last_snapshot_before_the_history_is_read() -> None:
    """Regression: a recovery completing after the final poll must not be lost to a race."""
    handle = fi.FaultInjectorHandle(base_url="http://control", seed=0, mean_interval_seconds=1e9, cell_type="rollout")

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = lambda url, timeout: _mock_response(
            {"items": [_phased("rollout-engine-0", "Running")]}
        )
        handle.start()
        handle.stop_and_join(timeout_seconds=5)

    assert handle.phase_history.phases_of_cell_name == {"rollout-engine-0": ["Running"]}


def _injector_with_history(history: fi.PhaseHistory) -> fi.FaultInjectorHandle:
    handle = fi.FaultInjectorHandle(base_url="http://control", seed=0, mean_interval_seconds=1e9, cell_type="rollout")
    handle.phase_history = history
    return handle


def test_the_rollout_witness_accepts_one_paired_recovery_per_injection() -> None:
    """Two injections each followed by their own Running -> Pending -> Running is the happy path."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import assert_rollout_healed_through_pending

    history = _replay(["Running", INJECT, "Pending", "Running", INJECT, "Pending", "Running"])
    assert_rollout_healed_through_pending(_injector_with_history(history))


def test_the_rollout_witness_rejects_one_recovery_standing_in_for_two_injections() -> None:
    """Regression: independent 'injected twice' and 'healed once' assertions passed this run."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import assert_rollout_healed_through_pending

    history = _replay(["Running", INJECT, "Pending", "Running", INJECT, "Running"])
    with pytest.raises(AssertionError, match="followed by no"):
        assert_rollout_healed_through_pending(_injector_with_history(history))


def test_the_rollout_witness_rejects_a_single_recovery() -> None:
    """One heal could be a fluke, so the soak must witness the floor of two."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import assert_rollout_healed_through_pending

    history = _replay(["Running", INJECT, "Pending", "Running"])
    with pytest.raises(AssertionError, match="need >= 2"):
        assert_rollout_healed_through_pending(_injector_with_history(history))


def test_the_rollout_witness_tolerates_a_recovery_that_training_end_cut_short() -> None:
    """A cell still out of Running when training stops is unfinished, not a healing failure."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import assert_rollout_healed_through_pending

    history = _replay(["Running", INJECT, "Pending", "Running", INJECT, "Pending", "Running", INJECT, "Pending"])
    assert_rollout_healed_through_pending(_injector_with_history(history))


def _mode(*ft_components: str) -> FTTestMode:
    return dataclasses.replace(next(iter(MODES.values())), ft_components=tuple(ft_components))


def test_a_trainer_only_soak_targets_actor_cells() -> None:
    """It must not crash engines that its assertions say nothing about."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("train")) == "actor"


def test_a_rollout_only_soak_targets_rollout_cells() -> None:
    """Crashing trainer cells here would exercise a component this mode did not enable ft on."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("rollout")) == "rollout"


def test_a_mixed_soak_targets_every_kind() -> None:
    """The point of the mixed mode is that both kinds fail during one run."""
    from tests.e2e.ft.conftest_ft.scenario_ft_random import compute_injected_cell_type

    assert compute_injected_cell_type(_mode("train", "rollout")) is None
