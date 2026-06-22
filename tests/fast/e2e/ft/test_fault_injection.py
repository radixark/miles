import threading
from unittest.mock import MagicMock, patch

from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.e2e.ft.conftest_ft.fault_injection import RecoveryGate, cell_is_alive


def _cell(name: str, *, healthy: bool) -> dict:
    status = "True" if healthy else "False"
    return {"metadata": {"name": name}, "status": {"conditions": [{"type": "Healthy", "status": status}]}}


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
    """The control server's stale 'still healthy' view must not count a just-killed cell."""
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
            poll_interval_seconds=1e-6,
        )

    assert len(injected) >= 2, f"expected a second injection after recovery, got {injected}"
