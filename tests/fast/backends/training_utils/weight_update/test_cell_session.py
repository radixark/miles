import asyncio
from argparse import Namespace

import pytest

from miles.backends.training_utils.weight_update.cell_session import _PerCellEngineSession, _raise_if_unsuccessful
from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth


class _FakeEngine:
    def __init__(
        self,
        log: list[tuple],
        cell_id: str,
        *,
        failing_ops: set[str] | None = None,
        unsuccessful: set[str] | None = None,
    ):
        self._log = log
        self._cell_id = cell_id
        self._failing_ops = failing_ops if failing_ops is not None else set()
        self._unsuccessful = unsuccessful if unsuccessful is not None else set()
        self.started: list[str] = []

    async def _answer(self, op: str, detail: object = None):
        self.started.append(op)
        self._log.append((op, self._cell_id, detail))
        if op in self._failing_ops:
            raise ConnectionError(f"{self._cell_id} is unreachable")
        if op in self._unsuccessful:
            return {"success": False, "message": f"{self._cell_id} refused {op}"}
        return {"success": True}

    async def pause_generation(self, mode: str):
        return await self._answer("pause_generation", mode)

    async def flush_cache(self):
        return await self._answer("flush_cache")

    async def begin_weight_update(self, selector: str, sync_base: bool):
        return await self._answer("begin_weight_update", (selector, sync_base))

    async def end_weight_update(self):
        return await self._answer("end_weight_update")

    async def update_weight_version(self, weight_version: str):
        return await self._answer("update_weight_version", weight_version)

    async def continue_generation(self):
        return await self._answer("continue_generation")


def _session(engines: dict[str, _FakeEngine], health: InferenceCellHealth, *, pause_mode: str = "retract"):
    return _PerCellEngineSession(Namespace(pause_generation_mode=pause_mode), engines, health)


def _fleet(log: list[tuple], cell_ids: list[str], **kwargs) -> dict[str, _FakeEngine]:
    return {cell_id: _FakeEngine(log, cell_id, **kwargs) for cell_id in cell_ids}


class TestFailureAttribution:
    """A cell that refuses or never answers must be the only cell that loses the update."""

    def test_a_transport_error_only_errors_its_own_cell(self) -> None:
        """One dead engine must not abort the session frame of the cells that are still serving."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0", "cell-1"])
        engines["cell-0"] = _FakeEngine(log, "cell-0", failing_ops={"begin_weight_update"})
        health = InferenceCellHealth(["cell-0", "cell-1"])

        _session(engines, health).begin(selector="all", sync_base=True)

        assert health.errored_cell_ids == ["cell-0"]
        assert isinstance(health.error_of("cell-0"), ConnectionError)
        assert [entry[1] for entry in log if entry[0] == "begin_weight_update"] == ["cell-0", "cell-1"]

    def test_a_refusal_without_an_exception_is_a_failure_too(self) -> None:
        """An engine answering success=False did not take the weights, and silently going on publishes a lie."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0", "cell-1"])
        engines["cell-0"] = _FakeEngine(log, "cell-0", unsuccessful={"end_weight_update"})
        health = InferenceCellHealth(["cell-0", "cell-1"])

        _session(engines, health).end()

        assert health.errored_cell_ids == ["cell-0"]
        assert "refused end_weight_update" in str(health.error_of("cell-0"))

    def test_an_errored_cell_is_skipped_by_every_later_stage(self) -> None:
        """A cell that failed to begin holds no session, and resuming it would serve stale weights."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0", "cell-1"])
        engines["cell-0"] = _FakeEngine(log, "cell-0", failing_ops={"begin_weight_update"})
        health = InferenceCellHealth(["cell-0", "cell-1"])
        session = _session(engines, health)

        session.begin(selector="all", sync_base=True)
        session.end()
        session.set_weight_version(7)
        session.resume()

        assert [entry[0] for entry in log if entry[1] == "cell-0"] == ["begin_weight_update"]
        assert [entry[0] for entry in log if entry[1] == "cell-1"] == [
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
            "continue_generation",
        ]

    def test_a_cell_that_fails_late_keeps_its_first_error(self) -> None:
        """The first failure names the real cause of the cell being dropped."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0"])
        engines["cell-0"] = _FakeEngine(log, "cell-0", unsuccessful={"update_weight_version"})
        health = InferenceCellHealth(["cell-0"])
        session = _session(engines, health)

        session.set_weight_version(7)
        session.resume()

        assert "update_weight_version" in str(health.error_of("cell-0"))
        assert [entry[0] for entry in log] == ["update_weight_version"]


class TestRequestScheduling:
    """Every healthy cell has its request issued before any of them is collected."""

    def test_a_slow_cell_does_not_delay_the_requests_of_the_others(self) -> None:
        """Collecting one cell before submitting the next would serialize the fleet behind a hung engine."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0", "cell-1"])
        second_started = asyncio.Event()

        async def waiting_begin(selector: str, sync_base: bool):
            log.append(("begin_weight_update", "cell-0", (selector, sync_base)))
            await asyncio.wait_for(second_started.wait(), timeout=30.0)
            return {"success": True}

        async def signalling_begin(selector: str, sync_base: bool):
            log.append(("begin_weight_update", "cell-1", (selector, sync_base)))
            second_started.set()
            return {"success": True}

        engines["cell-0"].begin_weight_update = waiting_begin
        engines["cell-1"].begin_weight_update = signalling_begin
        health = InferenceCellHealth(["cell-0", "cell-1"])

        _session(engines, health).begin(selector="all", sync_base=True)

        assert sorted(entry[1] for entry in log) == ["cell-0", "cell-1"]
        assert health.errored_cell_ids == []


class TestSessionFrame:
    """The per-cell frame issues the same engine calls as the fleet-wide one."""

    def test_a_retracting_pause_also_flushes_the_cache(self) -> None:
        """Only in_place pausing keeps the KV cache; every other mode has to drop it."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0"])
        health = InferenceCellHealth(["cell-0"])

        _session(engines, health, pause_mode="retract").pause()

        assert [entry[0] for entry in log] == ["pause_generation", "flush_cache"]
        assert log[0][2] == "retract"

    def test_an_in_place_pause_keeps_the_cache(self) -> None:
        """Flushing would discard exactly the KV cache that in_place pausing preserves."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0"])
        health = InferenceCellHealth(["cell-0"])

        _session(engines, health, pause_mode="in_place").pause()

        assert [entry[0] for entry in log] == ["pause_generation"]

    def test_a_cell_that_fails_to_pause_is_not_asked_to_flush(self) -> None:
        """A cell that never paused has nothing to flush, and the second failure would hide the first."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0", "cell-1"])
        engines["cell-0"] = _FakeEngine(log, "cell-0", failing_ops={"pause_generation"})
        health = InferenceCellHealth(["cell-0", "cell-1"])

        _session(engines, health, pause_mode="retract").pause()

        assert [entry for entry in log if entry[0] == "flush_cache"] == [("flush_cache", "cell-1", None)]
        assert health.errored_cell_ids == ["cell-0"]

    def test_the_published_version_reaches_every_healthy_cell_as_a_string(self) -> None:
        """The engines key their served version by string, so an int would publish a different version."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0"])
        health = InferenceCellHealth(["cell-0"])

        _session(engines, health).set_weight_version(12)

        assert log == [("update_weight_version", "cell-0", "12")]


class TestNothingToDrive:
    """A rank whose cells have all failed must not hang or raise on an empty fan-out."""

    def test_a_fleet_without_a_healthy_cell_issues_no_request(self) -> None:
        """The driver still walks the whole frame, so every stage has to tolerate an empty healthy set."""
        log: list[tuple] = []
        engines = _fleet(log, ["cell-0"])
        health = InferenceCellHealth(["cell-0"])
        health.mark_errored("cell-0", RuntimeError("boom"))
        session = _session(engines, health)

        session.pause()
        session.begin(selector="all", sync_base=True)
        session.end()
        session.set_weight_version(3)
        session.resume()

        assert log == []


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"success": False, "error_message": "bad"}, "bad"),
        ({"success": False, "error": "worse"}, "worse"),
        ({"success": False}, "unknown error"),
    ],
)
def test_an_unsuccessful_answer_names_the_reason_the_engine_gave(result, expected) -> None:
    """The engine's own explanation is the only evidence the trainer has about why the cell was dropped."""
    with pytest.raises(RuntimeError, match=expected):
        _raise_if_unsuccessful("end_weight_update", result)


def test_an_answer_without_a_success_field_is_accepted() -> None:
    """pause_generation answers with a raw HTTP response, which carries no success flag to inspect."""
    _raise_if_unsuccessful("pause_generation", object())
