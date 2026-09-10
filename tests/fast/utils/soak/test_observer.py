import random

import httpx
import pytest
from tests.fast.utils.soak.utils import StubFaultForm, typed_cell
from tests.utils.soak import core, observer, state, views


async def test_failed_and_empty_reads_have_different_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable API cannot erase cells or license injection from an old successful observation."""
    responses = [httpx.Response(503), httpx.Response(200, json={"items": []})]
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        observer.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(lambda request: responses.pop(0)),
            **kwargs,
        ),
    )
    reader = observer.SoakObserver(base_url="http://control", cell_types={"actor"})
    failed = await reader.observe()
    empty = await reader.observe()
    assert failed.cells is None and "cells" in failed.errors
    assert empty.cells == [] and empty.errors == {}
    assert views.project_legacy_events([failed]) == []
    assert len(views.project_legacy_events([empty])) == 1

    log = state.EventLog()
    log.note_schedule(state.SoakScheduleEvent(due_of_type={"actor": 0.0}))
    log.observe([typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")])
    log.note_observation(failed)
    scheduler = core.SoakActionScheduler(
        rng=random.Random(0),
        mean_intervals={"actor": 1.0},
        forms={"actor": [StubFaultForm("fault", lambda cell, rng: None)]},
    )
    assert scheduler.choose(events=log.events, now=10.0) is None
