from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _ApiServer, _CellOperations, _Clock, _Effects, _Timer

from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.fault_injector import request_executor
from miles.utils.test_utils.fault_injector.actions import remote
from miles.utils.test_utils.fault_injector.actions.base import FaultHookResources
from miles.utils.test_utils.fault_injector.controller import _FaultHookController
from miles.utils.test_utils.fault_injector.models import FaultHookRecord


@pytest.fixture
def operations() -> _CellOperations:
    return _CellOperations()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(request_executor, "time", fake)
    return fake


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> list[_Timer]:
    created: list[_Timer] = []

    def create(**kwargs: object) -> _Timer:
        created.append(timer := _Timer(**kwargs))
        return timer

    monkeypatch.setattr(request_executor, "threading", SimpleNamespace(Timer=create))
    return created


@pytest.fixture
def hook_records(tmp_path: Path) -> Iterator[Callable[[], list[FaultHookRecord]]]:
    log_dir = tmp_path / "events"
    set_event_logger(EventLogger(log_dir=log_dir, source=SimpleProcessIdentity(component="main")))

    def read() -> list[FaultHookRecord]:
        return [event.record for event in read_events(log_dir) if isinstance(event, FaultHookEvent)]

    yield read
    set_event_logger(None)


@pytest.fixture
def runtime_hooks(operations: _CellOperations) -> _FaultHookController:
    hooks = _FaultHookController()
    hooks.configure(resources=FaultHookResources(cell_operations=operations))
    return hooks


@pytest.fixture
def effects(monkeypatch: pytest.MonkeyPatch) -> _Effects:
    return _Effects(monkeypatch)


@pytest.fixture
def api_server(monkeypatch: pytest.MonkeyPatch) -> _ApiServer:
    server = _ApiServer(target={"kind": "observed", "cell_id": "rollout-0", "rank": 1, "workers_hash": "hash-a"})
    monkeypatch.setattr(remote, "httpx", SimpleNamespace(AsyncClient=server.client))
    return server
