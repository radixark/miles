import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _Effects


@pytest.fixture
def effects(monkeypatch: pytest.MonkeyPatch) -> _Effects:
    return _Effects(monkeypatch)
