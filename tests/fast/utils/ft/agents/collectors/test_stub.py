import pytest

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.stub import StubCollector
from miles.utils.ft.models import CollectorOutput


class TestStubCollector:
    @pytest.fixture()
    def stub(self) -> StubCollector:
        return StubCollector()

    def test_is_base_collector_subclass(self) -> None:
        assert issubclass(StubCollector, BaseCollector)

    async def test_collect_returns_empty_metrics(self, stub: StubCollector) -> None:
        result = await stub.collect()
        assert result == CollectorOutput(metrics=[])


class TestBaseCollectorABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            BaseCollector()  # type: ignore[abstract]
