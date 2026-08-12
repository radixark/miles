from collections.abc import Callable
from dataclasses import dataclass, field
import pytest

from miles.utils.external_utils.command_utils.ray_backend import command


@dataclass(frozen=True)
class FakeSchedulingStrategy:
    node_id: str
    soft: bool


@dataclass
class FakeRayExecution:
    function: Callable[[str, bool], str | None]
    scheduled_node_ids: list[str] = field(default_factory=list)
    strategy: FakeSchedulingStrategy | None = None

    def options(self, *, scheduling_strategy: FakeSchedulingStrategy) -> "FakeRayExecution":
        self.strategy = scheduling_strategy
        return self

    def remote(self, cmd: str, *, capture_output: bool) -> str | None:
        assert self.strategy is not None
        self.scheduled_node_ids.append(self.strategy.node_id)
        return self.function(cmd, capture_output)


@dataclass
class FakeRay:
    node_records: list[dict[str, object]]
    get_error: Exception | None = None
    init_addresses: list[str] = field(default_factory=list)
    shutdown_count: int = 0

    def init(self, *, address: str) -> None:
        self.init_addresses.append(address)

    def nodes(self) -> list[dict[str, object]]:
        return self.node_records

    def get(self, references: list[str | None]) -> list[str | None]:
        if self.get_error is not None:
            raise self.get_error
        return references

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def fake_ray_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[FakeRay, FakeRayExecution]]:
    def create(
        node_records: list[dict[str, object]], *, get_error: Exception | None = None
    ) -> tuple[FakeRay, FakeRayExecution]:
        fake_ray = FakeRay(node_records=node_records, get_error=get_error)
        ray_execution = FakeRayExecution(function=command._exec_command_on_node._function)
        monkeypatch.setattr(command, "ray", fake_ray)
        monkeypatch.setattr(command._exec_command_on_node, "options", ray_execution.options)
        monkeypatch.setattr(command, "NodeAffinitySchedulingStrategy", FakeSchedulingStrategy)
        return fake_ray, ray_execution

    return create
