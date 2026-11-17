import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.utils.test_utils.fault_injector.fakes import _arm_marker_hook

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
)
from miles.utils.audit_utils.event_logger.models import WeightTransferFailedEvent
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller
from miles.utils.test_utils.fault_injector.models import FaultHookName


@pytest.fixture
def p2p_transport() -> str:
    return "mooncake"


def _p2p_transport_module(p2p_protocol: ModuleType) -> ModuleType:
    return sys.modules[p2p_protocol.P2PTransport.__module__]


def test_a_weight_of_another_size_on_the_shard_is_never_written(p2p_sender: Any, transport: Any) -> None:
    """Writing a weight whose published size differs would run past the target's memory."""
    remote_shard = RemoteShard(
        cell_id="cell-a",
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        transport_peer="cell-a-r0",
        weight_locations_by_name={"w": RemoteWeightLocation(address=0x1000, numel=2, element_size=4)},
    )

    with pytest.raises(
        AssertionError, match="w is 16 bytes here but 8 bytes on the target of rollout engine 0 rank 0"
    ):
        transport.write([remote_shard], {"w": torch.zeros(4)}, run_write=lambda shard, write: write())
    assert p2p_sender.transfer_engine.writes == []


class TestBeforeSendFaultHook:
    def test_the_hook_fires_once_before_the_batch_write(
        self, p2p_sender: Any, transport: Any, make_remote_shard: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one-shot hook must fire immediately before the first actual native write."""
        log = p2p_sender.log
        _arm_marker_hook(monkeypatch, log=log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        tensor = torch.ones(4)
        transport.register_memory(tensor)
        shard = make_remote_shard()

        for _ in range(2):
            (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
            future.result(timeout=10)

        assert log == [
            ("hook", FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND.value),
            ("write", shard.transport_peer),
            ("write", shard.transport_peer),
        ]

    def test_a_failing_hook_never_enters_native_code_or_poison_memory(
        self, p2p_sender: Any, transport: Any, make_remote_shard: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook failure must leave the receiver untouched and the source memory safe to unregister."""
        _arm_marker_hook(monkeypatch, log=[], hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND, fail=True)
        tensor = torch.ones(4)
        transport.register_memory(tensor)
        (future,) = transport.write([make_remote_shard()], {"w": tensor}, run_write=lambda shard, write: write())

        with pytest.raises(RuntimeError, match="failed"):
            future.result(timeout=10)
        transport.unregister_memory(tensor)
        assert p2p_sender.transfer_engine.writes == []
        assert p2p_sender.transfer_engine.registered == []

    @pytest.mark.parametrize("case", ["empty", "missing_name", "wrong_size"])
    def test_an_empty_or_rejected_bucket_leaves_the_hook_armed(
        self, transport: Any, make_remote_shard: Any, monkeypatch: pytest.MonkeyPatch, case: str
    ) -> None:
        """Validation failures and empty writes must not consume the fault reserved for a real send."""
        log = []
        _arm_marker_hook(monkeypatch, log=log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        shard = make_remote_shard()
        tensors = {"empty": {}, "missing_name": {"missing": torch.ones(4)}, "wrong_size": {"w": torch.ones(2)}}[case]

        if case == "empty":
            (future,) = transport.write([shard], tensors, run_write=lambda shard, write: write())
            future.result(timeout=10)
        else:
            with pytest.raises(AssertionError):
                transport.write([shard], tensors, run_write=lambda shard, write: write())
        assert log == []


class TestNativeFailure:
    @pytest.mark.parametrize("failure", ["negative", "exception"])
    def test_failed_native_completion_keeps_memory_registered_across_reconnects(
        self, p2p_sender: Any, transport: Any, make_remote_shard: Any, failure: str
    ) -> None:
        """Native error completion does not prove quiescence and must never permit unregistering its source."""
        tensor = torch.ones(4)
        shard = make_remote_shard()
        transport.register_memory(tensor)
        if failure == "negative":
            p2p_sender.transfer_engine.failing_sessions.add(shard.transport_peer)
        else:
            p2p_sender.transfer_engine.errors_by_session[shard.transport_peer] = RuntimeError("native exception")
        (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
        with pytest.raises(RuntimeError):
            future.result(timeout=10)

        transport.connect({}, [], [], [])
        with pytest.raises(RuntimeError, match="unknown native completion"):
            transport.unregister_memory(tensor)
        assert p2p_sender.transfer_engine.registered == [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
        assert transport._registrations_by_address[tensor.data_ptr()][0] is tensor

    def test_a_late_native_failure_uses_the_submitted_update_and_cell_generation(
        self,
        p2p_sender: Any,
        transport: Any,
        make_remote_shard: Any,
        p2p_protocol: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A late worker failure must never borrow the active context of a later weight update."""
        events = []
        monkeypatch.setattr(_p2p_transport_module(p2p_protocol), "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(
            _p2p_transport_module(p2p_protocol),
            "get_event_logger",
            lambda: SimpleNamespace(log=lambda event_type, data, **kwargs: events.append((event_type, data, kwargs))),
        )
        tensor = torch.ones(4)
        shard = make_remote_shard()
        transport.register_memory(tensor)
        hold = p2p_sender.transfer_engine.hold(shard.transport_peer)
        p2p_sender.transfer_engine.failing_sessions.add(shard.transport_peer)
        old = FaultHookContext(debug_weight_update_id="old", snapshot_cell_id_to_hashes={"cell-a": "old-generation"})
        new = FaultHookContext(debug_weight_update_id="new", snapshot_cell_id_to_hashes={"cell-a": "new-generation"})
        with fault_hook_controller.with_context(old):
            (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
        assert hold.entered.wait(timeout=10)

        with fault_hook_controller.with_context(new):
            hold.release.set()
            with pytest.raises(RuntimeError, match="returned -1"):
                future.result(timeout=10)

        assert len(events) == 1
        event_type, data, kwargs = events[0]
        assert event_type is WeightTransferFailedEvent
        assert (data["debug_weight_update_id"], data["cell_id"], data["workers_hash"]) == (
            "old",
            "cell-a",
            "old-generation",
        )
        assert data["started_at"].tzinfo is not None
        assert "returned -1" in data["error"]
        assert kwargs == {"include_context": False}

    def test_a_broken_event_logger_cannot_replace_the_native_error(
        self,
        p2p_sender: Any,
        transport: Any,
        make_remote_shard: Any,
        p2p_protocol: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Observability failures must preserve the native error that makes the cell unusable."""

        def fail_log(*args: Any, **kwargs: Any) -> None:
            raise ValueError("logger unavailable")

        monkeypatch.setattr(_p2p_transport_module(p2p_protocol), "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(
            _p2p_transport_module(p2p_protocol), "get_event_logger", lambda: SimpleNamespace(log=fail_log)
        )
        tensor = torch.ones(4)
        shard = make_remote_shard()
        transport.register_memory(tensor)
        p2p_sender.transfer_engine.failing_sessions.add(shard.transport_peer)
        context = FaultHookContext(
            debug_weight_update_id="update", snapshot_cell_id_to_hashes={"cell-a": "generation"}
        )

        with fault_hook_controller.with_context(context):
            (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
        with pytest.raises(RuntimeError, match="returned -1"):
            future.result(timeout=10)


@pytest.mark.parametrize("protocol,device", [(None, None), ("tcp", "mlx5_0")])
def test_transfer_engine_uses_the_receiver_transport_environment(
    mooncake_module: ModuleType, monkeypatch: pytest.MonkeyPatch, protocol: str | None, device: str | None
) -> None:
    """Trainer and receiver must initialize Mooncake with matching protocol and device settings."""
    calls = []
    engine = SimpleNamespace(initialize=lambda *args: calls.append(args))
    sdk = ModuleType("mooncake.engine")
    sdk.TransferEngine = lambda: engine
    monkeypatch.setitem(sys.modules, "mooncake.engine", sdk)
    monkeypatch.setattr(mooncake_module.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    for name, value in (("MOONCAKE_PROTOCOL", protocol), ("MOONCAKE_DEVICE", device)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert mooncake_module._create_transfer_engine() is engine
    assert calls == [("127.0.0.1", "P2PHANDSHAKE", protocol or "rdma", device or "")]


def test_a_rollout_engine_publishing_for_another_transport_is_rejected_at_connect(
    p2p_sender: Any, make_rollout_api: Any
) -> None:
    """Mooncake cannot reach memory the engine registered with another transport; its writes would all fail."""
    protocol = p2p_sender.make_protocol()

    with pytest.raises(AssertionError, match="publishes its weights for nixl, but --p2p-transport is mooncake"):
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1, transport="nixl")])
