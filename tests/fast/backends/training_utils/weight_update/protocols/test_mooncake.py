from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.utils.test_utils.fault_injector.fakes import _arm_marker_hook

from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.models import FaultHookName


def test_a_weight_of_another_size_on_the_shard_is_never_written(p2p_sender: Any, mooncake_module: ModuleType) -> None:
    """Writing a weight whose published size differs would run past the target's memory."""
    transport = mooncake_module.MooncakeTransport()
    remote_shard = mooncake_module.RemoteShard(
        cell_id="cell-a",
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        session_id="cell-a-r0",
        weight_locations_by_name={"w": mooncake_module.RemoteWeightLocation(address=0x1000, numel=2, element_size=4)},
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
            ("write", shard.session_id),
            ("write", shard.session_id),
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

    def test_a_late_native_failure_uses_the_submitted_update_and_cell_generation(
        self,
        p2p_sender: Any,
        transport: Any,
        make_remote_shard: Any,
        mooncake_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A late worker failure must never borrow the active context of a later weight update."""
        events = []
        monkeypatch.setattr(mooncake_module, "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(
            mooncake_module,
            "get_event_logger",
            lambda: SimpleNamespace(log=lambda event_type, data, **kwargs: events.append((event_type, data, kwargs))),
        )
        tensor = torch.ones(4)
        shard = make_remote_shard()
        transport.register_memory(tensor)
        hold = p2p_sender.transfer_engine.hold(shard.session_id)
        p2p_sender.transfer_engine.failing_sessions.add(shard.session_id)
        old = FaultHookContext(debug_weight_update_id="old", snapshot_cell_id_to_hashes={"cell-a": "old-generation"})
        new = FaultHookContext(debug_weight_update_id="new", snapshot_cell_id_to_hashes={"cell-a": "new-generation"})
        with mooncake_module.fault_hook_controller.with_context(old):
            (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
        assert hold.entered.wait(timeout=10)

        with mooncake_module.fault_hook_controller.with_context(new):
            hold.release.set()
            with pytest.raises(RuntimeError, match="returned -1"):
                future.result(timeout=10)

        assert len(events) == 1
        event_type, data, kwargs = events[0]
        assert event_type is mooncake_module.WeightTransferFailedEvent
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
        mooncake_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Observability failures must preserve the native error that makes the cell unusable."""

        def fail_log(*args: Any, **kwargs: Any) -> None:
            raise ValueError("logger unavailable")

        monkeypatch.setattr(mooncake_module, "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(mooncake_module, "get_event_logger", lambda: SimpleNamespace(log=fail_log))
        tensor = torch.ones(4)
        shard = make_remote_shard()
        transport.register_memory(tensor)
        p2p_sender.transfer_engine.failing_sessions.add(shard.session_id)
        context = FaultHookContext(
            debug_weight_update_id="update", snapshot_cell_id_to_hashes={"cell-a": "generation"}
        )

        with mooncake_module.fault_hook_controller.with_context(context):
            (future,) = transport.write([shard], {"w": tensor}, run_write=lambda shard, write: write())
        with pytest.raises(RuntimeError, match="returned -1"):
            future.result(timeout=10)
