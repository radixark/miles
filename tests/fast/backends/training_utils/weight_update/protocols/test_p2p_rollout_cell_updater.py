import threading
from types import ModuleType
from typing import Any

import pytest


class _RecordingTransferEngine:
    def __init__(self, return_code: int = 0, error: Exception | None = None, gate: threading.Event | None = None):
        self.calls: list[tuple[str, list[int], list[int], list[int]]] = []
        self._return_code = return_code
        self._error = error
        self._gate = gate

    def batch_transfer_sync_write(
        self, session_id: str, source_ptrs: list[int], target_ptrs: list[int], source_lens: list[int]
    ) -> int:
        if self._gate is not None:
            self._gate.wait(timeout=30)
        self.calls.append((session_id, list(source_ptrs), list(target_ptrs), list(source_lens)))
        if self._error is not None:
            raise self._error
        return self._return_code


def _remote_session(
    p2p_rollout_cell_updater: ModuleType,
    p2p_transfer_utils: ModuleType,
    session_id: str,
    weights: dict[str, tuple[int, int, int]],
) -> Any:
    return p2p_rollout_cell_updater.RemoteWeightInfo(
        session_id,
        {name: p2p_transfer_utils.RemoteWeightLocation(*location) for name, location in weights.items()},
    )


class TestDoP2PWriteOneSession:
    """The single-session write from the shared CPU pinned buffers."""

    def test_every_registered_weight_is_written_in_one_batch(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """One batch call carries the whole bucket, addressed source pointer by target pointer."""
        transfer_engine = _RecordingTransferEngine()
        session = _remote_session(
            p2p_rollout_cell_updater,
            p2p_transfer_utils,
            "session-a",
            {"w0": (0x2000, 4, 2), "w1": (0x3000, 8, 2)},
        )

        p2p_rollout_cell_updater._do_p2p_write_one_session(
            transfer_engine,
            session,
            ["w0", "w1"],
            {"w0": (0x1000, 4, 2), "w1": (0x1100, 8, 2)},
        )

        assert transfer_engine.calls == [("session-a", [0x1000, 0x1100], [0x2000, 0x3000], [8, 16])]

    def test_an_unregistered_weight_is_rejected(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A weight with no local registration has no pinned source buffer to read from."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(AssertionError, match="_weight_memory_registry of w0"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(_RecordingTransferEngine(), session, ["w0"], {})

    def test_a_weight_the_remote_session_does_not_hold_is_rejected(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """Skipping a weight silently would pair the remaining source pointers with the wrong targets."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(AssertionError, match="Pointer count mismatch"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(
                _RecordingTransferEngine(),
                session,
                ["w0", "w1"],
                {"w0": (0x1000, 4, 2), "w1": (0x1100, 4, 2)},
            )

    def test_an_empty_bucket_transfers_nothing(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A bucket with no name must not issue an empty batch to the transfer engine."""
        transfer_engine = _RecordingTransferEngine()
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        p2p_rollout_cell_updater._do_p2p_write_one_session(transfer_engine, session, [], {"w0": (0x1000, 4, 2)})

        assert transfer_engine.calls == []

    def test_a_failing_return_code_raises(
        self, p2p_rollout_cell_updater: ModuleType, p2p_transfer_utils: ModuleType
    ) -> None:
        """A negative return code means the remote never received the weights."""
        session = _remote_session(p2p_rollout_cell_updater, p2p_transfer_utils, "session-a", {"w0": (0x2000, 4, 2)})

        with pytest.raises(RuntimeError, match="Transfer failed for session session-a"):
            p2p_rollout_cell_updater._do_p2p_write_one_session(
                _RecordingTransferEngine(return_code=-1), session, ["w0"], {"w0": (0x1000, 4, 2)}
            )
