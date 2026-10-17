from typing import Any

import pytest


class TestCellWrites:
    def test_a_worker_failure_marks_only_its_submitting_cell(self, make_cell_updater: Any) -> None:
        """A failed worker must blame its submitting cell without poisoning a healthy peer."""
        broken, healthy = make_cell_updater("broken"), make_cell_updater("healthy")
        calls = []
        error = RuntimeError("write failed")

        def fail() -> None:
            raise error

        with pytest.raises(RuntimeError, match="write failed"):
            broken.run_write(fail)
        healthy.run_write(lambda: calls.append("healthy"))

        assert broken._error is error
        assert not healthy.is_errored
        assert calls == ["healthy"]

    def test_an_errored_cell_skips_the_native_call(self, make_cell_updater: Any) -> None:
        """The executor must recheck cell state before any queued write enters native code."""
        updater = make_cell_updater()
        updater.mark_errored(RuntimeError("lost"))
        calls = []

        updater.run_write(lambda: calls.append("written"))

        assert calls == []
