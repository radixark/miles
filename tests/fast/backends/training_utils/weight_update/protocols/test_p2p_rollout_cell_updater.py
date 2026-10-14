from typing import Any


class TestCellWrites:
    def test_an_errored_cell_skips_the_native_call(self, make_cell_updater: Any) -> None:
        """The executor must recheck cell state before any queued write enters native code."""
        updater = make_cell_updater()
        updater.mark_errored(RuntimeError("lost"))
        calls = []

        updater.run_write(lambda: calls.append("written"))

        assert calls == []
