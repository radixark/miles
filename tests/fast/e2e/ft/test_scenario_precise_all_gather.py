import pytest
from tests.e2e.ft.conftest_ft import scenario_precise_all_gather


class TestPreciseAllGatherEntry:
    @pytest.mark.parametrize("fully_async", [False, True])
    def test_entry_always_selects_precise_faults_and_preserves_run_parameters(
        self, monkeypatch: pytest.MonkeyPatch, fully_async: bool
    ) -> None:
        """The precise scenario cannot silently delegate to ordinary wall-clock injection."""
        calls: list[dict] = []
        monkeypatch.setattr(scenario_precise_all_gather, "run_random_crash", lambda **kwargs: calls.append(kwargs))

        scenario_precise_all_gather.run_ci(mode="kill_train__dp2_tp2", seed=17, num_steps=80, fully_async=fully_async)

        assert calls == [
            dict(mode="kill_train__dp2_tp2", seed=17, num_steps=80, fully_async=fully_async, precise_all_gather=True)
        ]
