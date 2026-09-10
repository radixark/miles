import pytest
from tests.e2e.ft.conftest_ft import scenario_precise_p2p_all_targets


class TestAllTargetP2PEntry:
    @pytest.mark.parametrize("fully_async", [False, True])
    def test_entry_enables_batch_faults_without_changing_topology(
        self, monkeypatch: pytest.MonkeyPatch, fully_async: bool
    ) -> None:
        """The all-target entry enables zero survivors and recovery gating for the real multi-engine mode."""
        calls: list[dict] = []
        monkeypatch.setattr(
            scenario_precise_p2p_all_targets, "run_random_crash", lambda **kwargs: calls.append(kwargs)
        )
        scenario_precise_p2p_all_targets.run_ci(
            mode="kill_rollout__dp2_tp2", seed=17, num_steps=80, fully_async=fully_async
        )
        assert calls == [
            dict(
                mode="kill_rollout__dp2_tp2",
                seed=17,
                num_steps=80,
                fully_async=fully_async,
                precise_p2p=True,
                all_p2p_targets=True,
                min_survivors=0,
                allow_during_recovery=False,
            )
        ]
