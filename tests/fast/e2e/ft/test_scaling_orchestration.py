from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft import scaling
from tests.fast.e2e.ft.scaling_orchestration_fixtures import install_cluster_evidence


class TestScalingEvidenceChecks:
    @pytest.mark.parametrize(
        "corruption,message",
        [
            (None, None),
            ("missing_resize", "run ended after"),
            ("zero_gradient", "finite and non-zero"),
            ("counts", "only 0"),
        ],
    )
    def test_scenario_requires_resize_gradient_and_count_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        corruption: str | None,
        message: str | None,
    ) -> None:
        """A completed launcher cannot pass unless all three real post-run evidence checkers accept the run."""
        counts = install_cluster_evidence(monkeypatch=monkeypatch, tmp_path=tmp_path, corruption=corruption)
        kwargs = dict(
            test_name="scaling",
            mode=scaling.compute_scaling_mode(("train",)),
            pool_id="actor",
            cell_type="actor",
            scaling_flag="--actor-num-gpus-per-node",
            num_gpus_per_cell=2,
            read_counts_of_rollout_id=counts,
            counted="cells",
        )
        if corruption is None:
            scaling.run_scaling_scenario(**kwargs)
        else:
            with pytest.raises(AssertionError, match=message):
                scaling.run_scaling_scenario(**kwargs)
