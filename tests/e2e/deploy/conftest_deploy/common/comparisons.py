from tests.e2e.ft.conftest_ft import comparisons
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engine_count

# =========================== comparing the two sides ==========================


def compare_deterministic_sides(
    *,
    baseline_dir: str,
    target_dir: str,
    expected_engine_count: int,
    min_trained_rollouts: int,
    exclude_keys: list[str] | None = None,
) -> None:
    comparisons.compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        min_trained_rollouts=min_trained_rollouts,
        exclude_keys=exclude_keys,
    )

    for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
        assert_engine_count(side=side, dump_dir=side_dir, expected=expected_engine_count)
