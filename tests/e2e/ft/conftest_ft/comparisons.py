from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.dumps import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
)
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)
from miles.utils.test_utils.comparisons.metrics import (
    assert_every_metric_is_classified,
    assert_gradients_were_nonzero,
    compare_metrics,
)
from miles.utils.test_utils.reconfigure_assertions import assert_reconfigure_events

COMPARED_METRIC_PREFIXES: tuple[str, ...] = ("train/", "rollout/")
UNCOMPARED_METRIC_PREFIXES: tuple[str, ...] = ("perf/",)


def compare_deterministic_sides(
    *,
    baseline_dir: str,
    target_dir: str,
    min_trained_rollouts: int,
    expected_metric_deltas: dict[str, list[float]] | None = None,
) -> None:
    for side_dir in (baseline_dir, target_dir):
        assert_reconfigure_events(Path(side_dir) / EVENTS_DIRNAME, expected=[])

    for side_dir in (baseline_dir, target_dir):
        assert_every_metric_is_classified(
            side_dir, compared=COMPARED_METRIC_PREFIXES, ignored=UNCOMPARED_METRIC_PREFIXES
        )
    compare_metrics(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        rtol=0.0,
        atol=0.0,
        key_prefixes=list(COMPARED_METRIC_PREFIXES),
        exclude_keys=[],
        expected_deltas=expected_metric_deltas or {},
    )

    compare_dumps(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        diff_thresholds=[(".*", "rel <= 0")],
        allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
        allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    )

    compare_inference_engine_checksums(baseline_dir=baseline_dir, target_dir=target_dir)

    for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
        assert_engine_weights_moved(side=side, dump_dir=side_dir)
        assert_gradients_were_nonzero(side=side, dump_dir=side_dir, min_trained_rollouts=min_trained_rollouts)
