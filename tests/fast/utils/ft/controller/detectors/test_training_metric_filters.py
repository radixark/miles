from miles.utils.ft.controller.detectors.training_metric_filters import (
    FT_RUN_ID_LABEL,
    build_training_metric_filters,
)


class TestBuildTrainingMetricFilters:
    def test_includes_ft_run_id_when_provided(self) -> None:
        filters = build_training_metric_filters(rank="0", run_id="run-123")
        assert filters == {"rank": "0", FT_RUN_ID_LABEL: "run-123"}

    def test_omits_ft_run_id_when_none(self) -> None:
        filters = build_training_metric_filters(rank="0", run_id=None)
        assert filters == {"rank": "0"}

    def test_omits_ft_run_id_when_empty_string(self) -> None:
        filters = build_training_metric_filters(rank="0", run_id="")
        assert filters == {"rank": "0"}

    def test_passes_through_extra_labels(self) -> None:
        filters = build_training_metric_filters(
            rank="1", run_id="run-abc", node_id="node-0",
        )
        assert filters == {"rank": "1", FT_RUN_ID_LABEL: "run-abc", "node_id": "node-0"}
