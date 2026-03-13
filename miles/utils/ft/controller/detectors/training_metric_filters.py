"""Centralized label-filter construction for run-scoped training metrics.

All detectors that query training-rank metrics (heartbeat, phase) must use
these helpers instead of building label dicts manually. This ensures every
query includes ``ft_run_id`` so that stale data from previous runs cannot
pollute current-run decisions.
"""

from __future__ import annotations

FT_RUN_ID_LABEL = "ft_run_id"


def build_training_metric_filters(
    *,
    rank: str,
    run_id: str | None,
    **extra_labels: str,
) -> dict[str, str]:
    filters: dict[str, str] = {"rank": rank, **extra_labels}

    if run_id:
        filters[FT_RUN_ID_LABEL] = run_id

    return filters
