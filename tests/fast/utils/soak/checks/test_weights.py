from datetime import datetime, timezone

import pytest
from tests.utils.soak.checks.weights import assert_published_weight_checksums

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, WeightUpdateResultEvent


class TestPublishedWeightChecksums:
    @pytest.mark.parametrize(
        "case, error",
        [
            ("complete", None),
            ("reordered", None),
            ("new_epoch", None),
            ("missing_last", "do not cover every publication"),
            ("stale", "do not cover every publication"),
            ("duplicate", "Duplicate checksum publication"),
            ("unchanged", "failed movement checks"),
            ("wrong_update", "do not cover every publication"),
            ("missing_epoch", "Checksum lacks epoch or update identity"),
            ("reused_update", "Repeated weight update identity"),
            ("duplicate_publication", "Repeated weight update identity"),
            ("duplicate_target", "Repeated updated engine"),
            ("failed_target", "Published engine is also reported failed"),
            ("tail_interrupted", None),
            ("tail_missing", "do not cover every publication"),
            ("tail_single", "Expected at least 2"),
            ("tail_foreign", "do not cover every publication"),
            ("tail_old_unchanged", "failed movement checks"),
        ],
    )
    def test_every_publication_requires_its_exact_checksum_evidence(self, case: str, error: str | None) -> None:
        """Match publications by identity and reject incomplete, conflicting or stale evidence."""
        events = []
        tail = case.startswith("tail_")
        for version in [1, 2] if case == "tail_single" or not tail else [1, 2, 3]:
            epoch = f"epoch-{version}" if case == "new_epoch" else "controller-epoch"
            published_version = 1 if case == "new_epoch" else version
            update_id = "reused-update" if case == "reused_update" else f"update-{version}"
            events.append(
                WeightUpdateResultEvent.model_validate(
                    dict(
                        timestamp=f"2026-01-01T00:00:0{version}Z",
                        source={"component": "trainer_controller", "trainer_id": "actor"},
                        update_id=update_id,
                        version_epoch=epoch,
                        rollout_id=version - 1,
                        candidate_version=version,
                        published_version=published_version,
                        target_incarnations={"engine": "generation"},
                        updated_cell_ids=["engine", "engine"] if case == "duplicate_target" else ["engine"],
                        failed_cell_ids=["engine"] if case == "failed_target" else [],
                    )
                )
            )
            if case == "duplicate_publication":
                events.append(events[-1])
            if case == "missing_last" and version == 2:
                continue
            if (case == "tail_interrupted" and version == 1) or (case == "tail_missing" and version == 3):
                continue
            tensors = {"rank0/w": "same" if case in {"unchanged", "tail_old_unchanged"} else str(version)}
            event = InferenceEngineWeightChecksumEvent.model_validate(
                dict(
                    timestamp=f"2026-01-01T00:00:0{version + 1}Z",
                    source={"component": "main"},
                    rollout_id=version - 1,
                    weight_version=published_version,
                    version_epoch=None if case == "missing_epoch" else epoch,
                    update_id=(
                        "other-update"
                        if case == "wrong_update" or (case == "tail_foreign" and version == 1)
                        else update_id
                    ),
                    movement_skip_reasons=[],
                    engine_checksums=[tensors],
                    engine_snapshots=[
                        dict(
                            model_name="actor",
                            cell_id="engine",
                            workers_hash="stale" if case == "stale" else "generation",
                            tensors=tensors,
                        )
                    ],
                )
            )
            events.append(event)
            if case == "duplicate":
                events.append(event)
        if case == "reordered":
            events.reverse()
        if error is None:
            assert_published_weight_checksums(
                events,
                publication_since=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc) if tail else None,
                minimum_publications=2 if tail else 1,
            )
        else:
            with pytest.raises(AssertionError, match=error):
                assert_published_weight_checksums(
                    events,
                    publication_since=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc) if tail else None,
                    minimum_publications=2 if tail else 1,
                )
