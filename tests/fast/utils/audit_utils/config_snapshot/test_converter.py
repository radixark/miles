import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from collections.abc import Callable


from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotDelta, ConfigSnapshotProcess
from miles.utils.test_utils.snapshot import dump_snapshot


class TestSharedProcessBases:
    def test_repeated_runs_reconstruct_every_field_rank_and_stage(self, make_record: Callable) -> None:
        """Shared bases preserve full configurations and every sampled stage across ranks."""
        common = {f"setting_{index:03d}": index for index in range(100)}
        configs = [
            {**common, "removed": [1, 2], "nested": {"value": "original"}},
            {**common, "added": None, "nested": {"value": "changed"}},
        ]
        records = [
            make_record(run=run, rank=rank, stage=stage, config={**config, "rank": rank})
            for run, config in enumerate(configs)
            for rank in range(2)
            for stage in ("process_config", "train_first_step")
        ]

        case = ConfigSnapshotConverter.convert(records)
        assert dump_snapshot(case) == dump_snapshot(ConfigSnapshotConverter.convert(list(reversed(records))))
        first, second = case.processes.values()
        assert isinstance(first, ConfigSnapshotProcess)
        assert isinstance(second, ConfigSnapshotProcess)
        assert second.base == {"args": {**configs[1], "rank": "$RANK"}}
        assert first.ranks == second.ranks == [0, 1]
        assert first.diffs == second.diffs == {"train_first_step-0000": ConfigSnapshotDelta(set={}, remove=[])}

    @pytest.mark.parametrize("change", ["field", "stage", "rank"])
    def test_repeated_run_changes_remain_observable(self, make_record: Callable, change: str) -> None:
        """Compression never hides a changed setting, missing stage, or missing rank."""
        common = {f"setting_{index:03d}": index for index in range(100)}
        records = [
            make_record(run=run, rank=rank, stage=stage, config={**common, "rank": rank})
            for run in range(2)
            for rank in range(2)
            for stage in ("process_config", "train_first_step")
        ]
        expected = dump_snapshot(ConfigSnapshotConverter.convert(records))
        if change == "field":
            records = [
                (
                    record.model_copy(update={"config": {"args": {**record.config["args"], "setting_050": -1}}})
                    if record.context.name.endswith("0001")
                    else record
                )
                for record in records
            ]
        elif change == "stage":
            records = [
                record
                for record in records
                if not (record.context.name.endswith("0001") and record.point.stage == "train_first_step")
            ]
        else:
            records = [
                record
                for record in records
                if not (record.context.name.endswith("0001") and record.context.source.rank_within_cell == 1)
            ]
        assert dump_snapshot(ConfigSnapshotConverter.convert(records)) != expected

    def test_rank_disagreement_is_rejected_before_compression(self, make_record: Callable) -> None:
        """Different configurations on equivalent ranks cannot become one shared base."""
        with pytest.raises(ValueError, match="Processes disagree"):
            ConfigSnapshotConverter.convert(
                [make_record(rank=0, config={"value": 1}), make_record(rank=1, config={"value": 2})]
            )


@pytest.mark.parametrize("changed", [True, 1.0])
def test_stage_type_disagreement_between_ranks_is_rejected(make_record: Callable, changed: object) -> None:
    """Equivalent-looking stage values of different types cannot silently merge across ranks."""
    records = [
        make_record(rank=0, config={"value": 0}),
        make_record(rank=1, config={"value": 0}),
        make_record(rank=0, stage="train_first_step", config={"value": 1}),
        make_record(rank=1, stage="train_first_step", config={"value": changed}),
    ]

    with pytest.raises(ValueError, match="Processes disagree"):
        ConfigSnapshotConverter.convert(records)


def test_stage_delta_records_only_changed_and_removed_fields(make_record: Callable) -> None:
    """Captured stages retain explicit deletions and null assignments without unchanged values."""
    records = [
        make_record(config={"removed": None, "nullable": 1, "same": 3}),
        make_record(stage="train_first_step", config={"nullable": None, "same": 3}),
    ]

    case = ConfigSnapshotConverter.convert(records)
    process = next(iter(case.processes.values()))

    assert process.diffs["train_first_step-0000"] == ConfigSnapshotDelta(
        set={"/args/nullable": None}, remove=["/args/removed"]
    )


class TestCaptureIntegrity:
    def test_missing_initial_configuration_is_rejected(self, make_record: Callable) -> None:
        """Later stages cannot invent the process baseline needed to interpret their deltas."""
        with pytest.raises(ValueError, match="Missing process_config-0000"):
            ConfigSnapshotConverter.convert([make_record(stage="train_first_step")])

    def test_duplicate_stage_indices_are_rejected(self, make_record: Callable) -> None:
        """Two records claiming the same stage occurrence cannot silently overwrite one another."""
        record = make_record()
        with pytest.raises(ValueError, match="Duplicate snapshot stage"):
            ConfigSnapshotConverter.convert([record, record])

    def test_one_capture_id_cannot_mix_process_contexts(self, make_record: Callable) -> None:
        """Capture IDs cannot merge records from different deployment identities."""
        first = make_record()
        second = make_record(stage="train_first_step")
        second = second.model_copy(
            update={"context": second.context.model_copy(update={"deploy_instance_id": "other"})}
        )

        with pytest.raises(ValueError, match="Mixed snapshot contexts"):
            ConfigSnapshotConverter.convert([first, second])
