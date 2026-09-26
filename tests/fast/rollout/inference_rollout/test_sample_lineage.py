from miles.rollout.inference_rollout.inference_rollout_common import stamp_sample_lineage
from miles.utils.types import Sample


class TestStampSampleLineage:
    def test_child_rows_retain_the_original_source_obligation(self) -> None:
        """Generated child rows keep their source identity before later filtering."""
        rows = [Sample(index=101), Sample(index=102), Sample(index=103)]

        stamp_sample_lineage(rows, source_sample_index=7)

        assert [row.lineage.source_sample_index for row in rows] == [7, 7, 7]
        assert [row.lineage.output_index for row in rows] == [0, 1, 2]
        assert [row.lineage.output_count for row in rows] == [3, 3, 3]

    def test_single_sample_uses_the_ordinary_identity(self) -> None:
        """Ordinary generation remains one row for its original sample."""
        row = Sample(index=7)

        stamp_sample_lineage(row, source_sample_index=7)

        assert (
            row.lineage.source_sample_index,
            row.lineage.output_index,
            row.lineage.output_count,
        ) == (7, 0, 1)
