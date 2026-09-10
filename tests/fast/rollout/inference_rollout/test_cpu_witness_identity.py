from miles.rollout.inference_rollout.inference_rollout_common import _stamp_training_sample_identities
from miles.utils.types import Sample


class TestStampTrainingSampleIdentities:
    def test_child_rows_retain_the_original_source_obligation(self) -> None:
        """Generated child rows keep their source identity before later filtering."""
        rows = [Sample(index=101), Sample(index=102), Sample(index=103)]

        _stamp_training_sample_identities(rows, source_sample_index=7)

        assert [row.source_sample_index for row in rows] == [7, 7, 7]
        assert [row.sample_row_index for row in rows] == [0, 1, 2]
        assert [row.sample_row_count for row in rows] == [3, 3, 3]

    def test_single_sample_uses_the_ordinary_identity(self) -> None:
        """Ordinary generation remains one row for its original sample."""
        row = Sample(index=7)

        _stamp_training_sample_identities(row, source_sample_index=7)

        assert (row.source_sample_index, row.sample_row_index, row.sample_row_count) == (7, 0, 1)
