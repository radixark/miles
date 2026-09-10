from miles.backends.megatron_utils.model import _consumed_sample_identities
from miles.backends.training_utils.data import DataIterator
from miles.utils.audit_utils.witness.cpu import TrainingSampleIdentity


class TestConsumedSampleIdentities:
    def test_fixed_microbatches_include_only_the_completed_step(self) -> None:
        """Fixed microbatches expose exactly the rows consumed since step start."""
        iterator = DataIterator(
            {
                "source_sample_indices": [7, 7, 8],
                "sample_row_indices": [0, 1, 0],
                "sample_row_counts": [2, 2, 1],
            },
            micro_batch_size=1,
        )
        iterator.offset = 3

        assert _consumed_sample_identities(iterator, start=1) == [
            TrainingSampleIdentity(source_sample_index=7, row_index=1, row_count=2),
            TrainingSampleIdentity(source_sample_index=8, row_index=0, row_count=1),
        ]

    def test_dynamic_microbatches_preserve_schedule_order_and_duplicates(self) -> None:
        """Dynamic scheduling preserves exact row occurrences without set deduplication."""
        iterator = DataIterator(
            {
                "source_sample_indices": [7, 7, 8],
                "sample_row_indices": [0, 1, 0],
                "sample_row_counts": [2, 2, 1],
            },
            micro_batch_indices=[[2], [0, 1, 0]],
        )
        iterator.offset = 2

        assert _consumed_sample_identities(iterator, start=1) == [
            TrainingSampleIdentity(source_sample_index=7, row_index=0, row_count=2),
            TrainingSampleIdentity(source_sample_index=7, row_index=1, row_count=2),
            TrainingSampleIdentity(source_sample_index=7, row_index=0, row_count=2),
        ]
