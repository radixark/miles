from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.model_companion import SampleIdentityExtractor
from miles.utils.types import SampleLineage


class TestConsumedSampleIdentities:
    def test_fixed_microbatches_include_only_the_completed_step(self) -> None:
        """Fixed microbatches expose exactly the rows consumed since step start."""
        iterator = DataIterator(
            {
                "lineage_source_sample_indices": [7, 7, 8],
                "lineage_output_indices": [0, 1, 0],
                "lineage_output_counts": [2, 2, 1],
            },
            micro_batch_size=1,
        )
        iterator.offset = 3

        assert SampleIdentityExtractor.get_consumed_sample_identities(
            start_offset=1,
            end_offset=iterator.offset,
            data=iterator.rollout_data,
            micro_batch_indices=iterator.micro_batch_indices,
        ) == [
            SampleLineage(source_sample_index=7, output_index=1, output_count=2),
            SampleLineage(source_sample_index=8, output_index=0, output_count=1),
        ]

    def test_dynamic_microbatches_preserve_schedule_order_and_duplicates(self) -> None:
        """Dynamic scheduling preserves exact row occurrences without set deduplication."""
        iterator = DataIterator(
            {
                "lineage_source_sample_indices": [7, 7, 8],
                "lineage_output_indices": [0, 1, 0],
                "lineage_output_counts": [2, 2, 1],
            },
            micro_batch_indices=[[2], [0, 1, 0]],
        )
        iterator.offset = 2

        assert SampleIdentityExtractor.get_consumed_sample_identities(
            start_offset=1,
            end_offset=iterator.offset,
            data=iterator.rollout_data,
            micro_batch_indices=iterator.micro_batch_indices,
        ) == [
            SampleLineage(source_sample_index=7, output_index=0, output_count=2),
            SampleLineage(source_sample_index=7, output_index=1, output_count=2),
            SampleLineage(source_sample_index=7, output_index=0, output_count=2),
        ]
