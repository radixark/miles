from miles.utils.types import Sample


def stamp_training_sample_identities(output: Sample | list[Sample], *, source_sample_index: int | None) -> None:
    samples = output if isinstance(output, list) else [output]
    source_sample_index = source_sample_index if source_sample_index is not None else samples[0].index
    assert source_sample_index is not None, "Training samples require a source sample index"
    for row_index, sample in enumerate(samples):
        sample.source_sample_index = source_sample_index
        sample.sample_row_index = row_index
        sample.sample_row_count = len(samples)
