from miles.utils.types import Sample, SampleLineage


def stamp_sample_lineage(output: Sample | list[Sample], *, source_sample_index: int) -> None:
    samples = output if isinstance(output, list) else [output]
    assert source_sample_index is not None, "Sample lineage requires a source sample index"
    for output_index, sample in enumerate(samples):
        sample.lineage = SampleLineage(
            source_sample_index=source_sample_index, output_index=output_index, output_count=len(samples)
        )
