from miles.utils.workers.worker_spec import StaticMeta


class TestStaticMetaResolution:
    def test_zero_based_offsets_and_cell_indices_override_constants_without_mutating_them(self) -> None:
        """Per-cell metadata overrides stale constants and remains independent across observations."""
        declaration = StaticMeta(
            values={"role": "actor", "cell_index": 99, "gpu_offset": 99},
            include_cell_index=True,
            gpu_offset_base=0,
            gpu_offset_stride_per_cell=4,
        )

        first = declaration.resolve(cell_index=0)
        later = declaration.resolve(cell_index=3)
        first["role"] = "changed"

        assert first == {"role": "changed", "cell_index": 0, "gpu_offset": 0}
        assert later == {"role": "actor", "cell_index": 3, "gpu_offset": 12}
        assert declaration.values == {"role": "actor", "cell_index": 99, "gpu_offset": 99}
        assert declaration.resolve(cell_index=0) == {"role": "actor", "cell_index": 0, "gpu_offset": 0}
