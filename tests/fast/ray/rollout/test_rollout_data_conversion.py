from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size, postprocess_rollout_data


class TestComputeDynamicGlobalBatchSize:
    def test_rounds_down_to_multiple_of_dp_size(self):
        args = make_args(global_batch_size=64)
        # 13 samples, dp_size=4 → floor(13/4)*4 = 12
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=13)
        assert gbs == 12

    def test_returns_num_samples_when_already_aligned(self):
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=16)
        assert gbs == 16

    def test_falls_back_to_dp_size_when_below_dp_size(self):
        """When num_samples < dp_size, the rounded result is 0, which would
        produce a divide-by-zero downstream — fallback to dp_size."""
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=2)
        assert gbs == 4


class TestPostprocessRolloutData:
    def test_aligned_input_passes_through_unchanged(self):
        """Happy path: len(data) % global_batch_size == 0 → no trim, no metadata."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(8)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_unaligned_input_is_trimmed_to_multiple(self):
        """len % gbs != 0 → trim down to multiple. Tail samples are dropped."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_disable_rollout_trim_samples_keeps_unaligned_data(self):
        """With trim disabled, leave length as-is even if not divisible."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, _meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 11

    def test_raises_when_trim_would_produce_zero_samples(self):
        """5 samples, gbs=64 → trim_len=0 → can't proceed."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        with pytest.raises(ValueError, match="Not enough samples"):
            postprocess_rollout_data(args, [make_sample()] * 5, train_parallel_config={"dp_size": 1})

    def test_dynamic_batch_size_computed_and_recorded_in_metadata(self):
        """When use_dynamic_global_batch_size=True, the function recomputes gbs
        from the actual sample count and records it in metadata."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True)
        data = [make_sample(index=i) for i in range(10)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 2})
        # Dynamic gbs = floor(10/2)*2 = 10
        assert meta["dynamic_global_batch_size"] == 10
        assert len(out) == 10

    def test_flattens_nested_list_of_lists(self):
        """The function supports list[list[Sample]] input by flattening."""
        args = make_args(global_batch_size=2, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        nested = [[make_sample(index=0), make_sample(index=1)], [make_sample(index=2)]]
        out, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})
        assert len(out) == 3
