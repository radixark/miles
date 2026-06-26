from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size, postprocess_rollout_data
from miles.utils.misc import function_registry


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


class TestRolloutSampleFilterIsAppliedGenerically:
    """--rollout-sample-filter-path is honored at the single generic choke point
    (postprocess_rollout_data, reached by RolloutManager._get_rollout_data for
    EVERY rollout fn), not only inside the two built-in rollout fns. This is what
    lets a custom --rollout-function-path honor the documented filter hook."""

    def test_filter_receives_grouped_data_before_flatten(self):
        """The filter must see the documented `fn(args, data)` contract: `data`
        is the nested `list[list[Sample]]` grouped by n_samples_per_prompt,
        BEFORE the flatten/trim. It flags Sample.remove_sample in place."""
        seen: list = []

        def drop_last_in_group(args, data):
            # data is list[list[Sample]] grouped by n_samples_per_prompt.
            seen.append([len(group) for group in data])
            for group in data:
                group[-1].remove_sample = True

        args = make_args(
            global_batch_size=2,
            disable_rollout_trim_samples=True,
            use_dynamic_global_batch_size=False,
            rollout_sample_filter_path="test:drop_last_in_group",
        )
        nested = [
            [make_sample(group_index=0, index=0), make_sample(group_index=0, index=1)],
            [make_sample(group_index=1, index=2), make_sample(group_index=1, index=3)],
        ]
        with function_registry.temporary("test:drop_last_in_group", drop_last_in_group):
            out, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})

        # The filter saw grouped data (two groups of two) before flattening.
        assert seen == [[2, 2]]
        # Output is flattened; the filter's in-place mutation is preserved.
        assert len(out) == 4
        assert [s.remove_sample for s in out] == [False, True, False, True]

    def test_custom_rollout_fn_now_honors_filter_via_manager_path(self):
        """A *custom* rollout fn (any --rollout-function-path) need not apply the
        filter itself: its grouped output flows through postprocess_rollout_data,
        which applies the documented hook. Previously this silently no-op'd for
        custom fns because only the two built-in fns applied it in-fn."""

        def custom_rollout_fn_output():
            # Whatever a custom rollout fn produced: grouped, unfiltered.
            return [
                [make_sample(group_index=0, index=0, reward=1.0), make_sample(group_index=0, index=1, reward=0.0)],
                [make_sample(group_index=1, index=2, reward=1.0), make_sample(group_index=1, index=3, reward=0.0)],
            ]

        def drop_zero_reward(args, data):
            for group in data:
                for sample in group:
                    if sample.reward == 0.0:
                        sample.remove_sample = True

        args = make_args(
            global_batch_size=2,
            disable_rollout_trim_samples=True,
            use_dynamic_global_batch_size=False,
            rollout_sample_filter_path="test:drop_zero_reward",
        )
        with function_registry.temporary("test:drop_zero_reward", drop_zero_reward):
            out, _meta = postprocess_rollout_data(
                args, custom_rollout_fn_output(), train_parallel_config={"dp_size": 1}
            )

        # Zero-reward samples are flagged for loss exclusion; the consumer
        # (train_data_conversion: remove_sample -> loss_mask=0) is already generic.
        flagged = {s.index for s in out if s.remove_sample}
        assert flagged == {1, 3}

    def test_no_filter_path_is_a_no_op_default(self):
        """Default path: rollout_sample_filter_path=None must not flag anything
        and must leave the data bit-identical to the no-filter behavior."""
        args = make_args(
            global_batch_size=2,
            disable_rollout_trim_samples=True,
            use_dynamic_global_batch_size=False,
            rollout_sample_filter_path=None,
        )
        nested = [[make_sample(index=0), make_sample(index=1)], [make_sample(index=2)]]
        out, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})
        assert len(out) == 3
        assert all(s.remove_sample is False for s in out)
