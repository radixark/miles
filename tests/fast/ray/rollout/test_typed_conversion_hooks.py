from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.ray.rollout.train_data_conversion import _post_process_rewards, convert_samples_to_train_data
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.types import Sample


class TestTypedConversionHooks:
    @pytest.mark.parametrize("reward_postprocess", [False, True])
    def test_custom_conversion_boundary_receives_merged_fields_and_preserves_inputs(
        self, reward_postprocess: bool
    ) -> None:
        """Conversion and reward postprocessing receive isolated flat plugin fields at their real entrypoints."""
        hook = _hook()
        args = ImmutableNamespace(custom_convert_samples_to_train_data_path=hook, custom_reward_post_process_path=hook)
        samples = [Sample(prompt="input")]
        output = object()

        def callback(view: ImmutableNamespace, values: list[Sample]) -> Any:
            assert view.hook_marker == 41
            assert values is samples
            return output

        if reward_postprocess:
            result = _post_process_rewards(args, samples, custom_reward_post_process_func=callback)
        else:
            result = convert_samples_to_train_data(
                args,
                samples,
                metadata={},
                custom_convert_samples_to_train_data_func=callback,
                custom_reward_post_process_func=None,
            )
        assert result is output
        assert "hook_marker" not in dict(args)
