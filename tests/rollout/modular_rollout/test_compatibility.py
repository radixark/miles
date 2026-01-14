import asyncio
from unittest.mock import patch

import pytest

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.modular_rollout.compatibility import (
    LegacyRolloutFnAdapter,
    call_rollout_function,
    load_rollout_function,
)


@pytest.fixture
def constructor_input():
    return RolloutFnConstructorInput(args="dummy_args", data_source="dummy_data_source")


class TestSupportedRolloutFormats:
    """
    Documentation test to show various supported rollout function formats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_raw_output(self, constructor_input, evaluation):
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            if evaluation:
                return {"metric": {"accuracy": 0.9}}
            return [[{"text": "sample"}]]

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "path.to.fn")

        input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
        result = call_rollout_function(fn, input_cls(rollout_id=1))

        assert isinstance(fn, LegacyRolloutFnAdapter)
        if evaluation:
            assert isinstance(result, RolloutFnEvalOutput)
            assert result.data == {"metric": {"accuracy": 0.9}}
        else:
            assert isinstance(result, RolloutFnTrainOutput)
            assert result.samples == [[{"text": "sample"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_typed_output(self, constructor_input, evaluation):
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            if evaluation:
                return RolloutFnEvalOutput(data={"ds": {"acc": 0.95}})
            return RolloutFnTrainOutput(samples=[[{"text": "typed"}]])

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "path.to.fn")

        input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
        result = call_rollout_function(fn, input_cls(rollout_id=1))

        if evaluation:
            assert isinstance(result, RolloutFnEvalOutput)
            assert result.data == {"ds": {"acc": 0.95}}
        else:
            assert isinstance(result, RolloutFnTrainOutput)
            assert result.samples == [[{"text": "typed"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_sync_class(self, constructor_input, evaluation):
        class SyncRolloutFn:
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            def __call__(self, input):
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"test": {"score": 1}})
                return RolloutFnTrainOutput(samples=[[{"text": "sync"}]])

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=SyncRolloutFn):
            fn = load_rollout_function(constructor_input, "path.to.SyncRolloutFn")

        input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
        result = call_rollout_function(fn, input_cls(rollout_id=1))

        assert isinstance(fn, SyncRolloutFn)
        expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_async_class(self, constructor_input, evaluation):
        class AsyncRolloutFn:
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            async def __call__(self, input):
                await asyncio.sleep(0.001)
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"benchmark": {"accuracy": 0.98}})
                return RolloutFnTrainOutput(samples=[[{"text": "async"}]])

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=AsyncRolloutFn):
            fn = load_rollout_function(constructor_input, "path.to.AsyncRolloutFn")

        input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
        result = call_rollout_function(fn, input_cls(rollout_id=1))

        assert isinstance(fn, AsyncRolloutFn)
        expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
        assert isinstance(result, expected_type)
