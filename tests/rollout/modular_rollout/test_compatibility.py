import asyncio
from unittest.mock import MagicMock, patch

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


class TestLoadRolloutFunction:
    def test_load_class_returns_instance(self, constructor_input):
        class MockRolloutClass:
            def __init__(self, input):
                self.input = input

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=MockRolloutClass):
            result = load_rollout_function(constructor_input, "some.module.MockRolloutClass")

        assert isinstance(result, MockRolloutClass)
        assert result.input is constructor_input

    def test_load_function_returns_adapter(self, constructor_input):
        def mock_fn():
            pass

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=mock_fn):
            result = load_rollout_function(constructor_input, "some.module.mock_fn")

        assert isinstance(result, LegacyRolloutFnAdapter)
        assert result.fn is mock_fn
        assert result.args == "dummy_args"
        assert result.data_source == "dummy_data_source"


class TestLegacyRolloutFnAdapter:
    def test_call_with_train_input_wraps_output(self, constructor_input):
        mock_samples = [[{"text": "sample"}]]
        mock_fn = MagicMock(return_value=mock_samples)
        adapter = LegacyRolloutFnAdapter(constructor_input, mock_fn)

        result = call_rollout_function(adapter, RolloutFnTrainInput(rollout_id=1))

        mock_fn.assert_called_once_with("dummy_args", 1, "dummy_data_source", evaluation=False)
        assert isinstance(result, RolloutFnTrainOutput)
        assert result.samples == mock_samples

    def test_call_with_eval_input_wraps_output(self, constructor_input):
        mock_data = {"metric": {"accuracy": 0.9}}
        mock_fn = MagicMock(return_value=mock_data)
        adapter = LegacyRolloutFnAdapter(constructor_input, mock_fn)

        result = call_rollout_function(adapter, RolloutFnEvalInput(rollout_id=2))

        mock_fn.assert_called_once_with("dummy_args", 2, "dummy_data_source", evaluation=True)
        assert isinstance(result, RolloutFnEvalOutput)
        assert result.data == mock_data

    def test_passthrough_train_output(self, constructor_input):
        expected_output = RolloutFnTrainOutput(samples=[[]])
        mock_fn = MagicMock(return_value=expected_output)
        adapter = LegacyRolloutFnAdapter(constructor_input, mock_fn)

        result = call_rollout_function(adapter, RolloutFnTrainInput(rollout_id=0))

        assert result is expected_output

    def test_passthrough_eval_output(self, constructor_input):
        expected_output = RolloutFnEvalOutput(data={})
        mock_fn = MagicMock(return_value=expected_output)
        adapter = LegacyRolloutFnAdapter(constructor_input, mock_fn)

        result = call_rollout_function(adapter, RolloutFnEvalInput(rollout_id=0))

        assert result is expected_output


class MockSyncRolloutClass:
    def __init__(self, input):
        self.input = input

    def __call__(self, input):
        return RolloutFnTrainOutput(samples=[[{"text": "sync_class"}]])


class MockAsyncRolloutClass:
    def __init__(self, input):
        self.input = input

    async def __call__(self, input):
        await asyncio.sleep(0.01)
        return RolloutFnTrainOutput(samples=[[{"text": "async_class"}]])


class MockAsyncRolloutClassEval:
    def __init__(self, input):
        self.input = input

    async def __call__(self, input):
        await asyncio.sleep(0.01)
        return RolloutFnEvalOutput(data={"metric": {"accuracy": 0.98}})


class TestCallRolloutFunction:
    def test_sync_adapter(self, constructor_input):
        mock_samples = [[{"text": "sample"}]]
        mock_fn = MagicMock(return_value=mock_samples)
        adapter = LegacyRolloutFnAdapter(constructor_input, mock_fn)

        result = call_rollout_function(adapter, RolloutFnTrainInput(rollout_id=1))

        assert isinstance(result, RolloutFnTrainOutput)
        assert result.samples == mock_samples

    def test_sync_class(self, constructor_input):
        instance = MockSyncRolloutClass(constructor_input)

        result = call_rollout_function(instance, RolloutFnTrainInput(rollout_id=1))

        assert isinstance(result, RolloutFnTrainOutput)
        assert result.samples == [[{"text": "sync_class"}]]

    def test_async_class(self, constructor_input):
        instance = MockAsyncRolloutClass(constructor_input)

        result = call_rollout_function(instance, RolloutFnTrainInput(rollout_id=1))

        assert isinstance(result, RolloutFnTrainOutput)
        assert result.samples == [[{"text": "async_class"}]]

    def test_async_class_eval(self, constructor_input):
        instance = MockAsyncRolloutClassEval(constructor_input)

        result = call_rollout_function(instance, RolloutFnEvalInput(rollout_id=2))

        assert isinstance(result, RolloutFnEvalOutput)
        assert result.data == {"metric": {"accuracy": 0.98}}
