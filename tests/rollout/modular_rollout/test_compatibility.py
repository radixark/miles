import asyncio
from unittest.mock import MagicMock, patch

import pytest

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.modular_rollout.compatibility import (
    LegacyGenerateFnAdapter,
    LegacyRolloutFnAdapter,
    call_rollout_function,
    load_generate_function,
    load_rollout_function,
)
from miles.utils.async_utils import run


@pytest.fixture
def constructor_input():
    return RolloutFnConstructorInput(args="dummy_args", data_source="dummy_data_source")


@pytest.fixture
def make_generate_fn_input():
    def _make(evaluation: bool = False):
        state = MagicMock()
        state.args = MagicMock()

        return GenerateFnInput(
            state=state,
            sample={"text": "test prompt"},
            sampling_params={"temperature": 0.7},
            evaluation=evaluation,
        )

    return _make


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


class TestSupportedGenerateFormats:
    """
    Documentation test similar to TestSupportedRolloutFormats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_with_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params, evaluation=False):
            return "my_sample"

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=legacy_generate_fn):
            fn = load_generate_function("path.to.fn")

        result = run(fn(make_generate_fn_input(evaluation)))

        assert isinstance(fn, LegacyGenerateFnAdapter)
        assert isinstance(result, GenerateFnOutput)
        assert result.sample == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_without_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params):
            return "my_sample"

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=legacy_generate_fn):
            fn = load_generate_function("path.to.fn")

        result = run(fn(make_generate_fn_input(evaluation)))

        assert isinstance(fn, LegacyGenerateFnAdapter)
        assert isinstance(result, GenerateFnOutput)
        assert result.sample == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_new_async_function_api(self, make_generate_fn_input, evaluation):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            return GenerateFnOutput(sample="my_sample")

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=generate):
            fn = load_generate_function("path.to.fn")

        result = run(fn(make_generate_fn_input(evaluation)))

        assert isinstance(result, GenerateFnOutput)
        assert result.sample == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_new_class_api(self, make_generate_fn_input, evaluation):
        class MyGenerateFn:
            async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
                return GenerateFnOutput(sample="my_sample")

        with patch("miles.rollout.modular_rollout.compatibility.load_function", return_value=MyGenerateFn):
            fn = load_generate_function("path.to.fn")

        result = run(fn(make_generate_fn_input(evaluation)))

        assert isinstance(fn, MyGenerateFn)
        assert isinstance(result, GenerateFnOutput)
        assert result.sample == "my_sample"
