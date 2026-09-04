import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest

from miles.rollout.base_types import (
    BaseRolloutFn,
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.compatibility import (
    LegacyGenerateFnAdapter,
    LegacyRolloutFnAdapter,
    call_rollout_function,
    load_generate_function,
    load_rollout_function,
)
from miles.utils.async_utils import run
from miles.utils.function_registry import function_registry, load_function


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

        with function_registry.temporary("test:legacy_rollout", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_rollout")

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

        with function_registry.temporary("test:legacy_typed", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_typed")

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
        class SyncRolloutFn(BaseRolloutFn):
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            def __call__(self, input):
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"test": {"score": 1}})
                return RolloutFnTrainOutput(samples=[[{"text": "sync"}]])

        with function_registry.temporary("test:sync_class", SyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:sync_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, SyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_async_class(self, constructor_input, evaluation):
        class AsyncRolloutFn(BaseRolloutFn):
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            async def __call__(self, input):
                await asyncio.sleep(0.001)
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"benchmark": {"accuracy": 0.98}})
                return RolloutFnTrainOutput(samples=[[{"text": "async"}]])

        with function_registry.temporary("test:async_class", AsyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:async_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, AsyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)


class TestRolloutFnCheckpointing:
    def test_base_save_and_load_default_to_no_ops(self, constructor_input):
        """A rollout function that keeps no state inherits save/load and needs no code."""

        class StatelessRolloutFn(BaseRolloutFn):
            def __call__(self, input):
                return RolloutFnTrainOutput(samples=[])

        with function_registry.temporary("test:stateless", StatelessRolloutFn):
            fn = load_rollout_function(constructor_input, "test:stateless")

            assert fn.save(1) is None
            assert fn.load(1) is None

    def test_legacy_function_gets_no_op_save_and_load(self, constructor_input):
        """Plain slime-style rollout functions stay usable: the adapter supplies the new methods."""

        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_checkpoint", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_checkpoint")

            assert isinstance(fn, LegacyRolloutFnAdapter)
            assert fn.save(1) is None
            assert fn.load(None) is None

    def test_legacy_adapter_exposes_the_constructor_input(self, constructor_input):
        """The base class promises constructor_input, so the framework's own subclasses must set it too."""

        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_constructor_input", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_constructor_input")

            assert fn.constructor_input is constructor_input

    def test_custom_save_and_load_are_used(self, constructor_input):
        """A stateful rollout function checkpoints through its own save/load."""

        class StatefulRolloutFn(BaseRolloutFn):
            def __init__(self, input: RolloutFnConstructorInput):
                self.calls = []

            def __call__(self, input):
                return RolloutFnTrainOutput(samples=[])

            def save(self, rollout_id):
                self.calls.append(("save", rollout_id))

            def load(self, rollout_id):
                self.calls.append(("load", rollout_id))

        with function_registry.temporary("test:stateful", StatefulRolloutFn):
            fn = load_rollout_function(constructor_input, "test:stateful")
            fn.save(4)
            fn.load(4)

            assert fn.calls == [("save", 4), ("load", 4)]

    def test_same_path_loaded_twice_yields_independent_instances(self, constructor_input):
        """--eval-function-path defaults to --rollout-function-path, and each load builds its own object."""

        class StatefulRolloutFn(BaseRolloutFn):
            def __init__(self, input: RolloutFnConstructorInput):
                self.calls = []

            def __call__(self, input):
                return RolloutFnTrainOutput(samples=[])

            def save(self, rollout_id):
                self.calls.append(rollout_id)

        with function_registry.temporary("test:same_path", StatefulRolloutFn):
            train_fn = load_rollout_function(constructor_input, "test:same_path")
            eval_fn = load_rollout_function(constructor_input, "test:same_path")

            train_fn.save(2)

            assert train_fn is not eval_fn
            assert eval_fn.calls == []

    def test_subclass_without_call_cannot_be_instantiated(self, constructor_input):
        """__call__ is abstract, so a missing implementation fails at load, not mid-run."""

        class MissingCall(BaseRolloutFn):
            pass

        with function_registry.temporary("test:missing_call", MissingCall):
            with pytest.raises(TypeError, match="abstract"):
                load_rollout_function(constructor_input, "test:missing_call")

    def test_class_not_subclassing_base_is_rejected(self, constructor_input):
        """Rejected at load time, not at the first checkpoint hours into a run."""

        class NotARolloutFn:
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            def __call__(self, input):
                return RolloutFnTrainOutput(samples=[])

        with function_registry.temporary("test:not_a_rollout_fn", NotARolloutFn):
            with pytest.raises(TypeError, match="must subclass"):
                load_rollout_function(constructor_input, "test:not_a_rollout_fn")


class TestSupportedGenerateFormats:
    """
    Documentation test similar to TestSupportedRolloutFormats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_with_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params, evaluation=False):
            return "my_sample"

        with function_registry.temporary("test:legacy_gen_eval", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen_eval")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_without_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params):
            return "my_sample"

        with function_registry.temporary("test:legacy_gen", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_new_async_function_api(self, make_generate_fn_input, evaluation):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_async", generate):
            fn = load_generate_function("test:new_async")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_new_class_api(self, make_generate_fn_input, evaluation):
        class MyGenerateFn:
            async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
                return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_class", MyGenerateFn):
            fn = load_generate_function("test:new_class")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, MyGenerateFn)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"


class TestShippedRolloutFunctions:
    @pytest.mark.parametrize(
        "path",
        [
            "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
            "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn",
            "examples.infra_features.fully_async.external_eval_fn.ExternalSglangEvalFn",
        ],
    )
    def test_a_shipped_rollout_function_passes_the_loader_class_check(
        self, path: str, constructor_input: RolloutFnConstructorInput
    ) -> None:
        """load_rollout_function rejects a class outside the hierarchy, and it runs inside the
        rollout actor, so a miles-shipped path failing it kills the run at startup."""
        loaded = load_function(path)
        assert inspect.isclass(loaded)

        with patch.object(loaded, "__init__", return_value=None):
            fn = load_rollout_function(constructor_input, path)

        assert isinstance(fn, BaseRolloutFn)
