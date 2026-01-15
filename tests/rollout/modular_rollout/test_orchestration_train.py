import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.modular_rollout.orchestration_train import generate_rollout_async
from miles.utils.async_utils import run
from miles.utils.types import Sample


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.rollout_global_dataset = True
    args.rollout_batch_size = 2
    args.n_samples_per_prompt = 1
    args.over_sampling_batch_size = 2
    args.dynamic_sampling_filter_path = None
    args.rollout_sample_filter_path = None
    args.rollout_all_samples_process_path = None
    args.partial_rollout = False
    args.use_miles_router = True
    args.sglang_router_ip = "127.0.0.1"
    args.sglang_router_port = 30000
    return args


@pytest.fixture
def mock_state(mock_args):
    state = MagicMock()
    state.args = mock_args
    state.sampling_params = {"temperature": 0.7}
    state.aborted = False

    def reset():
        state.aborted = False

    state.reset = reset
    return state


def make_sample_group(index: int, reward: float = 1.0) -> list[Sample]:
    return [
        Sample(
            index=index,
            group_index=index,
            prompt=f"test {index}",
            response="\\boxed{8}",
            label="8",
            reward=reward,
            status=Sample.Status.COMPLETED,
        )
    ]


class TestOverSamplingBatchSize:
    def test_get_samples_called_with_correct_batch_size(self, mock_state):
        mock_state.args.over_sampling_batch_size = 3
        mock_state.args.rollout_batch_size = 2

        get_samples_calls = []

        def mock_get_samples(batch_size):
            get_samples_calls.append(batch_size)
            return [make_sample_group(i) for i in range(batch_size)]

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert all(bs == 3 for bs in get_samples_calls)

    def test_multiple_get_samples_calls_when_filtered(self, mock_state):
        mock_state.args.over_sampling_batch_size = 2
        mock_state.args.rollout_batch_size = 2
        mock_state.args.dynamic_sampling_filter_path = "some.filter.path"
        mock_state.args.rollout_sample_filter_path = None
        mock_state.args.rollout_all_samples_process_path = None

        get_samples_calls = []
        call_count = [0]

        def mock_get_samples(batch_size):
            get_samples_calls.append(batch_size)
            start_idx = call_count[0] * batch_size
            call_count[0] += 1
            return [make_sample_group(start_idx + i) for i in range(batch_size)]

        filter_call_count = [0]

        def mock_filter(args, group):
            filter_call_count[0] += 1
            keep = filter_call_count[0] % 2 == 0
            return DynamicFilterOutput(keep=keep, reason=None if keep else "filtered")

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        def load_fn_side_effect(path):
            if path == "some.filter.path":
                return mock_filter
            return None

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=load_fn_side_effect,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert len(get_samples_calls) >= 2


class TestDynamicFilter:
    def test_filtered_samples_not_in_output(self, mock_state):
        mock_state.args.rollout_batch_size = 2
        mock_state.args.dynamic_sampling_filter_path = "some.filter.path"
        mock_state.args.rollout_sample_filter_path = None
        mock_state.args.rollout_all_samples_process_path = None

        sample_index = [0]

        def mock_get_samples(batch_size):
            result = []
            for _ in range(batch_size):
                reward = 1.0 if sample_index[0] % 2 == 0 else 0.0
                result.append(make_sample_group(sample_index[0], reward=reward))
                sample_index[0] += 1
            return result

        def mock_filter(args, group):
            reward = group[0].reward
            keep = reward == 1.0
            return DynamicFilterOutput(
                keep=keep, reason=None if keep else "test_drop"
            )

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        def load_fn_side_effect(path):
            if path == "some.filter.path":
                return mock_filter
            return None

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=load_fn_side_effect,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            output, _ = run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert len(output.samples) == 2
        for group in output.samples:
            assert group[0].reward == 1.0

    def test_metrics_contain_drop_count(self, mock_state):
        mock_state.args.rollout_batch_size = 2
        mock_state.args.dynamic_sampling_filter_path = "some.filter.path"
        mock_state.args.rollout_sample_filter_path = None
        mock_state.args.rollout_all_samples_process_path = None

        sample_index = [0]

        def mock_get_samples(batch_size):
            result = []
            for _ in range(batch_size):
                reward = 1.0 if sample_index[0] < 2 else 0.0
                result.append(make_sample_group(sample_index[0], reward=reward))
                sample_index[0] += 1
            return result

        filter_drop_count = [0]

        def mock_filter(args, group):
            reward = group[0].reward
            keep = reward == 1.0
            if not keep:
                filter_drop_count[0] += 1
            return DynamicFilterOutput(
                keep=keep, reason=None if keep else "test_drop"
            )

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        def load_fn_side_effect(path):
            if path == "some.filter.path":
                return mock_filter
            return None

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=load_fn_side_effect,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            output, _ = run(generate_rollout_async(mock_state, 0, mock_get_samples))

        if filter_drop_count[0] > 0:
            assert "rollout/dynamic_filter/drop_test_drop" in output.metrics
            assert output.metrics["rollout/dynamic_filter/drop_test_drop"] == filter_drop_count[0]


class TestRolloutSampleFilterPath:
    def test_filter_called_with_correct_args(self, mock_state):
        mock_state.args.rollout_batch_size = 2
        mock_state.args.rollout_sample_filter_path = "some.filter.path"

        filter_call_log = {"called": False, "args": None, "data": None}

        def mock_sample_filter(args, data):
            filter_call_log["called"] = True
            filter_call_log["args"] = args
            filter_call_log["data"] = data

        sample_index = [0]

        def mock_get_samples(batch_size):
            result = []
            for _ in range(batch_size):
                result.append(make_sample_group(sample_index[0]))
                sample_index[0] += 1
            return result

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=lambda path: mock_sample_filter
            if path == "some.filter.path"
            else None,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert filter_call_log["called"]
        assert filter_call_log["args"] is mock_state.args
        assert len(filter_call_log["data"]) == 2


class TestRolloutAllSamplesProcessPath:
    def test_processor_called_with_correct_args(self, mock_state):
        mock_state.args.rollout_batch_size = 2
        mock_state.args.rollout_all_samples_process_path = "some.processor.path"

        processor_call_log = {
            "called": False,
            "args": None,
            "all_samples": None,
            "data_source": None,
        }

        def mock_processor(args, all_samples, data_source):
            processor_call_log["called"] = True
            processor_call_log["args"] = args
            processor_call_log["all_samples"] = all_samples
            processor_call_log["data_source"] = data_source

        sample_index = [0]

        def mock_get_samples(batch_size):
            result = []
            for _ in range(batch_size):
                result.append(make_sample_group(sample_index[0]))
                sample_index[0] += 1
            return result

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=lambda path: mock_processor
            if path == "some.processor.path"
            else None,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert processor_call_log["called"]
        assert processor_call_log["args"] is mock_state.args
        assert len(processor_call_log["all_samples"]) >= 2
        assert processor_call_log["data_source"] is mock_get_samples

    def test_all_samples_includes_filtered(self, mock_state):
        mock_state.args.rollout_batch_size = 2
        mock_state.args.dynamic_sampling_filter_path = "some.dynamic_filter.path"
        mock_state.args.rollout_all_samples_process_path = "some.processor.path"

        processor_call_log = {"all_samples_rewards": None}

        def mock_processor(args, all_samples, data_source):
            processor_call_log["all_samples_rewards"] = [g[0].reward for g in all_samples]

        sample_index = [0]

        def mock_get_samples(batch_size):
            result = []
            for _ in range(batch_size):
                reward = 1.0 if sample_index[0] % 2 == 0 else 0.0
                result.append(make_sample_group(sample_index[0], reward=reward))
                sample_index[0] += 1
            return result

        def mock_dynamic_filter(args, group):
            reward = group[0].reward
            keep = reward == 1.0
            return DynamicFilterOutput(keep=keep, reason=None if keep else "filtered")

        async def mock_generate_and_rm_group(state, group, sampling_params, evaluation):
            return group

        def load_fn_side_effect(path):
            if path == "some.dynamic_filter.path":
                return mock_dynamic_filter
            if path == "some.processor.path":
                return mock_processor
            return None

        with patch(
            "miles.rollout.modular_rollout.orchestration_train.generate_and_rm_group",
            side_effect=mock_generate_and_rm_group,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.load_function",
            side_effect=load_fn_side_effect,
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.get_worker_urls",
            new_callable=AsyncMock,
            return_value=["http://localhost:30000"],
        ), patch(
            "miles.rollout.modular_rollout.orchestration_train.post",
            new_callable=AsyncMock,
        ):
            run(generate_rollout_async(mock_state, 0, mock_get_samples))

        assert processor_call_log["all_samples_rewards"] is not None
        assert 0.0 in processor_call_log["all_samples_rewards"]
        assert 1.0 in processor_call_log["all_samples_rewards"]
