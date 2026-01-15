from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.modular_rollout.orchestration_common import GenerateState, generate_and_rm, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.types import Sample


class TestNonGroupRM:
    @pytest.fixture
    def mock_state(self, mock_args):
        with patch("miles.rollout.modular_rollout.orchestration_common.load_tokenizer"), patch(
            "miles.rollout.modular_rollout.orchestration_common.load_processor"
        ):
            state = GenerateState(mock_args)
            state.generate_function = AsyncMock(
                return_value=GenerateFnOutput(
                    samples=Sample(
                        prompt="test",
                        response="\\boxed{8}",
                        label="8",
                        status=Sample.Status.COMPLETED,
                    )
                )
            )
            return state

    def test_async_rm_called_for_single_sample(self, mock_state):
        mock_state.args.group_rm = False
        sample = Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING)

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.async_rm",
            new_callable=AsyncMock,
        ) as mock_async_rm:
            mock_async_rm.return_value = 1.0
            result = run(generate_and_rm(mock_state, sample, {"temperature": 0.7}, evaluation=False))
            mock_async_rm.assert_called_once()
            assert result.reward == 1.0

    def test_batched_async_rm_called_for_multi_samples(self, mock_state):
        mock_state.args.group_rm = False
        samples = [
            Sample(prompt="test", response="\\boxed{8}", label="8", status=Sample.Status.COMPLETED),
            Sample(prompt="test", response="\\boxed{8}", label="8", status=Sample.Status.COMPLETED),
        ]
        mock_state.generate_function = AsyncMock(return_value=GenerateFnOutput(samples=samples))

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            new_callable=AsyncMock,
        ) as mock_batched_rm:
            sample = Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING)
            run(generate_and_rm(mock_state, sample, {"temperature": 0.7}, evaluation=False))
            mock_batched_rm.assert_called_once()


class TestGroupRM:
    @pytest.fixture
    def mock_state(self, mock_args):
        mock_args.group_rm = True
        with patch("miles.rollout.modular_rollout.orchestration_common.load_tokenizer"), patch(
            "miles.rollout.modular_rollout.orchestration_common.load_processor"
        ):
            state = GenerateState(mock_args)
            state.generate_function = AsyncMock(
                return_value=GenerateFnOutput(
                    samples=Sample(
                        prompt="test",
                        response="\\boxed{8}",
                        label="8",
                        status=Sample.Status.COMPLETED,
                    )
                )
            )
            return state

    def test_async_rm_not_called_when_group_rm(self, mock_state):
        sample = Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING)

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.async_rm",
            new_callable=AsyncMock,
        ) as mock_async_rm:
            result = run(generate_and_rm(mock_state, sample, {"temperature": 0.7}, evaluation=False))
            mock_async_rm.assert_not_called()
            assert result.reward is None

    def test_batched_async_rm_called_in_group(self, mock_state):
        group = [
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
        ]

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.async_rm",
            new_callable=AsyncMock,
        ) as mock_async_rm, patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            new_callable=AsyncMock,
        ) as mock_batched_rm:
            run(generate_and_rm_group(mock_state, group, {"temperature": 0.7}, evaluation=False))
            mock_async_rm.assert_not_called()
            mock_batched_rm.assert_called_once()
            call_args = mock_batched_rm.call_args
            assert len(call_args[0][1]) == 2


class TestDeterministicInference:
    @pytest.fixture
    def mock_state(self, mock_args):
        with patch("miles.rollout.modular_rollout.orchestration_common.load_tokenizer"), patch(
            "miles.rollout.modular_rollout.orchestration_common.load_processor"
        ):
            state = GenerateState(mock_args)
            state.generate_function = AsyncMock(
                return_value=GenerateFnOutput(
                    samples=Sample(
                        prompt="test",
                        response="\\boxed{8}",
                        label="8",
                        status=Sample.Status.COMPLETED,
                    )
                )
            )
            return state

    def test_sampling_seed_set_when_enabled(self, mock_state):
        mock_state.args.sglang_enable_deterministic_inference = True
        mock_state.args.rollout_seed = 42
        mock_state.args.group_rm = True

        group = [
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
        ]

        captured_params = []

        async def capture_generate(input):
            captured_params.append(input.sampling_params.copy())
            return GenerateFnOutput(
                samples=Sample(
                    prompt="test",
                    response="\\boxed{8}",
                    label="8",
                    status=Sample.Status.COMPLETED,
                )
            )

        mock_state.generate_function = capture_generate

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            new_callable=AsyncMock,
        ):
            run(generate_and_rm_group(mock_state, group, {"temperature": 0.7}, evaluation=False))

        seeds = [p.get("sampling_seed") for p in captured_params]
        assert set(seeds) == {42, 43, 44}

    def test_sampling_seed_not_set_when_disabled(self, mock_state):
        mock_state.args.sglang_enable_deterministic_inference = False
        mock_state.args.group_rm = True

        group = [
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
            Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING),
        ]

        captured_params = []

        async def capture_generate(input):
            captured_params.append(input.sampling_params.copy())
            return GenerateFnOutput(
                samples=Sample(
                    prompt="test",
                    response="\\boxed{8}",
                    label="8",
                    status=Sample.Status.COMPLETED,
                )
            )

        mock_state.generate_function = capture_generate

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            new_callable=AsyncMock,
        ):
            run(generate_and_rm_group(mock_state, group, {"temperature": 0.7}, evaluation=False))

        seeds = [p.get("sampling_seed") for p in captured_params]
        assert all(seed is None for seed in seeds)


class TestMultiSampleOutput:
    @pytest.fixture
    def mock_state(self, mock_args):
        mock_args.group_rm = False
        with patch("miles.rollout.modular_rollout.orchestration_common.load_tokenizer"), patch(
            "miles.rollout.modular_rollout.orchestration_common.load_processor"
        ):
            state = GenerateState(mock_args)
            return state

    def test_multi_sample_output_partial_reward(self, mock_state):
        s1 = Sample(
            prompt="test",
            response="\\boxed{8}",
            label="8",
            reward=None,
            status=Sample.Status.COMPLETED,
        )
        s2 = Sample(
            prompt="test",
            response="\\boxed{8}",
            label="8",
            reward=0.5,
            status=Sample.Status.COMPLETED,
        )
        mock_state.generate_function = AsyncMock(return_value=GenerateFnOutput(samples=[s1, s2]))

        sample = Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING)

        async def mock_batched_rm(args, samples, inplace_set_reward_field=False):
            if inplace_set_reward_field:
                for s in samples:
                    if s.reward is None:
                        s.reward = 1.0
                return None
            return [1.0] * len(samples)

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            side_effect=mock_batched_rm,
        ):
            result = run(generate_and_rm(mock_state, sample, {"temperature": 0.7}, evaluation=False))

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].reward == 1.0
        assert result[1].reward == 0.5

    def test_multi_sample_output_aborted_skips_rm(self, mock_state):
        s1 = Sample(
            prompt="test",
            response="\\boxed{8}",
            label="8",
            reward=None,
            status=Sample.Status.ABORTED,
        )
        s2 = Sample(
            prompt="test",
            response="\\boxed{8}",
            label="8",
            reward=None,
            status=Sample.Status.COMPLETED,
        )
        mock_state.generate_function = AsyncMock(return_value=GenerateFnOutput(samples=[s1, s2]))

        sample = Sample(prompt="test", response="", label="8", status=Sample.Status.PENDING)

        with patch(
            "miles.rollout.modular_rollout.orchestration_common.batched_async_rm",
            new_callable=AsyncMock,
        ) as mock_batched_rm:
            result = run(generate_and_rm(mock_state, sample, {"temperature": 0.7}, evaluation=False))

        mock_batched_rm.assert_not_called()
        assert isinstance(result, list)
