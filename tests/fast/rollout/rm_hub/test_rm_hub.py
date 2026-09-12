import argparse
import asyncio
import importlib
import sys
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.arguments import _validate_reward_args, get_miles_extra_args_provider, miles_validate_args
from miles.utils.async_utils import run
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.types import RewardSpec, Sample


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = None
    args.rm_url = None
    args.reward_funcs = None
    args.reward_weights = None
    args.reward_key = None
    args.custom_config_path = None
    args.eval_reward_key = None
    args.group_rm = False
    args.hf_checkpoint = "unused"
    args.apply_chat_template = False
    args.chat_template_path = None
    args.rollout_stop = None
    args.rollout_stop_token_ids = None
    args.rollout_skip_special_tokens = True
    args.sglang_enable_deterministic_inference = False
    return args


@pytest.fixture
def custom_rewards(tmp_path, monkeypatch):
    module_name = "reward_weight_test_functions"
    (tmp_path / f"{module_name}.py").write_text(
        dedent(
            """\
            async def score(args, sample, **kwargs):
                return 2.0 if sample.label == "scored" else None

            async def bonus(args, sample, **kwargs):
                return kwargs.get("bonus")

            async def wait_for_peer(args, sample, **kwargs):
                await kwargs["ready"].wait()
                return 2.0

            async def signal_peer(args, sample, **kwargs):
                kwargs["ready"].set()
                return 3.0
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module_name
    sys.modules.pop(module_name, None)


class TestMultipleRewards:
    @pytest.mark.parametrize("weights,expected", [(None, 5 / 3), ([0.5, 0.75], 1.0), ([0.0, -1.5], -1.0)])
    def test_weighted_builtins(self, mock_args, weights, expected):
        mock_args.reward_funcs = ["boxed_f1", "f1"]
        mock_args.reward_weights = weights
        sample = Sample(response=r"42 \boxed{42}", label="42")
        assert run(async_rm(mock_args, sample)) == pytest.approx(expected)

    def test_none_opts_out_per_sample(self, mock_args, custom_rewards):
        mock_args.reward_funcs = [f"{custom_rewards}.score", f"{custom_rewards}.bonus"]
        mock_args.reward_weights = [0.25, 2.0]
        scored = Sample(label="scored")
        skipped = Sample(label="skipped")
        assert run(async_rm(mock_args, scored, bonus=3.0)) == 6.5
        assert run(async_rm(mock_args, skipped, bonus=3.0)) == 6.0
        assert skipped.metadata["reward_components"] == {
            f"{custom_rewards}.score": None,
            f"{custom_rewards}.bonus": 3.0,
        }

    def test_all_none_returns_zero(self, mock_args, custom_rewards):
        mock_args.reward_funcs = [f"{custom_rewards}.score", f"{custom_rewards}.bonus"]
        sample = Sample(label="skipped")
        assert run(async_rm(mock_args, sample)) == 0.0
        assert sample.metadata["reward_components"] == {
            f"{custom_rewards}.score": None,
            f"{custom_rewards}.bonus": None,
        }

    @pytest.mark.parametrize("metadata", [None, {"source": "test", "rm_type": "unknown"}])
    def test_records_unweighted_components(self, mock_args, metadata):
        mock_args.reward_funcs = ["boxed_f1", "f1"]
        mock_args.reward_weights = [0.5, 0.75]
        sample = Sample(response=r"42 \boxed{42}", label="42", metadata=metadata)
        run(async_rm(mock_args, sample))
        assert sample.metadata["reward_components"] == {"boxed_f1": 1.0, "f1": pytest.approx(2 / 3)}
        if metadata is not None:
            assert sample.metadata["source"] == "test"

    def test_custom_functions_run_concurrently(self, mock_args, custom_rewards):
        mock_args.reward_funcs = [f"{custom_rewards}.wait_for_peer", f"{custom_rewards}.signal_peer"]

        async def score():
            return await asyncio.wait_for(async_rm(mock_args, Sample(), ready=asyncio.Event()), timeout=1)

        assert run(score()) == 5.0

    def test_dict_component_and_scalar_reward_value(self, mock_args):
        mock_args.reward_funcs = ["dapo", "boxed_f1"]
        mock_args.reward_weights = [0.25, 0.5]
        mock_args.reward_key = "score"
        sample = Sample(response=r"Answer: \boxed{42}", label="42")
        sample.reward = run(async_rm(mock_args, sample))
        assert sample.reward == {"score": 0.75}
        assert sample.get_reward_value(mock_args) == 0.75
        assert sample.metadata["reward_components"] == {"dapo": 1.0, "boxed_f1": 1.0}

    def test_per_sample_reward_spec_overrides_the_composite(self, mock_args):
        mock_args.reward_funcs = ["boxed_f1", "f1"]
        mock_args.reward_weights = None
        sample = Sample(response=r"\boxed{42}", label="42")
        sample.reward_spec = RewardSpec(rm_type="math")
        assert run(async_rm(mock_args, sample)) == 1
        assert not (sample.metadata or {}).get("reward_components")

    @pytest.mark.parametrize("reward_key", [None, "missing"])
    def test_dict_component_requires_valid_reward_key(self, mock_args, reward_key):
        mock_args.reward_funcs = ["dapo"]
        mock_args.reward_key = reward_key
        with pytest.raises(ValueError, match="dapo.*--reward-key"):
            run(async_rm(mock_args, Sample(response=r"\boxed{42}", label="42")))

    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("weighted,expected", [(True, -0.25), (False, False)])
    def test_eval_handles_scalar_and_dict_rewards(self, mock_args, monkeypatch, legacy, weighted, expected):
        module = importlib.import_module(
            "miles.rollout.sglang_rollout" if legacy else "miles.rollout.inference_rollout.inference_rollout_eval"
        )
        mock_args.reward_key = "score"
        if weighted:
            # The composite is stored under --reward-key, so eval reads the same key.
            mock_args.eval_reward_key = "score"
            mock_args.reward_funcs = ["dapo", "boxed_f1"]
            mock_args.reward_weights = [0.25, 0.5]
        else:
            mock_args.eval_reward_key = "acc"
            mock_args.rm_type = "dapo"
        config = EvalDatasetConfig(name="test", path="unused", n_samples_per_eval_prompt=1)
        cache_key = config.cache_key + (
            mock_args.hf_checkpoint,
            mock_args.apply_chat_template,
            mock_args.chat_template_path,
        )
        cache = {cache_key: SimpleNamespace(samples=[Sample(response=r"\boxed{43}", label="42")])}

        async def generate_and_score(_, sample, **kwargs):
            sample.reward = await async_rm(mock_args, sample, **kwargs)
            return sample

        monkeypatch.setattr(module, "generate_and_rm", generate_and_score)
        monkeypatch.setattr(module, "policy_uses_routing_key", lambda args: False)
        if legacy:
            monkeypatch.setattr(module, "EVAL_PROMPT_DATASET", cache)
            result = run(module.eval_rollout_single_dataset(mock_args, 0, config))
        else:
            monkeypatch.setattr(module, "compute_sampling_params", lambda *args, **kwargs: {})
            result = run(module.eval_rollout_single_dataset(SimpleNamespace(args=mock_args), config, cache))
        assert result["test"]["rewards"] == [expected]

    def test_batched_rewards_are_set_inplace(self, mock_args, custom_rewards):
        mock_args.reward_funcs = [f"{custom_rewards}.score", f"{custom_rewards}.bonus"]
        mock_args.reward_weights = [0.5, 2.0]
        samples = [Sample(label="scored"), Sample(label="skipped")]
        run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True, bonus=3.0))
        assert [sample.reward for sample in samples] == [7.0, 6.0]


class TestRewardArgs:
    @pytest.mark.parametrize("weights", ["0.5", "0.5,1.0,2.0", [0.5]])
    def test_weights_length_validation(self, mock_args, weights):
        mock_args.reward_funcs = "math,f1"
        mock_args.reward_weights = weights
        with pytest.raises(ValueError, match="--reward-weights.*--reward-funcs"):
            miles_validate_args(mock_args)

    def test_eval_reward_key_must_match_reward_key(self, mock_args):
        mock_args.reward_funcs = "dapo,f1"
        mock_args.reward_key = "score"
        mock_args.eval_reward_key = "acc"
        with pytest.raises(ValueError, match="--eval-reward-key must equal --reward-key"):
            miles_validate_args(mock_args)

    @pytest.mark.parametrize("field,value", [("rm_type", "math"), ("custom_rm_path", "pkg.module.fn")])
    @pytest.mark.parametrize("via_config", [False, True])
    def test_mutual_exclusion(self, mock_args, tmp_path, field, value, via_config):
        mock_args.reward_funcs = "math,f1"
        if via_config:
            config = tmp_path / "rewards.yaml"
            config.write_text(f"{field}: {value}\n")
            mock_args.custom_config_path = str(config)
        else:
            setattr(mock_args, field, value)
        with pytest.raises(ValueError, match="--reward-funcs.*mutually exclusive"):
            miles_validate_args(mock_args)

    @pytest.mark.parametrize("weights,expected", [(None, [1.0, 1.0]), ("0.5, -2", [0.5, -2.0])])
    def test_cli_flags_and_weight_defaults(self, monkeypatch, weights, expected):
        argv = ["test", "--rollout-batch-size", "2", "--reward-funcs", " math, pkg.module.fn "]
        if weights is not None:
            argv += ["--reward-weights", weights]
        monkeypatch.setattr(sys, "argv", argv)
        parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
        args = parser.parse_args(argv[1:])
        _validate_reward_args(args)
        assert args.reward_funcs == ["math", "pkg.module.fn"]
        assert args.reward_weights == expected

    @pytest.mark.parametrize("funcs", ["", "math,", ",f1", []])
    def test_empty_functions_rejected(self, mock_args, funcs):
        mock_args.reward_funcs = funcs
        with pytest.raises(ValueError, match="--reward-funcs"):
            miles_validate_args(mock_args)

    def test_weights_require_functions(self, mock_args):
        mock_args.reward_weights = "1.0"
        with pytest.raises(ValueError, match="--reward-weights requires --reward-funcs"):
            miles_validate_args(mock_args)


class TestAsyncRm:
    @pytest.mark.parametrize(
        "rm_type,response,label,expected",
        [
            ("math", r"\boxed{42}", "42", 1),
            ("math", r"\boxed{wrong}", "42", 0),
            ("f1", "hello world", "hello world", 1.0),
            ("dapo", "Answer: 42", "42", {"score": 1.0}),
            ("deepscaler", r"</think>\boxed{42}", "42", 1),
            ("gpqa", "Answer: A", "A", 1.0),
            ("boxed_f1", r"Final answer is \boxed{hello world}", "hello world", 1.0),
        ],
    )
    def test_rm_types(self, mock_args, rm_type, response, label, expected):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response=response, label=label)
        reward = run(async_rm(mock_args, sample))
        if isinstance(expected, dict):
            for k, v in expected.items():
                assert reward[k] == v
        else:
            assert reward == expected

    def test_f1_rm_partial(self, mock_args):
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert 0 < reward < 1

    def test_random_rm(self, mock_args):
        mock_args.rm_type = "random"
        sample = Sample(prompt="", response="anything", label="anything")
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_returns_binary(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello", label="", tokens=[1, 2, 3])
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_is_deterministic(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello world", label="", tokens=[10, 20])
        rewards = [run(async_rm(mock_args, sample)) for _ in range(5)]
        assert len(set(rewards)) == 1

    def test_deterministic_random_rm_differs_by_response(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response=f"response_{i}", label="", tokens=[1, 2, 3]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_deterministic_random_rm_differs_by_tokens(self, mock_args):
        """Same response with different tokens yields both reward values across many samples."""
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response="same", label="", tokens=[i, i + 1, i + 2]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_rm_type_from_metadata(self, mock_args):
        mock_args.rm_type = None
        sample = Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"})
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    @pytest.mark.parametrize(
        "rm_type,match",
        [
            ("unknown_type", "not implemented"),
            ("", "not specified"),
        ],
    )
    def test_invalid_rm_type_raises(self, mock_args, rm_type, match):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match=match):
            run(async_rm(mock_args, sample))


class TestBatchedAsyncRm:
    @pytest.mark.parametrize(
        "rm_type,samples_data,expected",
        [
            (
                "math",
                [(r"\boxed{42}", "42"), (r"\boxed{100}", "100"), (r"\boxed{wrong}", "42")],
                [1, 1, 0],
            ),
            (
                "f1",
                [("hello world", "hello world"), ("different", "something else")],
                [1.0, 0],
            ),
        ],
    )
    def test_batched_rm(self, mock_args, rm_type, samples_data, expected):
        mock_args.rm_type = rm_type
        samples = [Sample(prompt="", response=r, label=label) for r, label in samples_data]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards == expected

    def test_inplace_set_reward_field(self, mock_args):
        mock_args.rm_type = "math"
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42"),
            Sample(prompt="", response=r"\boxed{100}", label="100"),
        ]
        result = run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))
        assert result is None
        assert samples[0].reward == 1
        assert samples[1].reward == 1

    def test_inplace_raises_on_existing_reward(self, mock_args):
        mock_args.rm_type = "math"
        samples = [Sample(prompt="", response=r"\boxed{42}", label="42", reward=0.5)]
        with pytest.raises(AssertionError, match="Overriding"):
            run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))

    def test_empty_samples(self, mock_args):
        mock_args.rm_type = "math"
        rewards = run(batched_async_rm(mock_args, []))
        assert rewards == []

    def test_mixed_rm_types_via_metadata(self, mock_args):
        mock_args.rm_type = None
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"}),
            Sample(prompt="", response="hello", label="hello", metadata={"rm_type": "f1"}),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards[0] == 1
        assert rewards[1] == 1.0
