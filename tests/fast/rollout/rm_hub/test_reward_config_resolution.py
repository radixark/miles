"""_resolve_reward_config precedence: reward_spec fields win when set, and
unset fields fall through to sample metadata and process-wide args. A
reward_spec with empty fields (an adapter config without its own rm_type)
must not disable the fallbacks."""

from types import SimpleNamespace

from miles.rollout.rm_hub import _resolve_reward_config
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import RewardSpec, Sample


async def _untyped_reward(args, sample):
    return 1.0


def _args(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(rm_type=None, custom_rm_path=None, **{k: v for k, v in kwargs.items()})


def test_spec_fields_win_over_args():
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type="math", custom_rm_path="pkg.fn"))
    args = SimpleNamespace(rm_type="deepscaler", custom_rm_path="other.fn")
    with function_registry.temporary("pkg.fn", _untyped_reward):
        assert _resolve_reward_config(args, sample) == (CustomFunctionConfig(path="pkg.fn"), "math")


def test_empty_spec_falls_back_to_args():
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type=None, custom_rm_path=None))
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "math")


def test_empty_spec_falls_back_to_sample_metadata_before_args():
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type=None, custom_rm_path=None))
    sample.metadata = {"rm_type": "gpqa"}
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "gpqa")


def test_no_spec_keeps_metadata_then_args_chain():
    sample = Sample(prompt="p")
    sample.metadata = {}
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "math")


def test_metadata_custom_rm_path_is_not_a_reward_channel():
    """reward_spec is the per-sample contract; a row's metadata.custom_rm_path is just data."""
    sample = Sample(prompt="p", metadata={"custom_rm_path": "row.fn"})
    args = SimpleNamespace(rm_type=None, custom_rm_path="global.fn")
    assert _resolve_reward_config(args, sample) == (CustomFunctionConfig(path="global.fn"), "")


def test_empty_spec_custom_rm_path_falls_back_to_args():
    sample = Sample(prompt="p", reward_spec=RewardSpec(custom_rm_path=""))
    args = SimpleNamespace(rm_type=None, custom_rm_path="global.fn")
    assert _resolve_reward_config(args, sample) == (CustomFunctionConfig(path="global.fn"), "")
