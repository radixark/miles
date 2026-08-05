"""_resolve_reward_config precedence: reward_spec fields win when set, and
unset fields fall through to sample metadata and process-wide args. A
reward_spec with empty fields (an adapter config without its own rm_type)
must not disable the fallbacks."""

from types import SimpleNamespace

from miles.rollout.rm_hub import _resolve_reward_config
from miles.utils.types import RewardSpec, Sample


def _args(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(rm_type=None, custom_rm_path=None, **{k: v for k, v in kwargs.items()})


def test_spec_fields_win_over_args():
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type="math", custom_rm_path="pkg.fn"))
    args = SimpleNamespace(rm_type="deepscaler", custom_rm_path="other.fn")
    assert _resolve_reward_config(args, sample) == ("pkg.fn", "math")


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
