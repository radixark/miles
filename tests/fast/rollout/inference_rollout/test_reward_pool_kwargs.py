"""The run's reward pools reach a custom reward function as `<name>_pool`, in train and in eval."""

from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.rollout.inference_rollout.conftest import make_state

from miles.ray.rollout import eval_fleet as eval_fleet_mod
from miles.ray.rollout.eval_fleet import RolloutExecutorEvalFleet
from miles.rollout import fully_async_rollout
from miles.rollout.base_types import GenerateFnOutput, RolloutFnConstructorInput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample

_POOL = object()
_calls: list[dict] = []


async def recording_rm(args, sample, **kwargs):
    samples = sample if isinstance(sample, list) else [sample]
    _calls.append(kwargs)
    return [1.0] * len(samples) if isinstance(sample, list) else 1.0


async def _completed(input):
    input.sample.status = Sample.Status.COMPLETED
    return GenerateFnOutput(samples=input.sample)


@pytest.fixture
def calls():
    _calls.clear()
    yield _calls
    _calls.clear()


def _state(*argv: str):
    args = parse_fsdp_test_config("--custom-rm-path", f"{__name__}.recording_rm", *argv)
    state = make_state(_completed, args=args)
    state.reward_kwargs = {"hpsv3_pool": _POOL}
    return state


@pytest.mark.parametrize("evaluation", [False, True])
async def test_a_per_sample_reward_gets_each_pool(calls, evaluation):
    sample = await common.generate_and_rm(_state(), Sample(index=0, prompt="p"), {}, evaluation=evaluation)

    assert sample.reward == 1.0
    assert calls == [{"hpsv3_pool": _POOL}]


async def test_a_group_reward_gets_each_pool(calls):
    state = _state("--group-rm")

    group = await common.generate_and_rm_group(state, [Sample(index=i, prompt="p") for i in range(2)], {})

    assert [sample.reward for sample in group] == [1.0, 1.0]
    assert calls == [{"hpsv3_pool": _POOL}]


def test_the_state_names_each_pool_name_pool(monkeypatch):
    monkeypatch.setattr(common, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(common, "load_processor", lambda *a, **k: None)
    args = parse_fsdp_test_config()

    assert common.GenerateState(args, reward_pools={"hpsv3": _POOL}).reward_kwargs == {"hpsv3_pool": _POOL}
    assert common.GenerateState(args).reward_kwargs == {}


@pytest.mark.parametrize("module, cls", [(common, "InferenceRolloutFn"), (fully_async_rollout, "FullyAsyncRolloutFn")])
def test_a_rollout_fn_builds_its_state_on_the_run_pools(monkeypatch, module, cls):
    built = []
    monkeypatch.setattr(module, "GenerateState", lambda args, **kwargs: built.append(kwargs) or SimpleNamespace())
    args = parse_fsdp_test_config()

    getattr(module, cls)(RolloutFnConstructorInput(args=args, data_source=None, reward_pools={"hpsv3": _POOL}))

    assert built == [{"reward_pools": {"hpsv3": _POOL}}]


def test_the_eval_fleet_state_scores_on_the_run_pools(monkeypatch):
    built = []
    monkeypatch.setattr(eval_fleet_mod, "GenerateState", lambda args, **kwargs: built.append(kwargs))
    monkeypatch.setattr(eval_fleet_mod, "retarget_args", lambda **kwargs: kwargs["args"])
    provider = SimpleNamespace(get_handle=lambda name: None)
    info = SimpleNamespace(router=SimpleNamespace(host="h", port=1), engine_gpu_counts=[1])

    RolloutExecutorEvalFleet(None, info=info, inference_controller_provider=provider, reward_pools={"hpsv3": _POOL})

    assert built == [{"reward_pools": {"hpsv3": _POOL}}]
