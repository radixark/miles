import asyncio
from collections.abc import Callable

import openenv_agent_function as oaf
import openenv_generate
import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.rollout.base_types import RolloutFnOutput
from miles.utils.test_utils.mock_sglang_server import ProcessResult


async def _run_agent(**kwargs):
    return await oaf.run(**kwargs)


def _config() -> RolloutEnvConfig:
    argv = extra_argv_for_variant("agentic_tool_call", custom_agent_function_path=f"{__name__}._run_agent")
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + argv
        + [
            "--rollout-batch-size",
            "1",
            "--n-samples-per-prompt",
            "2",
            "--custom-rm-path",
            f"{openenv_generate.__name__}.reward_func",
        ],
        data_rows=[{"input": "What is 1+7?", "label": "8"}],
    )


class _Action:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


class _Observation:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Result:
    def __init__(self, reward=None, **fields):
        self.reward = reward
        self.observation = _Observation(**fields)


def _env_class(first_evaluate: Callable):
    evaluations = []

    class _Env:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def reset(self, task_id=None):
            return _Result(instruction="do it")

        async def step(self, action):
            evaluations.append(action.action_type)
            if len(evaluations) == 1:
                return await first_evaluate()
            return _Result(reward=1.0, info={"harness": "tests/test.sh"})

    return _Env


async def _scoring_error():
    return _Result(reward=None, error="scoring failed")


async def _non_canonical_verdict():
    return _Result(reward=1.0, info={"exit_code": 0})


async def _env_lost():
    raise RuntimeError("env connection lost")


async def _evaluate_stalls():
    await asyncio.sleep(3600)


def _rollout(rollout_env, monkeypatch, first_evaluate: Callable) -> RolloutFnOutput:
    env = _env_class(first_evaluate)
    monkeypatch.setattr(oaf, "load_tbench2", lambda: {"env": env, "action": _Action})
    rollout_env.mock_server.process_fn = lambda prompt: ProcessResult(text="TASK_COMPLETE", finish_reason="stop")
    return load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train")


def _kept_rewards(out: RolloutFnOutput) -> list[float]:
    return sorted(sample.reward for group in out.samples for sub in group for sample in sub)


def _drops(out: RolloutFnOutput) -> dict[str, int]:
    return {key: count for key, count in out.metrics.items() if "dynamic_filter/drop_" in key}


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
@pytest.mark.parametrize(
    ("first_evaluate", "cause"),
    [(_scoring_error, "NoVerdict"), (_non_canonical_verdict, "NoVerdict"), (_env_lost, "EpisodeFailed")],
)
def test_episode_without_a_verdict_drops_the_group(rollout_env, monkeypatch, first_evaluate, cause):
    out = _rollout(rollout_env, monkeypatch, first_evaluate)

    assert _kept_rewards(out) == [1.0, 1.0]
    assert _drops(out) == {f"rollout/dynamic_filter/drop_group_has_aborted_{cause}": 1}


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
def test_timeout_stays_scored_zero(rollout_env, monkeypatch):
    monkeypatch.setattr(oaf, "MAX_ROLLOUT_TIME_S", 3.0)

    out = _rollout(rollout_env, monkeypatch, _evaluate_stalls)

    assert _kept_rewards(out) == [0.0, 1.0]
    assert _drops(out) == {}
