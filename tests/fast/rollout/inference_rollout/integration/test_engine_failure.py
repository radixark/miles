import contextlib

import httpx
import pytest
from fastapi import HTTPException
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils import mock_tools
from miles.utils.test_utils.mock_tools import TwoTurnStub

_DROPPED = "rollout/dynamic_filter/drop_group_has_aborted_ModelCallFailed"


def _config(version: str) -> RolloutEnvConfig:
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + extra_argv_for_variant("agentic_tool_call", custom_agent_function_path=f"{__name__}._agent_exiting_cleanly")
        + ["--use-session-server", version, "--custom-rm-path", f"{__name__}._answered"],
        data_rows=[{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}],
    )


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(v), id=v) for v in ("v1", "v2")], indirect=True)
@pytest.mark.parametrize(
    ("attempts", "then_reject", "rewards", "dropped"),
    [
        pytest.param(1, False, [1], 1, id="last_call_500"),
        pytest.param(2, False, [1], None, id="500_then_200"),
        pytest.param(1, True, [0], None, id="500_then_400"),
    ],
)
def test_engine_failure_on_the_last_call_discards_the_episode(
    rollout_env, monkeypatch, attempts, then_reject, rewards, dropped
):
    failures = [1]

    def engine(prompt: str):
        if prompt == TwoTurnStub.SECOND_PROMPT and failures[0]:
            failures[0] -= 1
            raise HTTPException(status_code=500, detail="engine failed")
        return TwoTurnStub.process_fn(prompt)

    async def post_with_attempts(url, payload, **kwargs):
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(attempts):
                response = await client.post(url, json=payload)
                if response.status_code < 500:
                    break
            if then_reject and response.status_code >= 500:
                response = await client.post(url, content=b"{")
        response.raise_for_status()
        return response.json()

    rollout_env.mock_server.process_fn = engine
    monkeypatch.setattr(mock_tools, "post", post_with_attempts)
    out = load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train")

    assert failures == [0]
    flat = [s for group in out.samples for sub in group for s in (sub if isinstance(sub, list) else [sub])]
    assert [sample.reward for sample in flat] == rewards
    assert out.metrics.get(_DROPPED) == dropped


async def _agent_exiting_cleanly(**kwargs):
    with contextlib.suppress(httpx.HTTPStatusError):
        await mock_tools.run_agentic_tool_call(**kwargs)


async def _answered(args, samples):
    if isinstance(samples, list):
        return [float(sample.label in sample.response) for sample in samples]
    return float(samples.label in samples.response)
