import asyncio

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.rollout import sglang_rollout
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.args.runtime import RolloutConfig


@pytest.mark.parametrize("legacy", [False, True])
async def test_generation_states_sample_the_current_engine_count_on_each_acquisition(
    legacy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both generation paths use live engine counts, retaining one slot when no engines are observed."""
    args = RolloutConfig.slice_from(parse_megatron_test_config("--sglang-server-concurrency", "1"))
    module = sglang_rollout if legacy else inference_rollout_common
    monkeypatch.setattr(module, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "load_processor", lambda *_args, **_kwargs: None)
    state = object.__new__(module.GenerateState)
    module.GenerateState.__init__(state, args)
    semaphore = state.semaphore if legacy else state.generate_fn_semaphore

    async with semaphore:
        args.inference_runtime_mut_state.engine_count = 2
        assert await asyncio.wait_for(semaphore.__aenter__(), timeout=1) is semaphore
        await semaphore.__aexit__(None, None, None)
        args.inference_runtime_mut_state.engine_count = 0
        waiter = asyncio.create_task(semaphore.__aenter__())
        await asyncio.sleep(0)
        assert not waiter.done()

    assert await asyncio.wait_for(waiter, timeout=1) is semaphore
    await semaphore.__aexit__(None, None, None)
