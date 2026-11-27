import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout import fully_async_rollout
from miles.rollout.base_types import RolloutFnEvalInput
from miles.utils.eval_config import EvalDatasetConfig


def test_due_every_interval_plus_first_and_last():
    cfg = EvalDatasetConfig(name="long", path="x.jsonl", interval=3)
    assert [r for r in range(8) if cfg.due(r, 8)] == [0, 2, 5, 7]
    assert cfg.due(None, 8)


def test_interval_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        EvalDatasetConfig(name="long", path="x.jsonl", interval=0)


@pytest.mark.parametrize("own_engines", [True, False])
def test_fully_async_eval_passes_the_rollout_id(monkeypatch, own_engines):
    seen = {}

    async def fake_run_eval_datasets(state, cache, *, kv_cache_namespace=None, rollout_id=None):
        seen["rollout_id"] = rollout_id
        return {}

    monkeypatch.setattr(fully_async_rollout, "run_eval_datasets", fake_run_eval_datasets)
    fn = object.__new__(fully_async_rollout.FullyAsyncRolloutFn)
    fn.args = SimpleNamespace(namespaced_radix_cache=False)
    fn.state = object()
    fn._eval_prompt_dataset_cache = {}
    fn._producer_resumed = asyncio.Event()
    state = object() if own_engines else None
    asyncio.run(fn._call_eval(RolloutFnEvalInput(rollout_id=7, generate_state=state)))
    assert seen["rollout_id"] == 7
