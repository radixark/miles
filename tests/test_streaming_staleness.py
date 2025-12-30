import asyncio
import time
from types import SimpleNamespace

from miles.rollout.streaming_rollout_manager import CompletedGroup, StreamingGroupBuffer, StreamingRolloutManager
from miles.utils.types import Sample


class _DummyDataSource:
    def get_samples(self, _num_samples: int):
        return []


def test_staleness_drop_on_dequeue():
    args = SimpleNamespace(
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=1,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
    )

    mgr = StreamingRolloutManager(
        args,
        _DummyDataSource(),
        engine_urls=["http://e0"],
        groups_per_train_step=1,
        queue_target=2,
        queue_cap=16,
        inflight_target=1,
        min_active_engines=1,
        weight_update_mode="rolling_drain",
    )

    async def _run():
        now = time.time()
        await mgr.buffer.put(CompletedGroup(behavior_version=0, finished_ts=now, engine_idx=0, group=[Sample(reward=0)]))
        await mgr.buffer.put(CompletedGroup(behavior_version=3, finished_ts=now, engine_idx=0, group=[Sample(reward=0)]))
        await mgr.buffer.put(CompletedGroup(behavior_version=4, finished_ts=now, engine_idx=0, group=[Sample(reward=0)]))

        groups, _extra = await mgr.get_next_groups(num_groups=1, target_version=5, max_staleness_versions=1)
        assert len(groups) == 1
        assert groups[0].behavior_version == 4

    asyncio.run(_run())


def test_priority_queue_tie_does_not_crash():
    async def _run():
        buf = StreamingGroupBuffer(queue_cap=16)
        now = 123.0

        await buf.put(CompletedGroup(behavior_version=0, finished_ts=now, engine_idx=0, group=[Sample(reward=0)]))
        await buf.put(CompletedGroup(behavior_version=0, finished_ts=now, engine_idx=1, group=[Sample(reward=0)]))

        g1 = await buf.get()
        g2 = await buf.get()
        assert {g1.engine_idx, g2.engine_idx} == {0, 1}

    asyncio.run(_run())


if __name__ == "__main__":
    test_staleness_drop_on_dequeue()
    test_priority_queue_tie_does_not_crash()
