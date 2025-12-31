import asyncio
import time
from types import SimpleNamespace

from miles.rollout.streaming_rollout_manager import CompletedGroup, StreamingGroupBuffer, StreamingRolloutManager
from miles.utils.types import Sample


class _DummyDataSource:
    def get_samples(self, _num_samples: int):
        return []


def test_queue_orders_by_behavior_version_without_dropping():
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
        groups_per_train_step=1,
        queue_target=2,
        queue_cap=16,
        inflight_target=1,
    )

    async def _run():
        now = time.time()
        await mgr.buffer.put(CompletedGroup(behavior_version=0, finished_ts=now, group=[Sample(reward=0)]))
        await mgr.buffer.put(CompletedGroup(behavior_version=3, finished_ts=now, group=[Sample(reward=0)]))
        await mgr.buffer.put(CompletedGroup(behavior_version=4, finished_ts=now, group=[Sample(reward=0)]))

        groups, extra = await mgr.get_next_groups(num_groups=3, target_version=5)
        assert [g.behavior_version for g in groups] == [0, 3, 4]
        assert extra["staleness_values"] == [5, 2, 1]

    asyncio.run(_run())


def test_priority_queue_tie_does_not_crash():
    async def _run():
        buf = StreamingGroupBuffer(queue_cap=16)
        now = 123.0

        await buf.put(CompletedGroup(behavior_version=0, finished_ts=now, group=[Sample(reward=0)]))
        await buf.put(CompletedGroup(behavior_version=0, finished_ts=now, group=[Sample(reward=1)]))

        g1 = await buf.get()
        g2 = await buf.get()
        assert sorted([g1.group[0].reward, g2.group[0].reward]) == [0, 1]

    asyncio.run(_run())


if __name__ == "__main__":
    test_queue_orders_by_behavior_version_without_dropping()
    test_priority_queue_tie_does_not_crash()
