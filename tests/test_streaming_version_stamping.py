import asyncio
from types import SimpleNamespace

from miles.rollout.streaming_rollout_manager import CompletedGroup, StreamingGroupBuffer, StreamingRolloutManager
from miles.utils.types import Sample


def test_group_buffer_orders_by_behavior_version_then_time():
    async def _run():
        buf = StreamingGroupBuffer(queue_cap=10)

        await buf.put(
            CompletedGroup(
                behavior_version=5,
                finished_ts=2.0,
                group=[Sample(prompt="p")],
            )
        )
        await buf.put(
            CompletedGroup(
                behavior_version=2,
                finished_ts=3.0,
                group=[Sample(prompt="p")],
            )
        )

        first = await buf.get()
        second = await buf.get()
        assert first.behavior_version == 2
        assert second.behavior_version == 5

    asyncio.run(_run())


def test_streaming_rollout_manager_stamps_weight_version_first():
    async def _run():
        import sys
        import types

        module_name = "miles.rollout.sglang_rollout"
        original_module = sys.modules.get(module_name)
        fake_module = types.ModuleType(module_name)

        async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
            return group

        fake_module.generate_and_rm_group = fake_generate_and_rm_group
        sys.modules[module_name] = fake_module

        try:
            args = SimpleNamespace(
                rollout_temperature=1.0,
                rollout_top_p=1.0,
                rollout_top_k=-1,
                rollout_max_response_len=16,
                rollout_stop=None,
                rollout_stop_token_ids=None,
                rollout_skip_special_tokens=True,
            )

            class FakeDataSource:
                def __init__(self):
                    self._samples = [[Sample(prompt="p")]]

                def get_samples(self, _n: int):
                    if not self._samples:
                        return []
                    return [self._samples.pop(0)]

            mgr = StreamingRolloutManager(
                args,
                FakeDataSource(),
                groups_per_train_step=1,
                queue_target=4,
                queue_cap=8,
                inflight_target=4,
                initial_published_version=0,
            )
            mgr.notify_new_version(7)
            mgr.start()
            groups, _extra = await asyncio.wait_for(mgr.get_next_groups(num_groups=1, target_version=7), timeout=2.0)
            await mgr.stop()

            assert groups[0].behavior_version == 7
            assert groups[0].group[0].metadata["weight_version_first"] == 7
        finally:
            if original_module is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_module

    asyncio.run(_run())


if __name__ == "__main__":
    test_group_buffer_orders_by_behavior_version_then_time()
    test_streaming_rollout_manager_stamps_weight_version_first()

