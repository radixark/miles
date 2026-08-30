from unittest.mock import AsyncMock

import pytest
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.rollout.rollout_manager import RolloutManager, _get_rollout_offload_tags


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (["kv_cache"], (GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE)),
        (["weight"], (GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_WEIGHTS)),
        (
            ["kv_cache", "weight"],
            (
                GPU_MEMORY_TYPE_CUDA_GRAPH,
                GPU_MEMORY_TYPE_KV_CACHE,
                GPU_MEMORY_TYPE_WEIGHTS,
            ),
        ),
    ],
)
def test_get_rollout_offload_tags(levels, expected):
    assert _get_rollout_offload_tags(levels) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("levels", "weight_tags", "inference_tags"),
    [
        (
            ["kv_cache"],
            None,
            [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
        ),
        (
            ["weight"],
            [GPU_MEMORY_TYPE_WEIGHTS],
            [GPU_MEMORY_TYPE_CUDA_GRAPH],
        ),
    ],
)
async def test_onload_only_restores_configured_allocations(levels, weight_tags, inference_tags):
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager._offload_tags = _get_rollout_offload_tags(levels)
    manager.onload = AsyncMock()

    await manager.onload_weights()
    if weight_tags is None:
        manager.onload.assert_not_awaited()
    else:
        manager.onload.assert_awaited_once_with(tags=weight_tags)

    manager.onload.reset_mock()
    await manager.onload_kv()
    manager.onload.assert_awaited_once_with(tags=inference_tags)
