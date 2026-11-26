from __future__ import annotations

import asyncio

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.ray.rollout.reward_pool import RewardPoolClient
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.worker_handle import WorkerUnreachableError


def _config(*, num_workers: int = 2, batch_size: int = 2, max_wait_ms: float = 0.0) -> RewardPoolConfig:
    return RewardPoolConfig(
        name="pickscore",
        scorer_path="unused.Scorer",
        num_workers=num_workers,
        uses_gpu=True,
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        num_gpus_per_node=8,
    )


class _Handle:
    def __init__(self, cluster: _Cluster, cell_index: int, generation: int) -> None:
        self.cluster, self.cell_index, self.generation = cluster, cell_index, generation

    async def score_batch(self, outputs, prompts):
        await asyncio.sleep(0)
        if self.generation != self.cluster.generation[self.cell_index]:
            raise WorkerUnreachableError("stale actor")
        if self.cell_index in self.cluster.dead:
            raise WorkerUnreachableError("actor died")
        self.cluster.calls.append((self.cell_index, list(prompts)))
        if prompts == ["short", "short"]:
            return [1.0]
        return [None if prompt == "unscorable" else len(prompt) for prompt in prompts]

    async def wait_ready(self, *, timeout):
        pass


class _Cluster:
    """Stands in for the worker manager: handles per cell generation, suspend/resume bump the generation."""

    def __init__(self, num_workers: int) -> None:
        self.generation = [0] * num_workers
        self.dead: set[int] = set()
        self.torn_down: set[int] = set()
        self.calls: list[tuple[int, list[str]]] = []
        self.operations: list[tuple[str, str]] = []

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[str]]:
        """A cell the manager's liveness scan tore down lists no worker, as RayWorkerManager.get_worker_infos does."""
        return [
            [] if int(cell_id.rsplit("-", 1)[1]) in self.torn_down else [f"{cell_id}-00000"] for cell_id in cell_ids
        ]

    def get_handles_of_worker_infos(self, infos: list[str]) -> dict[str, _Handle]:
        handles = {}
        for name in infos:
            _, cell_index, _ = parse_worker_name(name)
            handles[name] = _Handle(self, cell_index, self.generation[cell_index])
        return handles

    async def suspend(self, *, cell_id: str) -> None:
        self.operations.append(("suspend", cell_id))

    async def resume(self, *, cell_id: str) -> None:
        self.operations.append(("resume", cell_id))
        cell_index = int(cell_id.rsplit("-", 1)[1])
        self.generation[cell_index] += 1
        self.dead.discard(cell_index)
        self.torn_down.discard(cell_index)


def _client(cluster: _Cluster, config: RewardPoolConfig, **kwargs) -> RewardPoolClient:
    capability = FakeBackendCapability(static_provider=cluster, cell_operations=cluster)
    return RewardPoolClient(config=config, capability=capability, **kwargs)


async def test_batches_go_round_robin_across_calls_and_scores_keep_input_order():
    """Same dealing as the actor pool it replaces: chunks of batch_size, one counter that runs across calls."""
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())

    first = await client.score([None] * 3, ["a", "bb", "ccc"])
    second = await client.score([None] * 2, ["dddd", "e"])

    assert first == [1.0, 2.0, 3.0] and second == [4.0, 1.0]
    assert sorted(cluster.calls) == [(0, ["a", "bb"]), (0, ["dddd", "e"]), (1, ["ccc"])]


async def test_a_scorer_none_stays_none_so_the_missing_reward_filter_can_drop_it():
    client = _client(_Cluster(num_workers=1), _config(num_workers=1))

    assert await client.score([None] * 3, ["a", "unscorable", "ccc"]) == [1.0, None, 3.0]


async def test_a_worker_that_died_is_restarted_and_scores_the_same_batch():
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())
    await client.score([None] * 2, ["a", "bb"])
    cluster.dead.add(1)

    scores = await client.score([None] * 2, ["xyz", "q"])

    assert scores == [3.0, 1.0]
    assert cluster.operations == [("suspend", "reward-pickscore-00001"), ("resume", "reward-pickscore-00001")]
    assert cluster.calls[-1] == (1, ["xyz", "q"])


async def test_two_batches_that_find_the_same_worker_dead_restart_it_once():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, batch_size=1))
    await client.score([None], ["a"])
    cluster.dead.add(0)

    scores = await client.score([None] * 2, ["bb", "ccc"])

    assert scores == [2.0, 3.0]
    assert [op for op, _ in cluster.operations] == ["suspend", "resume"]


async def test_a_worker_that_keeps_dying_fails_the_call_after_the_restart_budget():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1), max_restarts_per_call=2)

    async def always_dead(*, cell_id):
        cluster.operations.append(("resume", cell_id))

    cluster.resume = always_dead
    cluster.dead.add(0)

    with pytest.raises(WorkerUnreachableError):
        await client.score([None], ["a"])
    assert [op for op, _ in cluster.operations].count("resume") == 2


async def test_concurrent_single_image_calls_share_a_batch_once_it_fills():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, batch_size=3, max_wait_ms=60_000))

    scores = await asyncio.gather(*(client.score([None], [prompt]) for prompt in ["a", "bb", "ccc"]))

    assert scores == [[1.0], [2.0], [3.0]]
    assert cluster.calls == [(0, ["a", "bb", "ccc"])]


async def test_a_partial_batch_is_sent_after_max_wait_ms():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, batch_size=8, max_wait_ms=20))

    scores = await asyncio.wait_for(asyncio.gather(client.score([None], ["a"]), client.score([None], ["bb"])), 5)

    assert scores == [[1.0], [2.0]]
    assert cluster.calls == [(0, ["a", "bb"])]


async def test_without_max_wait_ms_each_call_is_sent_on_its_own():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, batch_size=8))

    await asyncio.gather(client.score([None], ["a"]), client.score([None], ["bb"]))

    assert sorted(cluster.calls) == [(0, ["a"]), (0, ["bb"])]


async def test_coalesced_scores_go_back_to_their_own_calls_in_order_across_batches():
    """A call larger than a batch fills one and starts the next; batches go round robin."""
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config(batch_size=2, max_wait_ms=20))

    first, second = await asyncio.gather(
        client.score([None] * 3, ["a", "bb", "ccc"]), client.score([None] * 2, ["dddd", "unscorable"])
    )

    assert first == [1.0, 2.0, 3.0] and second == [4.0, None]
    assert cluster.calls == [(0, ["a", "bb"]), (1, ["ccc", "dddd"]), (0, ["unscorable"])]


async def test_a_coalesced_batch_whose_worker_keeps_dying_fails_every_call_in_it():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, max_wait_ms=20), max_restarts_per_call=1)

    async def always_dead(*, cell_id):
        cluster.operations.append(("resume", cell_id))

    cluster.resume = always_dead
    cluster.dead.add(0)

    results = await asyncio.gather(client.score([None], ["a"]), client.score([None], ["b"]), return_exceptions=True)

    assert all(isinstance(result, WorkerUnreachableError) for result in results)


async def test_a_coalesced_batch_whose_worker_died_is_restarted_and_resent():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, max_wait_ms=20))
    await client.score([None], ["warm"])
    cluster.dead.add(0)

    scores = await asyncio.gather(client.score([None], ["a"]), client.score([None], ["bb"]))

    assert scores == [[1.0], [2.0]]
    assert cluster.calls[-1] == (0, ["a", "bb"])
    assert [op for op, _ in cluster.operations] == ["suspend", "resume"]


async def test_a_scorer_that_returns_the_wrong_number_of_scores_fails_the_call():
    client = _client(_Cluster(num_workers=1), _config(num_workers=1))

    with pytest.raises(ValueError, match="returned 1 scores for 2 outputs"):
        await client.score([None] * 2, ["short", "short"])
