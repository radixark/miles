from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import NamedHostAndPorts


def _layout_args(**overrides):
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "rollout_num_gpus": 4,
        "eval_num_gpus": 0,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "rollout_external": False,
        "colocate": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("colocate", "expected"),
    [
        (False, (6, 2)),
        (True, (4, 0)),
    ],
)
def test_shared_ppo_counts_actor_bundles_once(colocate, expected):
    assert _get_placement_group_layout(_layout_args(colocate=colocate)) == expected


def test_debug_train_only_counts_actor_bundles_once():
    assert _get_placement_group_layout(_layout_args(debug_train_only=True)) == (2, 0)


def test_external_rollout_only_reserves_no_local_bundles():
    assert _get_placement_group_layout(_layout_args(debug_rollout_only=True, rollout_external=True)) == (0, 0)


class _RecordingRolloutExecutor:
    def __init__(self):
        self.train_parallel_config = None
        self.loaded_rollout_id = None
        self.set_train_parallel_config = SimpleNamespace(remote=self._set_train_parallel_config)
        self.load = SimpleNamespace(remote=self._load)

    async def _set_train_parallel_config(self, config):
        self.train_parallel_config = config

    async def _load(self, rollout_id):
        self.loaded_rollout_id = rollout_id


async def _stop_watch() -> None:
    return None


class _RecordingWorkerProvider(BaseWorkerProvider):
    def __init__(self) -> None:
        self.built_for: list[list[str]] = []
        self.watch_count = 0

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        self.watch_count += 1
        return _stop_watch


async def _fake_init(self: TrainerController) -> list[int]:
    """Stand in for init(), keeping its cell-observation prologue and dropping the GPU work."""
    provider = RayWorkerProvider.create(pool_ids=[self._pool_id])
    self._watcher_disposer = await provider.watch_cells(self._reconcile)
    await self._wait_expected_num_cells()
    return [0]


async def _fake_get_train_parallel_config(self: TrainerController) -> dict:
    return {"dp_size": 2}


def _patch_train_group(monkeypatch) -> list:
    groups = []

    class _Group:
        def __init__(self, *, args, role, **_kwargs):
            self.args = args
            self.role = role
            groups.append(self)

        @classmethod
        async def create(cls, args, **kwargs):
            return cls(args=args, **kwargs)

        async def init(self):
            return [0]

        async def get_train_parallel_config(self):
            return {"dp_size": 2 if self.role == "actor" else 99}

    monkeypatch.setattr(placement_group_module, "TrainerController", _Group)
    return groups


def _training_models_args(**overrides):
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "critic_num_nodes": 1,
        "critic_num_gpus_per_node": 2,
        "use_critic": True,
        "kl_coef": 0.1,
        "use_kl_loss": False,
        "use_opd": True,
        "opd_type": "megatron",
        "disable_param_buffers_cpu_backup": True,
        "start_rollout_id": None,
        "rollout_global_dataset": False,
    }
    values.update(overrides)
    return Namespace(**values)


_waited_roles: list[str] = []


async def _fake_wait_expected_num_cells(self: TrainerController) -> None:
    """The startup barrier waits for the provider to report cells, and this provider reports none.

    Recording the role keeps init()'s call to the barrier under test: deleting that await
    would otherwise go unnoticed, since nothing else drives init() end to end."""
    _waited_roles.append(self._role)


async def test_critic_role_disables_reward_kl_and_preserves_actor_args(monkeypatch):
    """Both training groups go through the real create(), and only the critic args are rewritten."""
    provider = _RecordingWorkerProvider()
    monkeypatch.setattr(TrainerController, "init", _fake_init)
    monkeypatch.setattr(TrainerController, "_wait_expected_num_cells", _fake_wait_expected_num_cells)
    monkeypatch.setattr(TrainerController, "get_train_parallel_config", _fake_get_train_parallel_config)
    _waited_roles.clear()

    args = _training_models_args(indep_dp=False, enable_witness=False)

    def _create(*, pool_ids: list[str] | None = None) -> _RecordingWorkerProvider:
        provider.built_for.append(list(pool_ids or []))
        return provider

    with patch("miles.utils.workers.worker_provider.ray.RayWorkerProvider.create", _create):
        actor, critic = await placement_group_module.create_training_models(
            args,
            inference_controller=object(),
            rollout_executor=_RecordingRolloutExecutor(),
        )

    assert provider.built_for == [["trainer-actor"], ["trainer-critic"]]
    assert provider.watch_count == 2
    assert _waited_roles == ["actor", "critic"]

    assert actor._role == "actor"
    assert actor.args is args
    assert actor.args.kl_coef == 0.1
    # Derived from the actor's kl_coef, not passed in: flip it and the KL term is silently zero.
    assert actor._with_ref is True

    assert critic._role == "critic"
    # A reference model on the critic would be a second full model on its GPUs.
    assert critic._with_ref is False
    assert critic.args is not args
    assert critic.args.kl_coef == 0
    assert critic.args.use_opd is False
    assert critic.args.disable_param_buffers_cpu_backup is False


async def test_train_parallel_config_travels_from_trainer_to_rollout_executor(monkeypatch):
    """The driver reads the parallel config off the trainer and writes it into the executor."""
    _patch_train_group(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=False),
        inference_controller=object(),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}
    assert rollout_executor.loaded_rollout_id == -1


async def test_train_parallel_config_comes_from_the_actor_not_the_critic(monkeypatch):
    """With a critic present, the config still comes from the actor group."""
    _patch_train_group(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=True),
        inference_controller=object(),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}
