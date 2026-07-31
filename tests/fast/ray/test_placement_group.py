from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.ray.placement_group import create_rollout_components

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        check_weight_update_equal=False,
        check_weight_update_skip_list=[],
        offload_rollout=False,
        sglang_router_ip=None,
        sglang_router_port=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class _FakeExecutorClass:
    def __init__(self, handle: MagicMock) -> None:
        self._handle = handle
        self.arg_snapshots: list[Namespace] = []

    def options(self, **_kwargs):
        return self

    def remote(self, args):
        self.arg_snapshots.append(deepcopy(args))
        return self._handle


@pytest.fixture
def fake_components():
    controller = MagicMock(name="inference_controller")
    controller.check_weights = AsyncMock()
    controller.offload = AsyncMock()

    def construct_controller(args, pg):
        async def _init():
            args.sglang_router_ip = "10.0.0.1"
            args.sglang_router_port = 4321

        controller.init = AsyncMock(side_effect=_init)
        return controller

    controller_cls = MagicMock(name="InferenceController", side_effect=construct_controller)

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.set_eval_fleet.remote = AsyncMock()
    executor_cls = _FakeExecutorClass(executor_handle)

    with patch("miles.ray.placement_group.InferenceController", controller_cls), patch(
        "miles.ray.placement_group.RolloutExecutor", executor_cls
    ), patch("miles.ray.placement_group.ray.get", return_value=5):
        yield Namespace(controller=controller, executor_cls=executor_cls, executor_handle=executor_handle)


class TestCreateRolloutComponents:
    async def test_executor_is_built_after_the_router_address_is_known(self, fake_components):
        """Starting the engines fills the router address into args, and Ray pickles args at construction."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args, pg=MagicMock())

        (executor_args,) = fake_components.executor_cls.arg_snapshots
        assert executor_args.sglang_router_ip == "10.0.0.1"
        assert executor_args.sglang_router_port == 4321

    async def test_returns_a_plain_controller_and_an_actor_handle(self, fake_components):
        """The controller stays in the driver; only the executor becomes a Ray actor."""
        args = _make_args(num_rollout=1)

        controller, executor, _ = await create_rollout_components(args, pg=MagicMock())

        assert controller is fake_components.controller
        assert executor is fake_components.executor_handle

    async def test_num_rollout_derived_from_executor_epoch_length(self, fake_components):
        """num_rollout comes from the dataset, which the executor owns."""
        args = _make_args(num_rollout=None, num_epoch=2)

        _, _, num_rollout_per_epoch = await create_rollout_components(args, pg=MagicMock())

        fake_components.executor_handle.get_num_rollout_per_epoch.remote.assert_called_once()
        assert num_rollout_per_epoch == 5
        assert args.num_rollout == 10

    async def test_num_rollout_left_alone_when_explicitly_set(self, fake_components):
        """An explicit --num-rollout skips asking the executor for the epoch length."""
        args = _make_args(num_rollout=3)

        _, _, num_rollout_per_epoch = await create_rollout_components(args, pg=MagicMock())

        fake_components.executor_handle.get_num_rollout_per_epoch.remote.assert_not_called()
        assert num_rollout_per_epoch is None
        assert args.num_rollout == 3

    async def test_weight_check_and_offload_go_to_the_controller(self, fake_components):
        """Engine-side startup steps go to the controller, never to the executor."""
        args = _make_args(num_rollout=1, check_weight_update_equal=True, offload_rollout=True)

        await create_rollout_components(args, pg=MagicMock())

        actions = [call.kwargs["action"] for call in fake_components.controller.check_weights.await_args_list]
        assert actions == ["snapshot", "reset_tensors"]
        fake_components.controller.offload.assert_awaited_once()
        fake_components.executor_handle.check_weights.remote.assert_not_called()


class TestCreatePlacementGroups:
    @staticmethod
    def _args(**overrides) -> Namespace:
        defaults = dict(
            debug_train_only=False,
            debug_rollout_only=False,
            rollout_external=False,
            colocate=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            critic_num_nodes=1,
            critic_num_gpus_per_node=1,
            rollout_num_gpus=3,
            eval_num_gpus=0,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    @staticmethod
    def _patched(monkeypatch, requested: list[int]):
        from miles.ray.placement_group import PlacementGroupInfo

        def _fake_create(num_gpus):
            requested.append(num_gpus)
            return PlacementGroupInfo(
                pg="pg-sentinel",
                pg_reordered_bundle_indices=[(index * 3 + 1) % num_gpus for index in range(num_gpus)],
                pg_reordered_gpu_ids=[100 + index for index in range(num_gpus)],
            )

        monkeypatch.setattr("miles.ray.placement_group._create_placement_group", _fake_create)

    def test_each_role_views_the_shared_pg_from_its_own_offset(self, monkeypatch):
        """Roles share one placement group; the critic reuses the actor slice and rollout starts after it."""
        from miles.ray.placement_group import create_placement_groups

        requested: list[int] = []
        self._patched(monkeypatch, requested)

        pgs = create_placement_groups(self._args())

        assert requested == [5]
        assert {name: info.pg for name, info in pgs.items()} == {role: "pg-sentinel" for role in pgs}
        assert pgs["actor"].pg_reordered_gpu_ids == [100, 101, 102, 103, 104]
        assert pgs["critic"] == pgs["actor"]
        assert pgs["rollout"].pg_reordered_gpu_ids == [102, 103, 104]
        assert pgs["rollout"].pg_reordered_bundle_indices == pgs["actor"].pg_reordered_bundle_indices[2:]

    def test_a_disabled_critic_gets_no_entry_at_all(self, monkeypatch):
        """Without a critic the role map simply omits it, so consumers never see a None placement group."""
        from miles.ray.placement_group import create_placement_groups

        requested: list[int] = []
        self._patched(monkeypatch, requested)

        pgs = create_placement_groups(self._args(use_critic=False))

        assert sorted(pgs) == ["actor", "rollout"]
        assert requested == [5]
        assert pgs["rollout"].pg_reordered_gpu_ids == [102, 103, 104]
