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
        colocate_memory_peak_device="cpu",
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

    def build_controller(args, pg):
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321
        return controller

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.set_eval_fleet.remote = AsyncMock()
    executor_cls = _FakeExecutorClass(executor_handle)

    with patch("miles.ray.placement_group.InferenceController", side_effect=build_controller), patch(
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
