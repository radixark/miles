from __future__ import annotations

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config, write_megatron_config_trainers

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import create_rollout_components, create_training_model, create_training_models
from miles.ray.rollout.eval_fleet import EvalFleetInfo
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        sglang_router_ip=None,
        sglang_router_port=None,
        cluster_backend="ray",
        eval_num_gpus=0,
        debug_train_only=False,
        use_session_server=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def fake_components():
    controller_handle = MagicMock(name="inference_controller")
    controller_handle.check_weights = AsyncMock()
    controller_handle.offload = AsyncMock()
    controller_handle.init = AsyncMock(return_value=None)
    controller_handle.get_eval_fleet_info = AsyncMock(return_value=None)

    async def resolve_router_addrs(args, *, router_providers) -> dict:
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321
        return {}

    events: list[str] = []

    async def fake_wait_session_server_ready(args, *, provider):
        args.session_server_addrs = ["10.0.0.2:5000"]
        args.session_server_instance_ids = ["session-0"]
        events.append("session_servers_ready")

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.init = AsyncMock(side_effect=lambda: events.append("executor_init"))
    executor_handle.get_num_rollout_per_epoch = AsyncMock(return_value=5)
    executor_handle.set_eval_fleet_info = AsyncMock(return_value=None)

    capability = FakeBackendCapability(static_provider=object())

    with patch(
        "miles.ray.placement_group.create_inference_controller_handle", lambda *, capability: controller_handle
    ), patch("miles.ray.placement_group.resolve_router_addrs", resolve_router_addrs), patch(
        "miles.ray.placement_group.wait_session_server_ready", fake_wait_session_server_ready
    ), patch(
        "miles.ray.placement_group.create_rollout_executor_handle", lambda *, capability: executor_handle
    ), patch(
        "miles.ray.placement_group.get_backend_capability", lambda args: capability
    ):
        yield Namespace(
            controller_handle=controller_handle,
            executor_handle=executor_handle,
            capability=capability,
            events=events,
        )


class TestCreateRolloutComponents:
    async def test_the_executor_is_inited_after_the_session_servers_are_known(self, fake_components):
        """The executor reads the session contract off args, so it must be written before init() runs."""
        args = _make_args(num_rollout=1, use_session_server=True)

        await create_rollout_components(args)

        assert fake_components.events == ["session_servers_ready", "executor_init"]
        assert args.session_server_addrs == ["10.0.0.2:5000"]
        assert args.session_server_instance_ids == ["session-0"]

    async def test_returns_two_worker_handles(self, fake_components):
        """Both halves of rollout are independent workers, so the driver only ever holds handles."""
        args = _make_args(num_rollout=1)

        controller, executor, _ = await create_rollout_components(args)

        assert controller is fake_components.controller_handle
        assert executor is fake_components.executor_handle

    async def test_both_workers_are_told_to_init(self, fake_components):
        """Their constructors cannot await anything, so nothing runs until the driver starts them."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        fake_components.controller_handle.init.assert_awaited_once_with()
        fake_components.executor_handle.init.assert_awaited_once_with()

    async def test_the_router_addresses_are_resolved_before_the_workers_are_initialized(self, fake_components):
        """The driver's args copy must carry the contract before anything downstream reads it."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 4321)

    async def test_num_rollout_derived_from_executor_epoch_length(self, fake_components):
        """num_rollout comes from the dataset, which the executor owns."""
        args = _make_args(num_rollout=None, num_epoch=2)

        _, _, num_rollout_per_epoch = await create_rollout_components(args)

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_awaited_once_with()
        assert num_rollout_per_epoch == 5
        assert args.num_rollout == 10

    async def test_num_rollout_left_alone_when_explicitly_set(self, fake_components):
        """An explicit --num-rollout skips asking the executor for the epoch length."""
        args = _make_args(num_rollout=3)

        _, _, num_rollout_per_epoch = await create_rollout_components(args)

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_not_awaited()
        assert num_rollout_per_epoch is None
        assert args.num_rollout == 3

    async def test_the_eval_fleet_reaches_the_executor_through_an_rpc_call(self, fake_components):
        """The controller is a worker: its fleet is only knowable by calling it, never by reading it."""
        args = _make_args(num_rollout=1, eval_num_gpus=2)
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)
        fake_components.controller_handle.get_eval_fleet_info = AsyncMock(return_value=info)

        await create_rollout_components(args)

        fake_components.executor_handle.set_eval_fleet_info.assert_awaited_once_with(info)

    async def test_a_run_without_an_eval_fleet_wires_nothing_up(self, fake_components):
        """The controller answers that it deploys no fleet, and the executor is left alone."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        fake_components.controller_handle.get_eval_fleet_info.assert_awaited_once_with()
        fake_components.executor_handle.set_eval_fleet_info.assert_not_awaited()

    async def test_a_train_only_run_resolves_no_inference_addresses(self, fake_components):
        """--debug-train-only deploys no routers or session servers, so nothing can be waited on."""
        args = _make_args(num_rollout=1, debug_train_only=True)

        await create_rollout_components(args)

        assert fake_components.capability.requested_static_pool_ids == []
        assert args.sglang_router_ip is None

    async def test_the_executor_is_handed_the_fleet_the_controller_just_built(self, fake_components):
        """Checkpoint eval pins snapshots to these engines, so publishing a pre-init fleet evaluates nothing."""
        args = _make_args(num_rollout=1)
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)

        async def _publish_fleet_on_init():
            fake_components.controller_handle.get_eval_fleet_info = AsyncMock(return_value=info)

        fake_components.controller_handle.init = AsyncMock(side_effect=_publish_fleet_on_init)

        await create_rollout_components(args)

        fake_components.executor_handle.set_eval_fleet_info.assert_awaited_once_with(info)


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
            megatron_config=None,
            critic_load=None,
            critic_save=None,
            critic_lr=None,
            critic_lr_warmup_iters=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    @staticmethod
    def _patched(monkeypatch, requested: list[int]):
        from miles.ray import placement_group as placement_group_module
        from miles.ray.placement_group import PlacementGroupInfo

        def _fake_create(num_gpus):
            requested.append(num_gpus)
            return PlacementGroupInfo(
                pg="pg-sentinel",
                pg_reordered_bundle_indices=[(index * 3 + 1) % num_gpus for index in range(num_gpus)],
                pg_reordered_gpu_ids=[100 + index for index in range(num_gpus)],
            )

        monkeypatch.setattr(placement_group_module, "_create_placement_group", _fake_create)

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


class TestUpdateWeights:
    def _fakes(self, *, weight_version: int | None):
        actor_model = MagicMock()
        actor_model.update_weights = AsyncMock(return_value=weight_version)
        rollout_executor = MagicMock()
        rollout_executor.set_weight_version = AsyncMock()
        return actor_model, rollout_executor

    async def test_the_executor_is_told_which_version_the_engines_now_serve(self):
        """Without this the executor stamps every sample it collects with weight_version=None."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)

        await update_weights(actor_model, rollout_executor, rollout_id=3)

        actor_model.update_weights.assert_awaited_once_with(rollout_id=3)
        rollout_executor.set_weight_version.assert_awaited_once_with(7, trainer_model_id=None)

    async def test_the_published_version_names_the_policy_it_belongs_to(self):
        """A version published under the wrong policy judges another policy's samples against these weights."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)

        await update_weights(actor_model, rollout_executor, rollout_id=3, trainer_model_id="alpha")

        rollout_executor.set_weight_version.assert_awaited_once_with(7, trainer_model_id="alpha")

    async def test_a_trainer_that_skipped_the_broadcast_publishes_nothing(self):
        """--debug-skip-weight-update leaves the engines on their old weights, so the version must not move."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=None)

        await update_weights(actor_model, rollout_executor)

        rollout_executor.set_weight_version.assert_not_awaited()


class TestCreateTrainingModels:
    @staticmethod
    def _patched(monkeypatch, requested: list[str]) -> None:
        def _create_handle(args, *, capability, trainer_id: str):
            requested.append(trainer_id)
            handle = MagicMock()
            handle.init = AsyncMock(return_value=[0])
            handle.get_train_parallel_config = AsyncMock(return_value=None)
            return handle

        monkeypatch.setattr(placement_group_module, "create_trainer_controller_handle", _create_handle)
        monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: object())

    @staticmethod
    def _rollout_executor() -> MagicMock:
        rollout_executor = MagicMock()
        rollout_executor.set_train_parallel_config = AsyncMock()
        rollout_executor.load = AsyncMock()
        return rollout_executor

    async def test_a_configured_policy_is_addressed_by_its_own_trainer_id(self, tmp_path, monkeypatch):
        """A single entry --megatron-config names the pool '<model_id>-actor'; 'actor' addresses nothing."""
        requested: list[str] = []
        self._patched(monkeypatch, requested)
        args = Namespace(
            megatron_config=write_megatron_config(tmp_path, "alpha"), use_critic=False, start_rollout_id=None
        )

        await create_training_models(args, self._rollout_executor())

        assert requested == ["alpha-actor"]

    async def test_a_run_without_a_megatron_config_still_addresses_the_actor_and_critic_pools(self, monkeypatch):
        """Every existing single policy deployment names its two pools 'actor' and 'critic'."""
        requested: list[str] = []
        self._patched(monkeypatch, requested)
        args = Namespace(
            megatron_config=None,
            use_critic=True,
            start_rollout_id=None,
            trainer_model_id=None,
            kl_coef=0,
            use_opd=False,
            disable_param_buffers_cpu_backup=False,
            load=None,
            save=None,
            lr=1e-6,
            lr_warmup_iters=None,
            critic_load=None,
            critic_save=None,
            critic_lr=None,
            critic_lr_warmup_iters=None,
        )

        await create_training_models(args, self._rollout_executor())

        assert requested == ["actor", "critic"]

    async def test_a_config_declaring_a_critic_without_use_critic_is_refused(self, tmp_path, monkeypatch):
        """The critic pool would be deployed and never inited, so the run would hang waiting for it."""
        self._patched(monkeypatch, [])
        args = Namespace(
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha"}, {"model_id": "alpha", "role": "critic"}]
            ),
            use_critic=False,
            start_rollout_id=None,
        )

        with pytest.raises(AssertionError, match="a run without --use-critic needs no critic"):
            await create_training_models(args, self._rollout_executor())


class TestCreateTrainingModel:
    @staticmethod
    def _patch_handle(monkeypatch, *, restored: list[int]) -> None:
        def _create_handle(args, *, capability, trainer_id: str):
            handle = MagicMock()
            handle.init = AsyncMock(return_value=restored)
            return handle

        monkeypatch.setattr(placement_group_module, "create_trainer_controller_handle", _create_handle)
        monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: object())

    async def test_a_trainer_whose_cells_restored_different_rollouts_is_refused(self, monkeypatch):
        """Cells of one trainer hold one model, so disagreeing positions mean a corrupted checkpoint set."""
        self._patch_handle(monkeypatch, restored=[5, 4])

        with pytest.raises(AssertionError, match=r"trainer 'alpha-actor' restored \[5, 4\]"):
            await create_training_model(Namespace(start_rollout_id=None), trainer_id="alpha-actor")

    async def test_a_trainer_starts_where_its_cells_restored(self, monkeypatch):
        """The restored position is what makes a resume continue instead of retraining old rounds."""
        self._patch_handle(monkeypatch, restored=[3, 3])

        info = await create_training_model(Namespace(start_rollout_id=None), trainer_id="alpha-actor")

        assert info.start_rollout_id == 3

    async def test_an_explicit_start_rollout_id_wins_over_the_restored_one(self, monkeypatch):
        """--start-rollout-id is the manual override for replaying or skipping rounds."""
        self._patch_handle(monkeypatch, restored=[3])

        info = await create_training_model(Namespace(start_rollout_id=9), trainer_id="alpha-actor")

        assert info.start_rollout_id == 9

    async def test_the_restored_position_is_kept_beside_the_overridden_start(self, monkeypatch):
        """Cross trainer checks compare where checkpoints actually were, which an override must not rewrite."""
        self._patch_handle(monkeypatch, restored=[3])

        info = await create_training_model(Namespace(start_rollout_id=9), trainer_id="alpha-actor")

        assert info.restored_rollout_id == 3
