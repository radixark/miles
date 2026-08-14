from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.ray.placement_group import create_rollout_components, create_training_models
from miles.ray.train.group import TrainerController

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        sglang_router_ip=None,
        sglang_router_port=None,
        cluster_backend="ray",
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
    controller_handle.eval_fleet = None
    eval_fleet = object()

    async def _init():
        controller_handle.eval_fleet = eval_fleet

    controller_handle.init = AsyncMock(side_effect=_init)

    async def resolve_router_addrs(args, *, provider) -> dict:
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321
        return {}

    events: list[str] = []

    async def fake_wait_session_server_ready(args, *, provider):
        args.session_server_addrs = ["10.0.0.2:5000"]
        args.session_server_instance_ids = ["session-0"]
        events.append("session_servers_ready")

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.set_eval_fleet.remote = AsyncMock()
    executor_handle.init = AsyncMock(side_effect=lambda: events.append("executor_init"))
    executor_handle.get_num_rollout_per_epoch = AsyncMock(return_value=5)

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
            eval_fleet=eval_fleet,
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

    async def test_a_train_only_run_resolves_no_inference_addresses(self, fake_components):
        """--debug-train-only deploys no routers or session servers, so nothing can be waited on."""
        args = _make_args(num_rollout=1, debug_train_only=True)

        await create_rollout_components(args)

        assert fake_components.capability.requested_static_pool_ids == []
        assert args.sglang_router_ip is None

    async def test_the_executor_is_handed_the_fleet_the_controller_just_built(self, fake_components):
        """Checkpoint eval pins snapshots to these engines, so publishing a pre-init fleet evaluates nothing."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        fake_components.executor_handle.set_eval_fleet.remote.assert_awaited_once_with(fake_components.eval_fleet)


class _FakeRolloutExecutorHandle:
    def __init__(self) -> None:
        self.loaded_rollout_ids: list[int] = []
        self.load = SimpleNamespace(remote=self._load_remote)

    async def _load_remote(self, rollout_id: int) -> None:
        self.loaded_rollout_ids.append(rollout_id)


_TRAINER_START_ROLLOUT_ID = 7


@pytest.fixture
def fake_trainer_controllers(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, str]] = []

    async def _fake_init(self: TrainerController) -> list[int]:
        events.append(("init", self._role))
        return [_TRAINER_START_ROLLOUT_ID]

    monkeypatch.setattr(TrainerController, "init", _fake_init)
    return SimpleNamespace(events=events)


def _training_args(**overrides) -> Namespace:
    defaults = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        critic_num_nodes=1,
        critic_num_gpus_per_node=2,
        use_critic=False,
        kl_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
        opd_type=None,
        disable_param_buffers_cpu_backup=True,
        start_rollout_id=None,
        rollout_global_dataset=False,
        indep_dp=False,
        enable_witness=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestCreateTrainingModels:
    async def test_only_the_actor_is_wired_to_the_rollout_path(self, fake_trainer_controllers):
        """The critic never broadcasts weights, so handing it the engines would let it publish over the actor's."""
        inference_controller = object()
        rollout_executor = _FakeRolloutExecutorHandle()

        actor, critic = await create_training_models(
            _training_args(use_critic=True, use_opd=True, opd_type="megatron"),
            inference_controller,
            rollout_executor,
        )

        assert actor._inference_controller is inference_controller
        assert actor._rollout_executor is rollout_executor
        assert critic._inference_controller is None
        assert critic._rollout_executor is None
        assert critic._with_opd_teacher is False

    @pytest.mark.parametrize(
        ("use_opd", "opd_type", "expected"),
        [(True, "megatron", True), (True, "sglang", False), (False, "megatron", False)],
    )
    async def test_the_actor_hosts_the_teacher_only_for_megatron_opd(
        self, fake_trainer_controllers, use_opd: bool, opd_type: str, expected: bool
    ):
        """Only the in-process Megatron teacher lives in the trainer; the sglang teacher is served by the engines."""
        actor, _ = await create_training_models(
            _training_args(use_opd=use_opd, opd_type=opd_type),
            object(),
            _FakeRolloutExecutorHandle(),
        )

        assert actor._with_opd_teacher is expected

    async def test_the_executor_is_connected_and_rewound_once_the_trainers_are_up(self, fake_trainer_controllers):
        """Cells accept the executor only after init, and the executor resumes from the checkpoint's rollout."""
        args = _training_args(rollout_global_dataset=False)
        rollout_executor = _FakeRolloutExecutorHandle()

        await create_training_models(args, object(), rollout_executor)

        assert fake_trainer_controllers.events == [("init", "actor")]
        assert args.start_rollout_id == _TRAINER_START_ROLLOUT_ID
        assert rollout_executor.loaded_rollout_ids == [_TRAINER_START_ROLLOUT_ID - 1]


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
        rollout_executor.set_weight_version.remote = AsyncMock()
        return actor_model, rollout_executor

    async def test_the_executor_is_told_which_version_the_engines_now_serve(self):
        """Without this the executor stamps every sample it collects with weight_version=None."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)

        await update_weights(actor_model, rollout_executor, rollout_id=3)

        actor_model.update_weights.assert_awaited_once_with(rollout_id=3)
        rollout_executor.set_weight_version.remote.assert_awaited_once_with(7)

    async def test_a_trainer_that_skipped_the_broadcast_publishes_nothing(self):
        """--debug-skip-weight-update leaves the engines on their old weights, so the version must not move."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=None)

        await update_weights(actor_model, rollout_executor)

        rollout_executor.set_weight_version.remote.assert_not_awaited()
