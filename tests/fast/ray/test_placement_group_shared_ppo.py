from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import _assert_external_trainer_in_run, _get_placement_group_layout
from miles.utils.workers.types import DeploymentIdentity

_RUN_UUID = "0" * 16


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
        "use_critic": True,
        "megatron_config": None,
        "critic_load": None,
        "critic_save": None,
        "critic_lr": None,
        "critic_lr_warmup_iters": None,
        "deploy_component": "all",
        "deploy_instance_id": None,
        "critic_num_nodes": 1,
        "critic_num_gpus_per_node": 3,
    }
    values.update(overrides)
    return Namespace(**values)


class TestTheLayoutOfOneTrainerDeployment:
    def test_a_trainer_deployment_reserves_the_gpus_of_the_trainer_its_arguments_describe(self):
        """Its arguments carry one model and never learn of the others, so they alone size the release."""
        args = _layout_args(
            deploy_component="trainer",
            deploy_instance_id="a-actor",
            use_critic=False,
            megatron_config=encode_megatron_config("a"),
        )

        assert _get_placement_group_layout(args) == (2, 2)

    def test_a_whole_run_still_reserves_every_trainer_it_drives(self):
        """An unsplit run carries the whole multi-model config, and its layout must not move."""
        args = _layout_args(use_critic=False, megatron_config=encode_megatron_config("a", "b"))

        assert _get_placement_group_layout(args) == (8, 4)


@pytest.mark.parametrize(
    ("colocate", "expected"),
    [
        (False, (6, 2)),
        (True, (4, 0)),
    ],
)
def test_shared_ppo_counts_actor_bundles_once(colocate, expected):
    assert _get_placement_group_layout(_layout_args(colocate=colocate)) == expected


def test_debug_train_only_counts_actor_bundles_once_and_leaves_the_rollout_entry_empty():
    """No engine is deployed, so every bundle belongs to the trainer and the rollout slice starts past the end."""
    assert _get_placement_group_layout(_layout_args(debug_train_only=True)) == (2, 2)


def test_debug_train_only_never_reads_the_rollout_size_it_was_not_given():
    """A train-only run is launched without --rollout-num-gpus, so naming the rollout contribution
    before returning makes every one of them die on None at startup."""
    assert _get_placement_group_layout(_layout_args(debug_train_only=True, rollout_num_gpus=None)) == (2, 2)


def test_debug_rollout_only_bundles_the_eval_engines_too():
    """--eval-num-gpus buys engines this run launches, and leaving them out of the group strands them."""
    assert _get_placement_group_layout(_layout_args(debug_rollout_only=True, eval_num_gpus=3)) == (7, 0)


def test_colocate_bundles_the_eval_engines_too():
    """Colocated engines share the trainer's gpus, but the eval engines are extra ones nobody placed."""
    assert _get_placement_group_layout(_layout_args(colocate=True, eval_num_gpus=3)) == (7, 0)


def test_external_rollout_only_reserves_no_local_bundles():
    assert _get_placement_group_layout(_layout_args(debug_rollout_only=True, rollout_external=True)) == (0, 0)


def test_external_rollout_reserves_gpus_for_the_trainer_only():
    """External engines run outside ray, so only the trainer's gpus may be bundled."""
    assert _get_placement_group_layout(_layout_args(rollout_external=True)) == (2, 2)


def _multi_policy_layout_args(num_policies: int, **overrides):
    model_ids = [chr(ord("a") + i) for i in range(num_policies)]
    return _layout_args(megatron_config=encode_megatron_config(*model_ids), use_critic=False, **overrides)


class TestPlacementGroupLayout:
    def test_every_policy_reserves_a_trainer_slice_of_its_own(self):
        """Sizing the group for one policy lands every policy on the same gpus."""
        assert _get_placement_group_layout(_multi_policy_layout_args(1)) == (6, 2)
        assert _get_placement_group_layout(_multi_policy_layout_args(2)) == (8, 4)
        assert _get_placement_group_layout(_multi_policy_layout_args(3)) == (10, 6)

    def test_the_rollout_side_of_the_layout_is_not_multiplied(self):
        """Policies train apart but share one inference fleet, whose size --rollout-num-gpus alone decides."""
        num_gpus, rollout_offset = _get_placement_group_layout(_multi_policy_layout_args(3))

        assert num_gpus - rollout_offset == 4

    def test_the_policy_count_multiplies_the_debug_and_external_layouts_too(self):
        """The flags that zero the rollout side must not also drop the policies out of the trainer side."""
        assert _get_placement_group_layout(_multi_policy_layout_args(3, debug_train_only=True)) == (6, 6)
        assert _get_placement_group_layout(_multi_policy_layout_args(3, rollout_external=True)) == (6, 6)

    def test_only_the_actors_are_counted_as_policies(self):
        """An actor and its critic share the actor placement group, so a critic must not widen it."""
        args = _layout_args(megatron_config=encode_megatron_config("a"), use_critic=True)

        assert _get_placement_group_layout(args) == (6, 2)


class TestWhichFlagWinsWhenTwoAreSet:
    """The layout is one if-chain, and reordering it changes only which engines land outside the group."""

    def test_training_only_outranks_every_rollout_flag(self):
        """It is the one flag that says there are no engines at all, so nothing below it can add any."""
        args = _layout_args(debug_train_only=True, rollout_external=True, debug_rollout_only=True, colocate=True)

        assert _get_placement_group_layout(args) == (2, 2)

    def test_an_external_rollout_outranks_colocation(self):
        """Colocation shares gpus with engines this run does not own, so it has nothing left to share."""
        args = _layout_args(rollout_external=True, colocate=True)

        assert _get_placement_group_layout(args) == (2, 2)

    def test_an_external_rollout_that_is_also_rollout_only_reserves_nothing(self):
        """No trainer of ours and no engine of ours leaves an empty group, not a trainer-sized one."""
        args = _layout_args(rollout_external=True, debug_rollout_only=True, colocate=True)

        assert _get_placement_group_layout(args) == (0, 0)

    def test_a_rollout_only_run_outranks_colocation(self):
        """There is no trainer to colocate with, so taking the max would size the group off a dead operand."""
        args = _layout_args(debug_rollout_only=True, colocate=True, eval_num_gpus=3)

        assert _get_placement_group_layout(args) == (7, 0)


class TestTheLayoutOfASplitDeployment:
    def test_a_trainer_deployment_bundles_the_trainer_gpus_and_leaves_the_rollout_entry_empty(self):
        """This release installs no engine, so a rollout slice over its bundles would hand them out twice."""
        assert _get_placement_group_layout(_layout_args(deploy_component="trainer")) == (2, 2)

    def test_an_inference_deployment_counts_the_eval_engines_too(self):
        """--eval-num-gpus buys engines that live in this release, so its group has to hold them."""
        assert _get_placement_group_layout(_layout_args(deploy_component="inference", eval_num_gpus=3)) == (7, 0)

    @pytest.mark.parametrize("overrides", [{"debug_train_only": True}, {"rollout_external": True}])
    def test_an_inference_deployment_without_local_engines_bundles_nothing(self, overrides):
        """Neither flag leaves an engine inside ray, and an empty group is what asks the cluster for nothing."""
        assert _get_placement_group_layout(_layout_args(deploy_component="inference", **overrides)) == (0, 0)

    def test_a_primary_deployment_bundles_nothing(self):
        """It holds no engine and no rank of its own, so every gpu it reserved would sit idle."""
        assert _get_placement_group_layout(_layout_args(deploy_component="primary")) == (0, 0)

    @pytest.mark.parametrize("deploy_instance_id", [None, "extra"])
    def test_an_inference_deployment_bundles_the_engine_gpus_alone(self, deploy_instance_id):
        """An engine release carries the same engines the primary would, and never a trainer rank."""
        args = _layout_args(deploy_component="inference", deploy_instance_id=deploy_instance_id)

        assert _get_placement_group_layout(args) == (4, 0)

    def test_a_trainer_deployment_of_a_rollout_only_run_bundles_nothing(self):
        """--debug-rollout-only trains nothing, so this release has no rank to place."""
        assert _get_placement_group_layout(_layout_args(deploy_component="trainer", debug_rollout_only=True)) == (0, 0)


class _RecordingRolloutExecutor:
    def __init__(self):
        self.train_parallel_config = None
        self.loaded_rollout_id = None

    async def set_train_parallel_config(self, config, trainer_model_id=None):
        self.train_parallel_config = config
        self.train_parallel_config_model_id = trainer_model_id

    async def load(self, rollout_id=None):
        self.loaded_rollout_id = rollout_id


def _patch_train_controller_handles(monkeypatch, *, restored: dict[str, list[int]] | None = None) -> list:
    handles = []
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: FakeBackendCapability())

    class _Handle:
        def __init__(self, trainer_id):
            self.trainer_id = trainer_id
            self.inited_with = None
            self.calls = calls
            handles.append(self)

        async def init(self, args):
            calls.append((self.trainer_id, "init"))
            self.inited_with = args
            return (restored or {}).get(self.trainer_id, [0])

        async def get_train_parallel_config(self):
            calls.append((self.trainer_id, "get_train_parallel_config"))
            return {"dp_size": 2 if self.trainer_id == "actor" else 99}

        async def is_initialized(self) -> bool:
            return False

        async def get_deployment_identity(self) -> DeploymentIdentity:
            return DeploymentIdentity(
                run_uuid=_RUN_UUID, deploy_component="trainer", deploy_instance_id=self.trainer_id
            )

    monkeypatch.setattr(
        placement_group_module,
        "create_trainer_controller_handle",
        lambda args, *, capability, trainer_id: _Handle(trainer_id),
    )
    return handles


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
        "megatron_config": None,
        "trainer_model_id": None,
        "load": "/ckpt/run",
        "save": "/ckpt/run",
        "lr": 1e-6,
        "lr_warmup_iters": 10,
        "critic_load": "/ckpt/critic",
        "critic_save": "/ckpt/run_critic",
        "critic_lr": 2e-6,
        "critic_lr_warmup_iters": 3,
        "trainer_controller_addrs": None,
        "run_uuid": _RUN_UUID,
    }
    values.update(overrides)
    return Namespace(**values)


async def test_an_actor_and_a_critic_that_restored_to_different_rollouts_are_refused(monkeypatch):
    """The two checkpoint trees are written one after the other, so a crash between them lands exactly here."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [4]})

    with pytest.raises(AssertionError):
        await placement_group_module.create_training_models(
            _training_models_args(), rollout_executor=_RecordingRolloutExecutor()
        )


async def test_an_explicit_start_rollout_id_does_not_hide_a_mismatch(monkeypatch):
    """The override says which rollout to run next, not that the two checkpoint trees agree."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [4]})

    with pytest.raises(AssertionError):
        await placement_group_module.create_training_models(
            _training_models_args(start_rollout_id=9), rollout_executor=_RecordingRolloutExecutor()
        )


async def test_an_actor_and_a_critic_that_agree_set_the_start_rollout_id(monkeypatch):
    """A resume takes its position from the checkpoints, and the executor replays from the rollout before it."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [5]})
    args = _training_models_args()
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(args, rollout_executor=rollout_executor)

    assert args.start_rollout_id == 5
    assert rollout_executor.loaded_rollout_id == 4


async def test_a_run_without_a_critic_takes_the_actor_position(monkeypatch):
    """With one trainer there is nothing to agree with, and the actor's own position must still be taken."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5]})
    args = _training_models_args(use_critic=False)

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    assert args.start_rollout_id == 5


async def test_a_critic_run_inits_one_controller_per_role(monkeypatch):
    """Each role is its own worker, and both have to be inited before anybody calls them."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert [handle.trainer_id for handle in handles] == ["actor", "critic"]
    assert all(handle.inited_with is not None for handle in handles)


async def test_the_critic_controller_is_inited_with_neutralized_args(monkeypatch):
    """A critic controller must not hand its cells the actor's KL and OPD settings."""
    handles = _patch_train_controller_handles(monkeypatch)
    args = _training_models_args()

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    actor_args, critic_args = (handle.inited_with for handle in handles)
    assert (actor_args.kl_coef, actor_args.use_opd, actor_args.disable_param_buffers_cpu_backup) == (0.1, True, True)
    assert (critic_args.kl_coef, critic_args.use_opd, critic_args.disable_param_buffers_cpu_backup) == (
        0,
        False,
        False,
    )
    assert (args.kl_coef, args.use_opd, args.disable_param_buffers_cpu_backup) == (0.1, True, True)


async def test_the_critic_controller_is_inited_with_the_critic_checkpoint_and_schedule(monkeypatch):
    """The worker no longer swaps critic_* onto the standard fields, so the args must arrive remapped."""
    handles = _patch_train_controller_handles(monkeypatch)
    args = _training_models_args()

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    _actor_args, critic_args = (handle.inited_with for handle in handles)
    assert (critic_args.load, critic_args.save, critic_args.lr, critic_args.lr_warmup_iters) == (
        "/ckpt/critic",
        "/ckpt/run_critic",
        2e-6,
        3,
    )
    assert (args.load, args.save, args.lr, args.lr_warmup_iters) == ("/ckpt/run", "/ckpt/run", 1e-6, 10)


async def test_the_controllers_are_inited_before_the_driver_calls_them(monkeypatch):
    """init() is what hands a controller its args, so any earlier call reaches a controller without them."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert handles[0].calls == [("actor", "init"), ("critic", "init"), ("actor", "get_train_parallel_config")]


async def test_a_run_without_a_critic_starts_only_the_actor_controller(monkeypatch):
    """A critic controller nobody asked for would sit waiting for cells that are never scheduled."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=False),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert [handle.trainer_id for handle in handles] == ["actor"]


async def test_train_parallel_config_travels_from_trainer_to_rollout_executor(monkeypatch):
    """The driver reads the parallel config off the trainer and writes it into the executor."""
    _patch_train_controller_handles(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=False),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}
    assert rollout_executor.loaded_rollout_id == -1


async def test_train_parallel_config_comes_from_the_actor_not_the_critic(monkeypatch):
    """With a critic present, the config still comes from the actor group."""
    _patch_train_controller_handles(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=True),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}


class TestTheRunWaitsForEveryTrainerItReachesByAddress:
    @staticmethod
    def _patched(monkeypatch) -> list[list[tuple[str, int]]]:
        dialled: list[list[tuple[str, int]]] = []
        _patch_train_controller_handles(monkeypatch)

        async def _dial(addrs) -> None:
            dialled.append([(addr.host, addr.port) for addr in addrs])

        monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", _dial)
        return dialled

    async def test_it_dials_every_addressed_controller_before_any_of_them_is_inited(self, monkeypatch):
        """A deployment installed a moment earlier is not listening yet, and init would simply fail against it."""
        dialled = self._patched(monkeypatch)
        args = _training_models_args(
            megatron_config=None, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:9000"]
        )

        await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

        assert dialled == [[("10.0.0.1", 8000), ("10.0.0.2", 9000)]]

    async def test_a_run_that_deploys_its_own_trainer_waits_for_nobody(self, monkeypatch):
        """It installs the trainer itself, so there is no other deployment whose readiness it could dial."""
        dialled = self._patched(monkeypatch)

        await placement_group_module.create_training_models(
            _training_models_args(), rollout_executor=_RecordingRolloutExecutor()
        )

        assert dialled == []


class _IdentifyingHandle:
    def __init__(self, *, trainer_id: str, run_uuid: str, deploy_component: str, calls: list[tuple[str, str]]) -> None:
        self.trainer_id = trainer_id
        self.identity = DeploymentIdentity(
            run_uuid=run_uuid, deploy_component=deploy_component, deploy_instance_id=trainer_id
        )
        self.calls = calls

    async def get_deployment_identity(self) -> DeploymentIdentity:
        self.calls.append((self.trainer_id, "get_deployment_identity"))
        return self.identity

    async def init(self, args) -> list[int]:
        self.calls.append((self.trainer_id, "init"))
        return [0]


def _split_run_args(**overrides):
    return _training_models_args(
        megatron_config=None,
        trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:8000"],
        **overrides,
    )


def _patch_identifying_handles(monkeypatch, *, identities: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: FakeBackendCapability())

    async def _dial(addrs) -> None:
        return None

    monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", _dial)
    monkeypatch.setattr(
        placement_group_module,
        "create_trainer_controller_handle",
        lambda args, *, capability, trainer_id: _IdentifyingHandle(
            trainer_id=trainer_id,
            run_uuid=identities[trainer_id][0],
            deploy_component=identities[trainer_id][1],
            calls=calls,
        ),
    )
    return calls


class TestEveryAddressedTrainerIsCheckedBeforeAnyInitRuns:
    async def test_a_second_controller_of_another_run_is_caught_before_the_first_is_inited(self, monkeypatch):
        """init runs once per deployment, so an init before the check leaves a trainer nothing can re-init."""
        calls = _patch_identifying_handles(
            monkeypatch, identities={"actor": ("0" * 16, "trainer"), "critic": ("f" * 16, "trainer")}
        )

        with pytest.raises(AssertionError, match="drives run"):
            await placement_group_module.create_training_models(
                _split_run_args(), rollout_executor=_RecordingRolloutExecutor()
            )

        assert sorted(calls) == [("actor", "get_deployment_identity"), ("critic", "get_deployment_identity")]

    async def test_a_deployment_that_carries_an_orchestration_script_of_its_own_is_refused(self, monkeypatch):
        """Its own script drives that trainer too, so both runs would train one model from two rollout streams."""
        calls = _patch_identifying_handles(
            monkeypatch, identities={"actor": ("0" * 16, "all"), "critic": ("0" * 16, "trainer")}
        )

        with pytest.raises(AssertionError, match="nothing but the trainer"):
            await placement_group_module.create_training_models(
                _split_run_args(), rollout_executor=_RecordingRolloutExecutor()
            )

        assert ("actor", "init") not in calls


class TestTheAddressesNameOneRun:
    @staticmethod
    def _identity(
        *, run_uuid: str, deploy_component: str = "trainer", deploy_instance_id: str | None = None
    ) -> DeploymentIdentity:
        return DeploymentIdentity(
            run_uuid=run_uuid, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
        )

    def test_a_deployment_of_this_run_is_accepted(self):
        """Every deployment of one run carries the same run uuid, so the usual case must pass silently."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        _assert_external_trainer_in_run(self._identity(run_uuid=args.run_uuid), args=args)

    def test_a_deployment_of_another_run_stops_the_launch(self):
        """Pointing at last run's release trains weights this run never updates, and looks like bad rewards."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="drives run"):
            _assert_external_trainer_in_run(self._identity(run_uuid="ffffffffffffffff"), args=args)

    def test_an_unsplit_release_of_this_run_stops_the_launch(self):
        """It carries an orchestration script of its own, which drives the very trainer this launch would drive."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="nothing but the trainer"):
            _assert_external_trainer_in_run(self._identity(run_uuid=args.run_uuid, deploy_component="all"), args=args)

    def test_a_deployment_reached_as_another_trainer_stops_the_launch(self):
        """Its ranks would be driven through the workflow of a trainer they do not belong to."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="are keyed by trainer id"):
            _assert_external_trainer_in_run(
                self._identity(run_uuid=args.run_uuid, deploy_instance_id="critic"), args=args, trainer_id="actor"
            )

    def test_a_deployment_that_was_never_named_is_accepted_as_any_trainer(self):
        """It was launched without --deploy-instance-id, so there is no name here to check the workflow against."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        _assert_external_trainer_in_run(self._identity(run_uuid=args.run_uuid), args=args, trainer_id="actor")
