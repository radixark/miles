from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import _get_placement_group_layout


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
        """These branches compute their own totals, so each one has to multiply on its own."""
        assert _get_placement_group_layout(_multi_policy_layout_args(3, debug_train_only=True)) == (6, 0)
        assert _get_placement_group_layout(_multi_policy_layout_args(3, rollout_external=True)) == (6, 6)

    def test_only_the_actors_are_counted_as_policies(self):
        """An actor and its critic share the actor placement group, so a critic must not widen it."""
        args = _layout_args(megatron_config=encode_megatron_config("a"), use_critic=True)

        assert _get_placement_group_layout(args) == (6, 2)


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

    monkeypatch.setattr(
        placement_group_module,
        "create_trainer_controller_handle",
        lambda *, capability, trainer_id: _Handle(trainer_id),
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
