from argparse import Namespace

import pytest

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

    async def set_train_parallel_config(self, config):
        self.train_parallel_config = config

    async def load(self, rollout_id):
        self.loaded_rollout_id = rollout_id


def _patch_train_controller_handles(monkeypatch) -> list:
    handles = []
    calls: list[tuple[str, str]] = []

    class _Handle:
        def __init__(self, role):
            self.role = role
            self.inited_with = None
            self.calls = calls
            handles.append(self)

        async def init(self, args):
            calls.append((self.role, "init"))
            self.inited_with = args
            return [0]

        async def get_train_parallel_config(self):
            calls.append((self.role, "get_train_parallel_config"))
            return {"dp_size": 2 if self.role == "actor" else 99}

    monkeypatch.setattr(placement_group_module, "create_trainer_controller_handle", lambda *, role: _Handle(role))
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
    }
    values.update(overrides)
    return Namespace(**values)


async def test_a_critic_run_inits_one_controller_per_role(monkeypatch):
    """Each role is its own worker, and both have to be inited before anybody calls them."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert [handle.role for handle in handles] == ["actor", "critic"]
    assert all(handle.inited_with is not None for handle in handles)


async def test_the_critic_controller_is_inited_with_neutralized_args(monkeypatch):
    """A critic controller must not hand its cells the actor's KL and OPD settings."""
    handles = _patch_train_controller_handles(monkeypatch)
    args = _training_models_args()

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    actor_args, critic_args = (handle.inited_with for handle in handles)
    assert actor_args is args
    assert (critic_args.kl_coef, critic_args.use_opd, critic_args.disable_param_buffers_cpu_backup) == (
        0,
        False,
        False,
    )
    assert (args.kl_coef, args.use_opd, args.disable_param_buffers_cpu_backup) == (0.1, True, True)


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

    assert [handle.role for handle in handles] == ["actor"]


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
