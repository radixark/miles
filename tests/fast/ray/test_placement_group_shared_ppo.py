from argparse import Namespace
from types import SimpleNamespace

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


async def _noop_remote(*_args, **_kwargs):
    return None


async def test_critic_role_disables_reward_kl_and_preserves_actor_args(monkeypatch):
    groups = []

    class _Group:
        def __init__(self, *, args, role, **_kwargs):
            self.args = args
            self.role = role
            groups.append(self)

        async def init(self):
            return [0]

        async def set_rollout_executor(self):
            return None

    monkeypatch.setattr(placement_group_module, "RayTrainGroup", _Group)
    args = Namespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        critic_num_nodes=1,
        critic_num_gpus_per_node=2,
        use_critic=True,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=True,
        opd_type="megatron",
        disable_param_buffers_cpu_backup=True,
        start_rollout_id=None,
        rollout_global_dataset=False,
    )

    await placement_group_module.create_training_models(
        args,
        pgs={"actor": object(), "critic": object()},
        inference_controller=object(),
        rollout_executor=SimpleNamespace(load=SimpleNamespace(remote=_noop_remote)),
    )

    actor, critic = groups
    assert actor.role == "actor"
    assert actor.args is args
    assert actor.args.kl_coef == 0.1

    assert critic.role == "critic"
    assert critic.args is not args
    assert critic.args.kl_coef == 0
    assert critic.args.use_opd is False
    assert critic.args.disable_param_buffers_cpu_backup is False
