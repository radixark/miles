from argparse import Namespace

import pytest

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import _get_placement_group_layout


def _layout_args(**overrides):
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "rollout_num_gpus": 4,
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


async def test_critic_role_disables_reward_kl_and_preserves_actor_args(monkeypatch):
    groups = []

    class _Group:
        def __init__(self, args, role, with_ref):
            self.args = args
            self.role = role
            self.with_ref = with_ref

        async def init(self):
            return [0]

        async def set_rollout_manager(self):
            return None

    def _allocate_train_group(*, args, role, with_ref, **_kwargs):
        group = _Group(args, role, with_ref)
        groups.append(group)
        return group

    monkeypatch.setattr(placement_group_module, "allocate_train_group", _allocate_train_group)
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
        rollout_manager=object(),
    )

    actor, critic = groups
    assert actor.role == "actor"
    assert actor.args is args
    assert actor.args.kl_coef == 0.1
    assert actor.with_ref is True

    assert critic.role == "critic"
    assert critic.args is not args
    assert critic.args.kl_coef == 0
    assert critic.args.use_opd is False
    assert critic.args.disable_param_buffers_cpu_backup is False
    assert critic.with_ref is False
