from argparse import Namespace
from unittest.mock import Mock

import pytest

import miles.ray.placement_group as placement


def test_default_order_is_unchanged() -> None:
    bundles = [(0, "10.0.0.3", 1), (1, "10.0.0.1", 0), (2, "10.0.0.3", 0)]
    assert placement._sort_bundle_infos(bundles) == sorted(bundles, key=placement.sort_key)


def test_preferred_node_stays_whole_and_preserves_gpu_order() -> None:
    bundles = [(node * 8 + gpu, f"10.0.0.{node}", gpu) for node in (1, 2, 3) for gpu in reversed(range(8))]
    original = list(bundles)
    ordered = placement._sort_bundle_infos(bundles, "10.0.0.3")
    assert [item[1] for item in ordered[:8]] == ["10.0.0.3"] * 8
    assert [item[2] for item in ordered[:8]] == list(range(8))
    assert [item[1] for item in ordered[8:]] == ["10.0.0.1"] * 8 + ["10.0.0.2"] * 8
    assert bundles == original


def test_unknown_node_fails_instead_of_silently_using_another_disk() -> None:
    with pytest.raises(ValueError, match="not in the allocated"):
        placement._sort_bundle_infos([(0, "10.0.0.1", 0)], "10.0.0.9")


def test_preference_reaches_placement_and_rollout_offsets_stay_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    create = Mock(return_value=placement.PlacementGroupInfo("group", list(range(24)), list(range(8)) * 3))
    monkeypatch.setattr(placement, "_create_placement_group", create)
    args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        rollout_external=False,
        colocate=False,
        use_critic=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=16,
        eval_num_gpus=0,
        actor_preferred_node_ip="10.0.0.3",
    )
    groups = placement.create_placement_groups(args)
    create.assert_called_once_with(24, preferred_node_ip="10.0.0.3")
    assert groups["rollout"].pg_reordered_bundle_indices == list(range(8, 24))
    args.actor_num_nodes = 2
    with pytest.raises(ValueError, match="single actor training node"):
        placement.create_placement_groups(args)
