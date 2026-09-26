import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


_SINGLE_NODE_4LAYER = {
    "hardware": "H200",
    "model_name": "DeepSeek-V4-Flash-FP8-4layer",
    "num_nodes": 1,
    "num_gpus_per_node": 4,
}
_EIGHT_NODES = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}
_EIGHT_NODES_OF_8 = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 8}


def _train_command(monkeypatch, tmp_path, overrides):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    call_entrypoint(module, "train", overrides, sandbox=tmp_path)
    return recording.commands[-1]


@pytest.mark.parametrize(
    ("cp_size", "expected"),
    [
        (None, ["--tensor-model-parallel-size 4", "--sequence-parallel", "--context-parallel-size 1"]),
        (2, ["--tensor-model-parallel-size 2", "--sequence-parallel", "--context-parallel-size 2", "--allgather-cp"]),
        (4, ["--tensor-model-parallel-size 1", "--context-parallel-size 4", "--allgather-cp"]),
    ],
)
def test_single_node_miles_impl_splits_gpus_between_tp_and_cp(monkeypatch, tmp_path, cp_size, expected):
    """DSV4 CP must use the all-gather split; TP takes the GPUs CP leaves, with SP only above TP1."""
    overrides = _SINGLE_NODE_4LAYER | {"dsv4_impl": "miles"} | ({"cp_size": cp_size} if cp_size else {})
    command = _train_command(monkeypatch, tmp_path, overrides)

    for flag in expected:
        assert flag in command
    tp_size = 4 // (cp_size or 1)
    assert ("--allgather-cp" in command) == (tp_size < 4)
    assert ("--sequence-parallel" in command) == (tp_size > 1)


# Every recipe that pins its CP size, with that size: the single-node megatron impl (dsv4_hybrid
# needs cp_partition_mode='contiguous' for CP>1, which miles does not set) and the multi-node ones.
_FIXED_CP_RECIPES = [
    (_SINGLE_NODE_4LAYER | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "miles"}, 2),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
]


@pytest.mark.parametrize(("recipe", "recipe_cp_size"), _FIXED_CP_RECIPES)
def test_recipes_with_a_fixed_cp_size_accept_it_and_reject_another(monkeypatch, tmp_path, recipe, recipe_cp_size):
    command = _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size})
    assert f"--context-parallel-size {recipe_cp_size}" in command

    with pytest.raises(NotImplementedError, match="is untested here"):
        _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size * 2})


def test_a_node_count_without_a_recipe_reports_the_missing_recipe(monkeypatch, tmp_path):
    with pytest.raises(NotImplementedError, match="No pre-set parallel config"):
        _train_command(
            monkeypatch, tmp_path, {"hardware": "H200", "num_nodes": 2, "num_gpus_per_node": 8, "cp_size": 2}
        )


@pytest.mark.parametrize(
    ("overrides", "expected_size"),
    [
        ({"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}, 4),
        ({"hardware": "GB300", "num_nodes": 8, "num_gpus_per_node": 4}, 8),
        (
            {
                "hardware": "H200",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
        (
            {
                "hardware": "GB300",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
    ],
)
def test_the_rollout_profile_follows_the_hardware(monkeypatch, tmp_path, overrides, expected_size):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command
