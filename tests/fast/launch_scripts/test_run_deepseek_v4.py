from unittest.mock import Mock

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
_THIRTY_TWO_NODES_OF_8 = {"hardware": "H200", "num_nodes": 32, "num_gpus_per_node": 8}


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


# Every recipe that pins its CP size, with that size.
_FIXED_CP_RECIPES = [
    (_SINGLE_NODE_4LAYER | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "miles"}, 2),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
    (_THIRTY_TWO_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
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


def _direct_hf_args(tmp_path, *, rollout_mxfp8: bool):
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    return module, module.ScriptArgs(
        model_name="DeepSeek-V4-Flash",
        init_model_source="hf",
        rollout_weight_source="trainer",
        model_dir=str(tmp_path),
        model_local_dir=str(tmp_path),
        hardware="B300",
        train_fp8=False,
        train_mxfp8=True,
        rollout_fp8=not rollout_mxfp8,
        rollout_mxfp8=rollout_mxfp8,
    )


def test_direct_hf_full_train_skips_offline_weight_conversion(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    prepare_download = Mock()
    prepare_single = Mock()
    prepare_mxfp8 = Mock()
    prepare_spmd = Mock()
    prepare_schema = Mock()
    train = Mock()
    monkeypatch.setattr(module, "_prepare_download", prepare_download)
    monkeypatch.setattr(module, "_prepare_single", prepare_single)
    monkeypatch.setattr(module, "_prepare_mxfp8", prepare_mxfp8)
    monkeypatch.setattr(module, "_prepare_spmd", prepare_spmd)
    monkeypatch.setattr(module, "_prepare_mxfp8_schema", prepare_schema)
    monkeypatch.setattr(module, "_train", train)

    module._full_train(args)

    prepare_download.assert_called_once_with(args)
    prepare_single.assert_not_called()
    prepare_mxfp8.assert_not_called()
    prepare_spmd.assert_not_called()
    prepare_schema.assert_called_once_with(args)
    train.assert_called_once_with(args)


def test_trainer_owned_rollout_uses_dummy_sglang_initialization(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype": "fp4"}', encoding="utf-8")
    execute_train = Mock()
    backend = Mock(execute_train=execute_train)
    monkeypatch.setattr(args, "create_backend", lambda: backend)

    module._train(args)

    extra_env_vars = execute_train.call_args.kwargs["extra_env_vars"]
    assert extra_env_vars["MILES_SGLANG_DUMMY_LOAD"] == "1"
    assert extra_env_vars["SGLANG_DSV4_FP4_EXPERTS"] == "0"


def test_trainer_owned_rollout_paths_select_source_and_mxfp8_schema(tmp_path):
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    source = tmp_path / "DeepSeek-V4-Flash"

    assert module._trainer_checkpoint_path(args) == str(source)
    assert module._rollout_checkpoint_path(args) == str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema")


def test_direct_hf_prepare_command_copies_seed_and_schema_to_worker_storage(tmp_path):
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    args.model_local_dir = str(tmp_path / "worker")

    command = module._prepare_cmd(args)["trainer"]

    assert str(tmp_path / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash-MXFP8-schema") in command
    assert args.torch_dist_name not in command
    assert module._prepare_cmd(module.ScriptArgs(hardware="B300")) == {}
