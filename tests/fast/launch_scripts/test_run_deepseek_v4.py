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


_W4A8 = "mxfp8-mxfp4-routed-expert"


def _direct_hf_args(tmp_path, *, train_precision="mxfp8", rollout_precision="mxfp8", **overrides):
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    fields = {
        "model_name": "DeepSeek-V4-Flash",
        "init_model_source": "hf",
        "rollout_weight_source": "trainer",
        "model_dir": str(tmp_path),
        "model_local_dir": str(tmp_path),
        "hardware": "B300",
        "train_precision": train_precision,
        "rollout_precision": rollout_precision,
    }
    return module, module.ScriptArgs(**(fields | overrides))


def _run_train(monkeypatch, module, args):
    execute_train = Mock()
    monkeypatch.setattr(args, "create_backend", lambda: Mock(execute_train=execute_train))
    module._train(args)
    return execute_train.call_args.kwargs


def test_direct_hf_full_train_skips_offline_weight_conversion(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path)
    steps = {
        name: Mock()
        for name in (
            "_prepare_download",
            "_prepare_fp8",
            "_prepare_single",
            "_prepare_spmd",
            "_prepare_mxfp8_expert_schema",
            "_train",
        )
    }
    for name, step in steps.items():
        monkeypatch.setattr(module, name, step)

    module._full_train(args)

    steps["_prepare_download"].assert_called_once_with(args)
    steps["_prepare_fp8"].assert_not_called()
    steps["_prepare_single"].assert_not_called()
    steps["_prepare_spmd"].assert_not_called()
    steps["_prepare_mxfp8_expert_schema"].assert_called_once_with(args)
    steps["_train"].assert_called_once_with(args)


def test_checkpoint_owned_block_wise_rollout_of_an_mxfp4_release_loads_the_fp8_cast(tmp_path, monkeypatch):
    module, args = _direct_hf_args(
        tmp_path,
        train_precision="blockwise-fp8",
        rollout_precision="blockwise-fp8",
        init_model_source="torch_dist",
        rollout_weight_source="checkpoint",
    )
    prepare_fp8 = Mock()
    for name in ("_prepare_download", "_prepare_single", "_prepare_spmd", "_train"):
        monkeypatch.setattr(module, name, Mock())
    monkeypatch.setattr(module, "_prepare_fp8", prepare_fp8)

    module._full_train(args)

    prepare_fp8.assert_called_once_with(args)
    assert module._rollout_checkpoint_path(args) == str(tmp_path / "DeepSeek-V4-Flash-FP8")


def test_trainer_owned_rollout_paths_select_source_and_mxfp8_expert_schema(tmp_path):
    module, mxfp8_args = _direct_hf_args(tmp_path)
    _, w4a8_args = _direct_hf_args(tmp_path, train_precision=_W4A8, rollout_precision=_W4A8)
    _, fp8_args = _direct_hf_args(tmp_path, rollout_precision="blockwise-fp8")
    source = str(tmp_path / "DeepSeek-V4-Flash")

    assert module._trainer_checkpoint_path(mxfp8_args) == source
    assert module._rollout_checkpoint_path(mxfp8_args) == str(tmp_path / "DeepSeek-V4-Flash-MXFP8-experts-schema")
    assert module._rollout_checkpoint_path(w4a8_args) == source
    assert module._rollout_checkpoint_path(fp8_args) == source


def test_mxfp8_expert_schema_is_written_from_the_source_checkpoint(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path)
    backend = Mock()
    monkeypatch.setattr(args, "create_backend", lambda: backend)

    module._prepare_mxfp8_expert_schema(args)

    command = backend.exec_command_cpu.call_args.args[0]
    assert command.startswith("python -m miles.utils.hf_rollout_schema ")
    assert f"--model-dir {tmp_path / 'DeepSeek-V4-Flash'} " in command
    assert f"--save-dir {tmp_path / 'DeepSeek-V4-Flash-MXFP8-experts-schema'} " in command


def test_direct_hf_prepare_command_copies_seed_and_schema_to_worker_storage(tmp_path):
    module, args = _direct_hf_args(tmp_path, model_local_dir=str(tmp_path / "worker"))

    command = module._prepare_cmd(args)["trainer"]

    assert str(tmp_path / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "DeepSeek-V4-Flash-MXFP8-experts-schema") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash-MXFP8-experts-schema") in command
    assert args.torch_dist_name not in command
    assert module._prepare_cmd(module.ScriptArgs(hardware="B300")) == {}


def test_train_copies_the_source_checkpoint_to_worker_storage(tmp_path, monkeypatch):
    module, args = _direct_hf_args(
        tmp_path,
        train_precision="mxfp8-mxfp4-routed-expert",
        rollout_precision="mxfp8-mxfp4-routed-expert",
        model_local_dir=str(tmp_path / "worker"),
        skip_saving=True,
    )

    command = _run_train(monkeypatch, module, args)["prepare_cmd"]["trainer"]

    assert f"{tmp_path / 'DeepSeek-V4-Flash'}/ {tmp_path / 'worker' / 'DeepSeek-V4-Flash'}" in command
    assert args.hf_checkpoint is None


@pytest.mark.parametrize("train_precision", ["bf16", "blockwise-fp8", "mxfp8"])
@pytest.mark.parametrize("rollout_precision", ["bf16", "blockwise-fp8"])
def test_precision_selections_reach_training_recipe_and_rollout_checkpoint(
    monkeypatch, tmp_path, train_precision, rollout_precision
):
    command = _train_command(
        monkeypatch,
        tmp_path,
        _SINGLE_NODE_4LAYER
        | {"hardware": "B300", "train_precision": train_precision, "rollout_precision": rollout_precision},
    )

    assert "--bf16" in command
    if train_precision == "bf16":
        assert "--fp8-recipe" not in command
        assert "--te-precision-config-file" not in command
    else:
        recipe = "blockwise" if train_precision == "blockwise-fp8" else "mxfp8"
        assert f"--fp8-recipe {recipe}" in command
        assert "--te-precision-config-file" in command
    assert "--sglang-fp8-gemm-backend auto" in command
    assert "--sglang-moe-runner-backend auto" in command
    assert "--rollout-routed-expert-format" not in command
    suffix = {"bf16": "-bf16", "blockwise-fp8": ""}[rollout_precision]
    assert f"--hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8-4layer{suffix} " in command


@pytest.mark.parametrize(
    ("train_precision", "rollout_precision", "expert_format", "moe_backend"),
    [
        ("mxfp8", "blockwise-fp8", None, "auto"),
        ("mxfp8", "mxfp8", "mxfp8", "flashinfer_trtllm_routed"),
        ("mxfp8", _W4A8, "mxfp4", "flashinfer_mxfp4"),
        (_W4A8, _W4A8, "mxfp4", "flashinfer_mxfp4"),
    ],
)
def test_routed_expert_formats_reach_sglang_and_online_updates(
    tmp_path, monkeypatch, train_precision, rollout_precision, expert_format, moe_backend
):
    module, args = _direct_hf_args(
        tmp_path, train_precision=train_precision, rollout_precision=rollout_precision, skip_saving=True
    )

    call = _run_train(monkeypatch, module, args)

    train_args = call["train_args"]
    assert "--megatron-to-hf-mode bridge" in train_args
    assert "--dsv4-impl megatron" in train_args
    assert "--qkv-format thd" in train_args
    assert "--fp8-recipe mxfp8" in train_args
    assert ("--dsv4-mxfp4-qat" in train_args) == (train_precision == _W4A8)
    if expert_format is None:
        assert "--rollout-routed-expert-format" not in train_args
    else:
        assert f"--rollout-routed-expert-format {expert_format} " in train_args
    assert f"--sglang-moe-runner-backend {moe_backend} " in train_args
    assert "--sglang-fp8-gemm-backend auto " in train_args
    env = call["extra_env_vars"]
    assert env["SGLANG_DSV4_FP4_EXPERTS"] == ("1" if expert_format == "mxfp4" else "0")
    assert env["MILES_SGLANG_DUMMY_LOAD"] == "1"


@pytest.mark.parametrize(
    ("rollout_precision", "rollout_num_nodes", "packed"),
    [(_W4A8, 0, False), (_W4A8, 1, True), ("mxfp8", 1, False)],
)
def test_disaggregated_mxfp4_rollout_broadcasts_packed_buckets(
    tmp_path, monkeypatch, rollout_precision, rollout_num_nodes, packed
):
    module, args = _direct_hf_args(
        tmp_path,
        rollout_precision=rollout_precision,
        num_nodes=1 + rollout_num_nodes,
        rollout_num_nodes=rollout_num_nodes,
        skip_saving=True,
    )

    train_args = _run_train(monkeypatch, module, args)["train_args"]

    assert ("--update-weight-transfer-mode broadcast_packed " in train_args) == packed


def test_explicit_backends_override_the_rollout_precision_defaults(tmp_path, monkeypatch):
    module, args = _direct_hf_args(
        tmp_path, skip_saving=True, sglang_moe_runner_backend="auto", sglang_fp8_gemm_backend="deep_gemm"
    )

    train_args = _run_train(monkeypatch, module, args)["train_args"]

    assert "--sglang-moe-runner-backend auto " in train_args
    assert "--sglang-fp8-gemm-backend deep_gemm " in train_args


@pytest.mark.parametrize("field", ["train_precision", "rollout_precision"])
@pytest.mark.parametrize("precision", ["mxfp8", _W4A8])
def test_mxfp_precisions_require_blackwell(field, precision):
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    with pytest.raises(AssertionError, match="require Blackwell"):
        module.ScriptArgs(hardware="H200", **{field: precision})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rollout_precision": "bf16"}, "BF16 rollout loads a BF16 checkpoint"),
        ({"rollout_weight_source": "checkpoint"}, "No checkpoint has the mxfp8 rollout layout"),
        ({"rollout_precision": _W4A8, "model_name": "DeepSeek-V4-Flash-FP8"}, "has no MXFP4 routed experts"),
        ({"rollout_precision": _W4A8, "train_precision": "blockwise-fp8"}, "pairs with MXFP8 training"),
        ({"train_precision": _W4A8}, "requires the same --rollout-precision"),
        (
            {"train_precision": _W4A8, "rollout_precision": _W4A8, "init_model_source": "torch_dist"},
            "Megatron-Bridge provider",
        ),
    ],
)
def test_precision_pairs_without_a_rollout_layout_or_training_path_are_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _direct_hf_args(tmp_path, **overrides)
