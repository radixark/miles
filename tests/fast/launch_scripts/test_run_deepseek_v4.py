import json
from unittest.mock import Mock

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
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
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype": "fp4"}', encoding="utf-8")
    execute_train = Mock()
    monkeypatch.setattr(module.U, "execute_train", execute_train)

    module._train(args)

    extra_env_vars = execute_train.call_args.kwargs["extra_env_vars"]
    assert extra_env_vars["MILES_SGLANG_DUMMY_LOAD"] == "1"


def test_trainer_owned_rollout_paths_select_source_and_p1_schema(tmp_path):
    module, p1 = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    _, p2 = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    source = tmp_path / "DeepSeek-V4-Flash"

    assert module._trainer_checkpoint_path(p1) == str(source)
    assert module._rollout_checkpoint_path(p1) == str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema")
    assert module._trainer_checkpoint_path(p2) == str(source)
    assert module._rollout_checkpoint_path(p2) == str(source)


def test_p3_uses_native_hybrid_trainer_and_packed_mxfp4_rollout(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    args.dsv4_mxfp4_qat = True
    args.skip_saving = True
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"expert_dtype": "fp4"}), encoding="utf-8")
    execute_train = Mock()
    monkeypatch.setattr(module.U, "execute_train", execute_train)

    module._train(args)

    train_args = execute_train.call_args.kwargs["train_args"]
    assert "--megatron-to-hf-mode bridge" in train_args
    assert "--dsv4-impl megatron" in train_args
    assert "--qkv-format thd" in train_args
    assert "--dsv4-mxfp4-qat" in train_args
    assert "--fp8-recipe mxfp8" in train_args
    assert "--rollout-fp4-experts" in train_args
    assert execute_train.call_args.kwargs["extra_env_vars"]["SGLANG_DSV4_FP4_EXPERTS"] == "1"
