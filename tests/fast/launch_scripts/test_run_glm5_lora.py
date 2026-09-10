import asyncio
import shlex
from argparse import Namespace

import pytest
from tests.fast.launch_scripts.model_args_harness import expand_model_args
from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import assert_matches_snapshot

from miles.backends.sglang_utils.sglang_config import _RawSglangConfig

_SCRIPT = REPO_ROOT / "scripts/run_glm5_2_744b_a40b_lora.py"


@pytest.mark.parametrize("entrypoint", ["prepare", "train", "full_train"])
def test_glm53_fp8_multinode_launch_snapshot(monkeypatch, tmp_path, entrypoint):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(_SCRIPT)
    call_entrypoint(
        module,
        entrypoint,
        {
            "model_name": "GLM-5.3",
            "fp8_rollout": True,
            "num_nodes": 4,
            "num_gpus_per_node": 8,
            "save_dir": str(tmp_path),
        },
        sandbox=tmp_path,
    )
    config_path = tmp_path / module.U.create_run_id() / "sglang_fp8_rollout.yaml"
    if entrypoint != "prepare":
        recording.pseudo_files.append(config_path.read_text())
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/glm53-lora" / f"{entrypoint}.txt"
    formatted = "\n".join(line.rstrip() for line in format_recording(recording, sandbox=tmp_path).splitlines())
    assert_matches_snapshot(snapshot, formatted.rstrip() + "\n", f"GLM-5.3::{entrypoint}")


def test_glm53_defaults_use_bf16_for_training_and_the_official_fp8_rollout():
    module = import_launch_script(_SCRIPT)
    args = module.ScriptArgs(model_name="GLM-5.3", fp8_rollout=True, num_nodes=4, num_gpus_per_node=8)
    assert args.hf_checkpoint == "/root/models/GLM-5.3-BF16"
    assert args.fp8_rollout_checkpoint == "/root/models/GLM-5.3"
    assert args.rollout_num_gpus_per_engine == 8
    assert args.sglang_mem_fraction_static == 0.8


def test_explicit_checkpoint_paths_and_engine_size_are_preserved(monkeypatch, tmp_path):
    module = import_launch_script(_SCRIPT)
    args = module.ScriptArgs(
        model_name="GLM-5.3",
        fp8_rollout=True,
        num_nodes=4,
        num_gpus_per_node=8,
        hf_checkpoint="/models/train",
        fp8_rollout_checkpoint="/models/rollout",
        rollout_num_gpus_per_engine=16,
        save_dir=str(tmp_path),
    )
    assert args.hf_checkpoint == "/models/train"
    assert args.fp8_rollout_checkpoint == "/models/rollout"
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module._train(args)
    assert "--rollout-num-gpus-per-engine 16 " in "\n".join(recording.commands)
    config = _RawSglangConfig.from_yaml(tmp_path / args.run_id / "sglang_fp8_rollout.yaml")
    engine = config.models[0]
    assert engine.model_path == "/models/rollout"
    assert engine.update_weights is True
    assert engine.server_groups[0].num_gpus == 32
    assert "--expert-model-parallel-size 32 " in module._get_parallel_config(args)


def test_glm53_model_geometry_matches_glm52():
    assert expand_model_args("glm5.3-744B-A40B_lora") == expand_model_args("glm5.2-744B-A40B_lora")


def test_dapo_8k_launch_rewards_the_requested_answer_format(monkeypatch, tmp_path):
    from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import Sample

    module = import_launch_script(_SCRIPT)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    captured = {}
    monkeypatch.setattr(module.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    module._train(
        module.ScriptArgs(
            model_name="GLM-5.3",
            task="dapo-math",
            seq_window=8192,
            fp8_rollout=True,
            num_nodes=2,
            num_gpus_per_node=8,
            dapo_dynamic_sampling=True,
            enable_wandb=False,
            save_dir=str(tmp_path),
        )
    )
    argv = shlex.split(captured["train_args"])
    assert argv[argv.index("--seq-length") + 1] == "8192"
    assert argv[argv.index("--rollout-max-context-len") + 1] == "8192"
    reward_args = Namespace(
        rm_type=argv[argv.index("--rm-type") + 1],
        reward_key=argv[argv.index("--reward-key") + 1] if "--reward-key" in argv else None,
    )
    samples = [Sample(response=f"Answer: {answer}", label="42") for answer in (42, 43)]
    for sample in samples:
        sample.reward = asyncio.run(async_rm(reward_args, sample))
    assert [sample.get_reward_value(reward_args) for sample in samples] == [1, 0]
    assert apply_reward_nonzero_std_filter(reward_args, samples).keep


def test_multinode_training_requires_the_existing_ray_cluster(monkeypatch):
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    module = import_launch_script(_SCRIPT)
    args = module.ScriptArgs(model_name="GLM-5.3", num_nodes=4, num_gpus_per_node=8)
    with pytest.raises(ValueError, match="MILES_SCRIPT_EXTERNAL_RAY=1"):
        module._train(args)


@pytest.mark.parametrize("engine_size", [-1, 3, 16])
def test_invalid_engine_size_is_rejected(engine_size):
    module = import_launch_script(_SCRIPT)
    with pytest.raises(ValueError, match="rollout_num_gpus_per_engine"):
        module.ScriptArgs(num_nodes=1, num_gpus_per_node=4, rollout_num_gpus_per_engine=engine_size)
