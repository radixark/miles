import json
from unittest.mock import Mock, call

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize("entrypoint", ["prepare", "full_train"])
@pytest.mark.parametrize("fp8_rollout", [False, True])
def test_prepare_commands(monkeypatch, tmp_path, entrypoint, fp8_rollout):
    freeze_environment(monkeypatch, hardware="MI355X")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_glm5_2_744b_a40b.py")
    cpu = Mock(wraps=module.U.exec_command_cpu)
    gpu = Mock(wraps=module.U.exec_command_gpu)
    monkeypatch.setattr(module.U, "exec_command_cpu", cpu)
    monkeypatch.setattr(module.U, "exec_command_gpu", gpu)

    model_dir = tmp_path / "models"
    data_dir = tmp_path / "datasets"
    checkpoint = model_dir / "GLM-5.2_5layer"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "num_hidden_layers": 5,
            }
        )
    )

    call_entrypoint(
        module,
        entrypoint,
        {
            "model_dir": str(model_dir),
            "model_local_dir": str(model_dir),
            "data_dir": str(data_dir),
            "fp8_rollout": fp8_rollout,
        },
        sandbox=tmp_path,
    )

    downloads = [
        f"mkdir -p {model_dir} {data_dir}",
        f"hf download Pinaster/GLM-5.2_5layer --local-dir {checkpoint}",
        f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {data_dir}/dapo-math-17k",
    ]
    assert recording.commands[:3] == downloads
    assert cpu.call_args_list[:3] == [call(command) for command in downloads]

    fp8_command = (
        f"python tools/convert_hf_to_fp8.py --model-dir {checkpoint} --save-dir {checkpoint}_fp8 "
        "--strategy block --block-size 128 128 --max-workers 16"
    )
    if fp8_rollout:
        assert recording.commands[3] == fp8_command
        assert gpu.call_args_list[0] == call(fp8_command)
    else:
        assert not any("convert_hf_to_fp8.py" in command for command in recording.commands)

    conversion = recording.commands[4 if fp8_rollout else 3]
    assert f"torchrun --nproc-per-node 1 {REPO_ROOT}/tools/convert_hf_to_torch_dist.py " in conversion
    assert f"--hf-checkpoint {checkpoint} --save {checkpoint}_torch_dist " in conversion
    assert "--num-layers 5 " in conversion
    assert "--pipeline-model-parallel-size 1 --expert-model-parallel-size 1 " in conversion
    gpu.assert_any_call(conversion)

    submissions = [command for command in recording.commands if "ray job submit" in command]
    if entrypoint == "full_train":
        assert len(submissions) == 1
        hf_checkpoint = f"{checkpoint}_fp8" if fp8_rollout else str(checkpoint)
        assert f"--hf-checkpoint {hf_checkpoint} --ref-load {checkpoint}_torch_dist " in submissions[0]
    else:
        assert not submissions


@pytest.mark.parametrize("fp8_rollout", [False, True])
def test_train_uses_prepared_checkpoints(monkeypatch, tmp_path, fp8_rollout):
    freeze_environment(monkeypatch, hardware="MI355X")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_glm5_2_744b_a40b.py")

    call_entrypoint(module, "train", {"fp8_rollout": fp8_rollout}, sandbox=tmp_path)

    assert not any("hf download" in command or "tools/convert_" in command for command in recording.commands)
    submissions = [command for command in recording.commands if "ray job submit" in command]
    assert len(submissions) == 1
    command = submissions[0]
    hf_name = "GLM-5.2_5layer_fp8" if fp8_rollout else "GLM-5.2_5layer"
    assert f"--hf-checkpoint /root/models/{hf_name} " in command
    assert "--ref-load /root/models/GLM-5.2_5layer_torch_dist " in command
    assert "--num-gpus-per-node 4 " in command
    assert "--sglang-nsa-decode-backend tilelang " in command
    assert "--sglang-nsa-prefill-backend tilelang " in command
    assert "--moe-token-dispatcher-type alltoall " in command
    assert ("--sglang-moe-a2a-backend mori " in command) == fp8_rollout
