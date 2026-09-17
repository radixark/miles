import sys
from types import SimpleNamespace

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


@pytest.fixture
def amd_launcher(monkeypatch):
    freeze_environment(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_deepseek_v4.py")
    monkeypatch.setattr(module, "_resolve_colocate_memory_profile", lambda args: "192gb")
    monkeypatch.setattr(module, "_is_gfx942", lambda: True)
    return module


def _set_capabilities(monkeypatch, module, *, blockwise=False, tensorwise=True, device="gfx942"):
    caps = module._FP8Capabilities(
        device=device,
        te_version="2.8-test",
        blockwise=(blockwise, "" if blockwise else "FP8 block scaled gemm not yet supported for ROCm"),
        tensorwise=(tensorwise, "" if tensorwise else "Device arch gfx94x or gfx95x required"),
    )
    monkeypatch.setattr(module, "_probe_fp8_capabilities", lambda: caps)


@pytest.mark.parametrize(
    ("requested", "blockwise", "tensorwise", "device", "expected"),
    [
        ("auto", False, True, "gfx942", "tensorwise"),
        ("auto", True, True, "gfx950", "blockwise"),
        ("auto", True, True, "gfx942", "blockwise"),
        ("auto", False, True, "gfx950", "tensorwise"),
        ("blockwise", True, True, "gfx950", "blockwise"),
        ("tensorwise", True, True, "gfx950", "tensorwise"),
        ("tensorwise", False, True, "gfx942", "tensorwise"),
    ],
)
def test_amd_fp8_recipe_uses_te_capability(
    monkeypatch, amd_launcher, capsys, requested, blockwise, tensorwise, device, expected
):
    _set_capabilities(monkeypatch, amd_launcher, blockwise=blockwise, tensorwise=tensorwise, device=device)
    assert amd_launcher._resolve_fp8_recipe(requested) == expected
    log = capsys.readouterr().out
    assert f"requested={requested} selected={expected}" in log
    assert device in log and "TE=2.8-test" in log


@pytest.mark.parametrize("requested", ["auto", "blockwise", "tensorwise"])
def test_amd_unsupported_fp8_never_becomes_bf16(monkeypatch, amd_launcher, requested):
    _set_capabilities(monkeypatch, amd_launcher, tensorwise=False)
    with pytest.raises(RuntimeError, match="No BF16 fallback"):
        amd_launcher._resolve_fp8_recipe(requested)


@pytest.mark.parametrize("entrypoint", ["train", "full_train"])
def test_amd_explicit_blockwise_fails_before_any_commands(monkeypatch, amd_launcher, tmp_path, entrypoint):
    _set_capabilities(monkeypatch, amd_launcher)
    recording = install_command_recorder(monkeypatch)
    with pytest.raises(RuntimeError, match="block scaled gemm not yet supported for ROCm"):
        call_entrypoint(amd_launcher, entrypoint, {"fp8_recipe": "blockwise"}, sandbox=tmp_path)
    assert not recording.commands


@pytest.mark.parametrize("missing", [None, "Float8CurrentScaling", "check_fp8_support", "no_gpu"])
def test_amd_probe_checks_te_api_and_visible_device(monkeypatch, amd_launcher, missing):
    recipe = SimpleNamespace(Float8BlockScaling=object, Float8CurrentScaling=object)
    fp8 = SimpleNamespace(
        check_fp8_block_scaling_support=lambda: (False, "ROCm block scaling unsupported"),
        check_fp8_support=lambda: (True, ""),
    )
    if missing in ("Float8CurrentScaling", "check_fp8_support"):
        delattr(recipe if missing == "Float8CurrentScaling" else fp8, missing)
    cuda = SimpleNamespace(
        is_available=lambda: missing != "no_gpu",
        current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(name="MI308X", gcnArchName="gfx942"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "transformer_engine", SimpleNamespace(__version__="mock-te"))
    monkeypatch.setitem(sys.modules, "transformer_engine.common", SimpleNamespace(recipe=recipe))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", SimpleNamespace(fp8=fp8))
    if missing == "no_gpu":
        with pytest.raises(RuntimeError, match="visible training GPU"):
            amd_launcher._probe_fp8_capabilities()
    else:
        caps = amd_launcher._probe_fp8_capabilities()
        assert caps.device == "MI308X (gfx942)"
        assert caps.te_version == "mock-te"
        assert caps.blockwise == (False, "ROCm block scaling unsupported")
        assert caps.tensorwise[0] == (missing is None)


@pytest.mark.parametrize(
    ("requested", "blockwise", "selected"),
    [("auto", False, "tensorwise"), ("auto", True, "blockwise"), ("tensorwise", True, "tensorwise")],
)
def test_amd_train_emits_real_fp8_recipe(monkeypatch, amd_launcher, tmp_path, requested, blockwise, selected):
    _set_capabilities(monkeypatch, amd_launcher, blockwise=blockwise)
    recording = install_command_recorder(monkeypatch)
    call_entrypoint(amd_launcher, "train", {"fp8_recipe": requested}, sandbox=tmp_path)
    command = recording.commands[-1]
    assert f"--fp8-recipe {selected}" in command
    assert command.count("--fp8-recipe") == 1
    assert "--fp8-format e4m3" in command
    assert "--transformer-impl transformer_engine" in command
    assert "--no-gradient-accumulation-fusion" in command
    assert ("NVTE_FP8_BLOCK_SCALING_FP32_SCALES" in command) == (selected == "blockwise")
    assert "--sglang-moe-runner-backend triton" in command
    assert "--sglang-disable-custom-all-reduce" in command
    assert "--hf-checkpoint None" not in command
    assert '"TORCHINDUCTOR_MAX_AUTOTUNE": "0"' in command
    assert "--sglang-quantization unquant" not in command
    assert "SGLANG_DSV4_FP4_EXPERTS" in command


def test_amd_explicit_bf16_does_not_probe_te(monkeypatch, amd_launcher, tmp_path):
    def forbidden_probe():
        pytest.fail("BF16 training must not import or probe optional TE FP8 support")

    monkeypatch.setattr(amd_launcher, "_probe_fp8_capabilities", forbidden_probe)
    recording = install_command_recorder(monkeypatch)
    call_entrypoint(amd_launcher, "train", {"fp8_training": False}, sandbox=tmp_path)
    assert "--fp8-recipe" not in recording.commands[-1]
    assert "--fp8-format" not in recording.commands[-1]


@pytest.mark.parametrize("extra_args", ["--fp8-recipe blockwise", "--fp8-recipe=blockwise"])
def test_amd_extra_args_cannot_bypass_recipe_preflight(monkeypatch, amd_launcher, tmp_path, extra_args):
    recording = install_command_recorder(monkeypatch)
    with pytest.raises(ValueError, match="not --extra-args"):
        call_entrypoint(amd_launcher, "train", {"extra_args": extra_args}, sandbox=tmp_path)
    assert not recording.commands
