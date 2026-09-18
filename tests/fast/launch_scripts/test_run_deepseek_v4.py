import json
import shlex
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

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


def test_dapo_grader_accepts_the_dataset_answer_format():
    from miles.rollout.rm_hub.math_dapo_utils import compute_score

    assert compute_score("Reasoning.\nAnswer: 45", "45")["score"] == 1.0
    assert compute_score("Reasoning.\nAnswer: 45<｜end▁of▁sentence｜>", "45")["score"] == 1.0
    assert compute_score("Reasoning.\nAnswer: 46", "45")["score"] == -1.0
    assert compute_score("Reasoning.\nAnswer: 46<｜end▁of▁sentence｜>", "45")["score"] == -1.0


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
    assert "--rm-type dapo --reward-key score --eval-reward-key score" in command
    assert "--sglang-router-policy round_robin" in command
    assert '"GPU_MAX_HW_QUEUES": "1"' in command
    assert "--sglang-moe-runner-backend triton" in command
    assert "--sglang-disable-custom-all-reduce" in command
    assert "--hf-checkpoint None" not in command
    assert '"TORCHINDUCTOR_MAX_AUTOTUNE": "0"' in command
    assert "--sglang-quantization unquant" not in command
    assert "SGLANG_DSV4_FP4_EXPERTS" in command


@pytest.mark.parametrize("use_env", [False, True])
def test_amd_cli_safe_external_ray_preserves_submission(monkeypatch, amd_launcher, use_env):
    _set_capabilities(monkeypatch, amd_launcher)
    recording = install_command_recorder(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.2:8265")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.2")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    monkeypatch.setenv("MILES_SCRIPT_JOIN_RAY_WORKERS", "1")
    monkeypatch.delenv("MILES_SCRIPT_SKIP_PROCESS_CLEANUP", raising=False)
    cli_args = ["train", "--num-nodes", "2", "--no-join-ray-workers", "--extra-args", "--run-name 'safe job'"]
    runner = CliRunner()
    baseline = runner.invoke(amd_launcher.app, cli_args)
    assert baseline.exit_code == 0, baseline.exception
    assert "pkill -9 sglang" in recording.commands[0]
    expected_submit = recording.commands[-1]
    recording.commands.clear()

    if use_env:
        monkeypatch.setenv("MILES_SCRIPT_SKIP_PROCESS_CLEANUP", "1")
    else:
        cli_args.append("--skip-process-cleanup")
    result = runner.invoke(amd_launcher.app, cli_args)

    assert result.exit_code == 0, result.exception
    assert recording.commands == [expected_submit]
    assert all(token not in expected_submit for token in ("pkill", "ray stop", "ray start", "ssh "))
    argv = shlex.split(expected_submit)
    assert "--address=http://127.0.0.1:8265" not in argv
    assert argv[argv.index("--run-name") + 1] == "safe job"
    assert argv[argv.index("--actor-num-nodes") + 1] == "2"
    assert argv[argv.index("--fp8-recipe") + 1] == "tensorwise"
    runtime_arg = next(arg for arg in argv if arg.startswith("--runtime-env-json="))
    env = json.loads(runtime_arg.split("=", 1)[1])["env_vars"]
    assert env["MASTER_ADDR"] == "10.0.0.2"
    assert env["GPU_MAX_HW_QUEUES"] == "1"
    assert env["SGLANG_SET_CPU_AFFINITY"] == "0"


@pytest.mark.parametrize("entrypoint", ["train", "full-train"])
@pytest.mark.parametrize(
    ("external_ray", "join_workers", "num_nodes", "message"),
    [
        ("0", False, 1, "MILES_SCRIPT_EXTERNAL_RAY=1"),
        ("1", True, 1, "--no-join-ray-workers"),
        ("1", True, 2, "--no-join-ray-workers"),
    ],
)
def test_amd_safe_mode_fails_before_probes_or_commands(
    monkeypatch, amd_launcher, entrypoint, external_ray, join_workers, num_nodes, message
):
    recording = install_command_recorder(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", external_ray)
    monkeypatch.setattr(amd_launcher, "_probe_fp8_capabilities", lambda: pytest.fail("unexpected GPU probe"))
    monkeypatch.setattr(amd_launcher, "_ensure_4layer_model_type", lambda args: pytest.fail("unexpected model edit"))
    cli_args = [entrypoint, "--skip-process-cleanup", "--num-nodes", str(num_nodes)]
    if not join_workers:
        cli_args.append("--no-join-ray-workers")

    result = CliRunner().invoke(amd_launcher.app, cli_args)

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert message in str(result.exception)
    assert recording.commands == []


@pytest.fixture
def fp8_smoke_launcher(monkeypatch, amd_launcher):
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    monkeypatch.setenv("RAY_ADDRESS", "http://ray-head:8265")
    monkeypatch.setenv("MASTER_ADDR", "ray-head")
    for name in ("PROFILE", "SKIP_PROCESS_CLEANUP", "JOIN_RAY_WORKERS", "EXTRA_ARGS"):
        monkeypatch.delenv(f"MILES_SCRIPT_{name}", raising=False)
    # Even when blockwise is supported, this reproducible profile selects tensorwise.
    _set_capabilities(monkeypatch, amd_launcher, blockwise=True)
    monkeypatch.setattr(amd_launcher, "_resolve_colocate_memory_profile", lambda args: args.colocate_memory_profile)
    return amd_launcher


def _fp8_smoke_args(module, **overrides):
    return module.ScriptArgs(
        **dict(profile="fp8_smoke", skip_process_cleanup=True, join_ray_workers=False, **overrides)
    )


def test_fp8_smoke_resolves_defaults_without_changing_default_profile(fp8_smoke_launcher):
    module = fp8_smoke_launcher
    plain = module.ScriptArgs()
    assert (plain.profile, plain.num_nodes, plain.context_parallel_size) == ("default", 1, 1)
    assert (plain.fp8_recipe, plain.colocate_memory_profile) == ("auto", "auto")
    assert plain.enable_eval and plain.use_fault_tolerance and not plain.skip_saving

    args = _fp8_smoke_args(module)
    assert (args.num_nodes, args.actor_num_nodes, args.actor_num_gpus_per_node) == (2, 2, 8)
    assert args.colocate and args.rollout_num_gpus == 16
    assert (args.context_parallel_size, args.fp8_recipe, args.colocate_memory_profile) == (8, "tensorwise", "192gb")
    assert args.mode == "normal" and not args.enable_eval and not args.use_fault_tolerance
    assert args.fp8_training and args.optimizer_offload and not args.skip_saving


def _assert_fp8_smoke_argv(argv):
    expected = {
        "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "2",
        "--context-parallel-size": "8",
        "--expert-model-parallel-size": "8",
        "--expert-tensor-parallel-size": "1",
        "--decoder-first-pipeline-num-layers": "22",
        "--decoder-last-pipeline-num-layers": "21",
        "--actor-num-nodes": "2",
        "--actor-num-gpus-per-node": "8",
        "--num-gpus-per-node": "8",
        "--rollout-num-gpus-per-engine": "4",
        "--sglang-tp-size": "4",
        "--sglang-ep-size": "4",
        "--sglang-dp-size": "1",
        "--fp8-recipe": "tensorwise",
        "--fp8-format": "e4m3",
        "--transformer-impl": "transformer_engine",
        "--moe-router-dtype": "fp32",
        "--num-rollout": "3000",
        "--rollout-batch-size": "2",
        "--n-samples-per-prompt": "4",
        "--num-steps-per-rollout": "1",
        "--over-sampling-batch-size": "2",
        "--rollout-max-candidate-groups": "32",
        "--rollout-timeout-seconds": "1800",
        "--rollout-function-path": "miles.rollout.sglang_rollout.generate_rollout",
        "--dynamic-sampling-filter-path": "miles.rollout.filter_hub.truncated_response_filters.mask_truncated_and_require_completed_reward_diversity",
        "--rollout-max-response-len": "2048",
        "--apply-chat-template-kwargs": '{"thinking_mode":"chat"}',
        "--rm-type": "dapo_strict",
        "--reward-key": "score",
        "--eval-reward-key": "acc",
        "--input-key": "prompt",
        "--sglang-context-length": "4096",
        "--sglang-max-total-tokens": "32768",
        "--sglang-chunked-prefill-size": "2048",
        "--sglang-max-running-requests": "8",
        "--sglang-kv-cache-dtype": "bfloat16",
        "--sglang-cuda-graph-backend-decode": "disabled",
        "--sglang-cuda-graph-backend-prefill": "disabled",
        "--optimizer-offload-fraction": "1.0",
        "--dsv4-kv-qat": "off",
        "--dsv4-index-qat": "fp8_dynamic",
        "--sglang-mem-fraction-static": "0.85",
        "--train-memory-margin-bytes": "1073741824",
        "--sglang-router-policy": "round_robin",
    }
    for option, value in expected.items():
        assert argv.count(option) == 1, option
        assert argv[argv.index(option) + 1] == value, option
    for flag in (
        "--allgather-cp",
        "--colocate",
        "--bf16",
        "--offload-train",
        "--optimizer-cpu-offload",
        "--use-precision-aware-optimizer",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--no-gradient-accumulation-fusion",
        "--apply-chat-template",
        "--continuous-rollout",
        "--optimizer-cpu-streaming-gradients",
        "--no-pin-cpu-grads",
        "--no-pin-cpu-params",
        "--rematerialize-param-from-master-weight",
        "--check-rematerialize-param-from-master-weight",
    ):
        assert argv.count(flag) == 1
    assert not set(argv) & {
        "--sequence-parallel",
        "--rollout-num-gpus",
        "--custom-rm-path",
        "--custom-reward-post-process-path",
        "--debug-train-only",
        "--debug-rollout-only",
        "--debug-disable-optimizer",
        "--use-rollout-logprobs",
        "--skip-actor-forward-only",
        "--eval-interval",
        "--use-fault-tolerance",
        "--no-offload-train",
    }


def test_fp8_smoke_cli_emits_exact_recipe_and_config_paths(monkeypatch, fp8_smoke_launcher):
    recording = install_command_recorder(monkeypatch)
    result = CliRunner().invoke(
        fp8_smoke_launcher.app,
        [
            "train",
            "--profile", "fp8_smoke",
            "--skip-process-cleanup",
            "--no-join-ray-workers",
            "--model-dir", "/models",
            "--model-local-dir", "/local-models",
            "--data-dir", "/data",
            "--save-dir", "/checkpoints",
            "--run-id", "profile-test",
            "--megatron-path", "/megatron",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert len(recording.commands) == 1
    submit = recording.commands[0]
    assert "ray job submit" in submit
    assert all(text not in submit for text in ("pkill", "ssh ", "ray start", "audit.", "/apps/"))
    argv = shlex.split(submit)
    _assert_fp8_smoke_argv(argv)
    paths = {
        "--hf-checkpoint": "/models/DeepSeek-V4-Flash-FP8",
        "--ref-load": "/local-models/DeepSeek-V4-Flash-FP8_torch_dist",
        "--prompt-data": "/data/dapo-math-17k/dapo-math-17k.jsonl",
        "--save": "/checkpoints/profile-test/checkpoints",
        "--load": "/checkpoints/profile-test/checkpoints",
    }
    for option, value in paths.items():
        assert argv[argv.index(option) + 1] == value
    runtime_arg = next(token for token in argv if token.startswith("--runtime-env-json="))
    env = json.loads(runtime_arg.split("=", 1)[1])["env_vars"]
    assert env["PYTHONPATH"].startswith(f"{REPO_ROOT}:/megatron:")
    assert env["MASTER_ADDR"] == "ray-head" and env["NCCL_NVLS_ENABLE"] == "0"
    assert env["SGLANG_DSV4_FP4_EXPERTS"] == "0" and env["GPU_MAX_HW_QUEUES"] == "1"
    assert not any("AUDIT" in name for name in env)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_nodes": 4},
        {"context_parallel_size": 4},
        {"num_gpus_per_node": 4},
        {"rollout_num_nodes": 1},
        {"model_name": "DeepSeek-V4-Flash-FP8-4layer"},
        {"task": "gsm8k"},
        {"fp8_recipe": "blockwise"},
        {"fp8_training": False},
        {"optimizer_offload": False},
        {"colocate_memory_profile": "288gb"},
        {"enable_mtp": True},
        {"enable_mis": True},
        {"enable_r3": False},
        {"train_deterministic": False},
        {"debug_train_run_id": "saved-run"},
    ],
)
def test_fp8_smoke_rejects_conflicting_config_before_commands(monkeypatch, fp8_smoke_launcher, overrides):
    recording = install_command_recorder(monkeypatch)
    monkeypatch.setattr(fp8_smoke_launcher, "_probe_fp8_capabilities", lambda: pytest.fail("unexpected GPU probe"))
    with pytest.raises(ValueError, match="fp8_smoke requires"):
        _fp8_smoke_args(fp8_smoke_launcher, **overrides)
    assert recording.commands == []


@pytest.mark.parametrize("entrypoint", ["train", "full-train"])
@pytest.mark.parametrize("missing", ["external_ray", "skip_process_cleanup", "no_join"])
def test_fp8_smoke_requires_explicit_safe_cluster(monkeypatch, fp8_smoke_launcher, entrypoint, missing):
    recording = install_command_recorder(monkeypatch)
    monkeypatch.setattr(fp8_smoke_launcher, "_probe_fp8_capabilities", lambda: pytest.fail("unexpected GPU probe"))
    cli_args = [entrypoint, "--profile", "fp8_smoke"]
    if missing == "external_ray":
        monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    if missing != "skip_process_cleanup":
        cli_args.append("--skip-process-cleanup")
    if missing != "no_join":
        cli_args.append("--no-join-ray-workers")
    result = CliRunner().invoke(fp8_smoke_launcher.app, cli_args)
    assert isinstance(result.exception, ValueError)
    assert "fp8_smoke requires" in str(result.exception)
    assert recording.commands == []


@pytest.mark.parametrize(
    "extra_args",
    [
        "--rollout-batch-size 3",
        "--rollout-max-candidate-groups=0",
        "--rollout-timeout-seconds 0",
        "--rollout-function-path custom.rollout",
        "--dynamic-sampling-filter-path custom.filter",
        "--custom-rm-path custom.reward",
        "--custom-config-path overrides.yaml",
        "--fp8-recipe=blockwise",
        "--fp8-format e5m2",
        "--exp-avg-dtype fp32",
        "--context-parallel-size 1",
        "--sglang-context-length=8192",
        "--debug-disable-optimizer",
        "--load-debug-rollout-data data.pt",
        "--skip-actor-forward-only",
        "--use-rollout-logprobs",
        "--num-roll 2",
        "--num-rollout 0",
        "--num-rollout -1",
        "--num-rollout",
        "--num-rollout 2; true",
        "--wandb-group x; true",
    ],
)
def test_fp8_smoke_rejects_extra_args_escape_hatches(monkeypatch, fp8_smoke_launcher, tmp_path, extra_args):
    recording = install_command_recorder(monkeypatch)
    monkeypatch.setattr(fp8_smoke_launcher, "_probe_fp8_capabilities", lambda: pytest.fail("unexpected GPU probe"))
    with pytest.raises(ValueError, match="fp8_smoke"):
        call_entrypoint(
            fp8_smoke_launcher,
            "full_train",
            dict(profile="fp8_smoke", skip_process_cleanup=True, join_ray_workers=False, extra_args=extra_args),
            sandbox=tmp_path,
        )
    assert recording.commands == []


def test_fp8_smoke_allows_run_length_and_log_overrides(monkeypatch, fp8_smoke_launcher):
    recording = install_command_recorder(monkeypatch)
    args = _fp8_smoke_args(
        fp8_smoke_launcher,
        num_nodes=2,
        context_parallel_size=8,
        colocate_memory_profile="192gb",
        fp8_recipe="tensorwise",
        skip_saving=True,
        hf_checkpoint="/hf/full-model",
        extra_args="--num-rollout=6000 --log-interval 1 --wandb-group 'quoted group; no shell'",
    )
    fp8_smoke_launcher._train(args)
    argv = shlex.split(recording.commands[0])
    assert argv[-5:] == ["--num-rollout=6000", "--log-interval", "1", "--wandb-group", "quoted group; no shell"]
    assert "--save" not in argv and "--load" not in argv
    assert argv[argv.index("--hf-checkpoint") + 1] == "/hf/full-model"


@pytest.mark.parametrize(
    ("rewards", "keep"),
    [([-1.0, -1.0, -1.0, -1.0], False), ([1.0, 1.0, 1.0, 1.0], False), ([1.0, -1.0, 1.0, -1.0], True)],
)
def test_fp8_smoke_standard_filter_does_not_rewrite_rewards_or_require_completion(rewards, keep):
    from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
    from miles.utils.types import Sample

    # This profile intentionally keeps the native variance filter, not an audit-only
    # "completed correct AND incorrect" rule or a fabricated reward policy.
    samples = [Sample(reward={"score": reward}, status=Sample.Status.TRUNCATED) for reward in rewards]
    result = check_reward_nonzero_std(SimpleNamespace(reward_key="score"), samples)
    assert bool(result.keep) is keep
    assert [sample.reward["score"] for sample in samples] == rewards
    assert all(sample.status == Sample.Status.TRUNCATED for sample in samples)


def test_fp8_smoke_never_falls_back_when_tensorwise_is_unavailable(monkeypatch, fp8_smoke_launcher):
    _set_capabilities(monkeypatch, fp8_smoke_launcher, blockwise=True, tensorwise=False)
    recording = install_command_recorder(monkeypatch)
    with pytest.raises(RuntimeError, match="No BF16 fallback"):
        fp8_smoke_launcher._train(_fp8_smoke_args(fp8_smoke_launcher))
    assert recording.commands == []


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
