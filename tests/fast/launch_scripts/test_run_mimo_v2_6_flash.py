from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

from miles.utils.external_utils.command_utils import exclusive_path_lock
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

_SCRIPT = REPO_ROOT / "scripts/run_mimo_v2_6_flash.py"


@pytest.mark.parametrize("mode", ["rl", "sft"])
@pytest.mark.parametrize("checkpoint_exists", [False, True])
def test_prepare_and_execute_with_the_configured_backend(monkeypatch, tmp_path, mode, checkpoint_exists):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode=mode, model_dir=str(tmp_path / "models"), data_dir=str(tmp_path / "data"))
    if checkpoint_exists:
        checkpoint = tmp_path / "models" / args.model_name
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors.index.json").write_text("{}")

    launcher.prepare(args)
    launcher.execute(args)

    commands = recording.commands
    assert any("hf download XiaomiMiMo/MiMo-V2.6-Flash-RL" in cmd for cmd in commands) is not checkpoint_exists
    assert any("convert_mimo_v2_to_bf16.py" in cmd for cmd in commands) is not checkpoint_exists
    assert any("hf download --repo-type dataset zhuzilin/dapo-math-17k" in cmd for cmd in commands) is (mode == "rl")
    train_commands = [cmd for cmd in commands if "ray job submit" in cmd]
    assert len(train_commands) == 1
    train_script = "train.py" if mode == "rl" else "train_async.py"
    assert f"/{train_script} " in train_commands[0]


def test_prepare_waits_for_shared_checkpoint_conversion(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode="sft", model_dir=str(tmp_path / "models"))
    target = tmp_path / "models" / args.model_name
    started, converted = Event(), Event()
    monkeypatch.setattr(BaseCommandBackend, "exec_command_gpu", lambda *args, **kwargs: converted.set())

    def prepare():
        started.set()
        launcher.prepare(args)

    # Preparation takes a real filesystem lock even when its shell commands are
    # recorded, so the launcher snapshots also need a writable model_dir.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with exclusive_path_lock(str(target)):
            future = pool.submit(prepare)
            assert started.wait(timeout=5)
            attempted_conversion = converted.wait(timeout=0.2)
            target.mkdir(parents=True)
            (target / "model.safetensors.index.json").write_text("{}")
        future.result(timeout=5)

    assert not attempted_conversion
    assert not converted.is_set()


def _run(monkeypatch, tmp_path, entrypoint, **overrides):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(_SCRIPT)
    call_entrypoint(module, entrypoint, {"model_dir": str(tmp_path / "models"), **overrides}, sandbox=tmp_path)
    return recording.commands


@pytest.mark.parametrize(
    ("precision", "engine_name"),
    [("mxfp4_w4a8_linear", "MiMo-V2.6-Flash-RL"), ("mxfp4_w4a16_linear", "MiMo-V2.6-Flash-RL-w4a16")],
)
def test_the_mxfp4_engines_serve_their_checkpoint_while_the_trainer_loads_bf16(
    monkeypatch, tmp_path, precision, engine_name
):
    models = tmp_path / "models"
    train = _run(monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16", sglang_precision=precision)[
        -1
    ]

    assert f"--hf-checkpoint {models}/{engine_name} --ref-load {models}/MiMo-V2.6-Flash-RL-bf16 " in train
    assert "--sglang-moe-runner-backend marlin" in train
    assert "--sglang-attention-backend" not in train
    # the fused qkv_proj slices into 4 kv-head shards, so the engine's attention TP must divide 4
    assert "--rollout-num-gpus-per-engine 4 " in train
    # only FP8 block scales come back inexact from the BF16 weights; w4a16 has none
    assert ("--check-weight-update-allow-quant-error" in train) == (precision == "mxfp4_w4a8_linear")


def test_the_bf16_engine_serves_the_bf16_conversion(monkeypatch, tmp_path):
    train = _run(monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16")[-1]

    assert f"--hf-checkpoint {tmp_path}/models/MiMo-V2.6-Flash-RL-bf16 --megatron-to-hf-mode bridge" in train
    assert "--ref-load" not in train and "marlin" not in train


@pytest.mark.parametrize(
    ("precision", "engine_args"),
    [
        ("bf16", "--rollout-num-gpus-per-engine 8 --sglang-attention-backend fa4 "),
        (
            "mxfp4_w4a16_linear",
            "--rollout-num-gpus-per-engine 4 --sglang-moe-runner-backend marlin --sglang-attention-backend fa4 ",
        ),
        (
            "mxfp4_w4a8_linear",
            "--rollout-num-gpus-per-engine 4 --sglang-moe-runner-backend deep_gemm "
            "--check-weight-update-allow-quant-error --sglang-attention-backend fa4 ",
        ),
    ],
)
def test_b300_engines_take_the_cookbook_kernels(monkeypatch, tmp_path, precision, engine_args):
    """FA4 everywhere; DeepGEMM for the official format's experts, Marlin (BF16 activations) for w4a16."""
    train = _run(
        monkeypatch,
        tmp_path,
        "execute",
        model_name="MiMo-V2.6-Flash-RL-bf16",
        sglang_precision=precision,
        hardware="B300",
    )[-1]

    assert engine_args in train
    # the BF16 engine keeps SGLang's own MoE runner
    assert ("--sglang-moe-runner-backend" in train) == (precision != "bf16")


def test_the_full_model_trains_on_one_b300_node(monkeypatch, tmp_path):
    train = _run(monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16", hardware="B300")[-1]

    assert "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 " in train
    assert "--expert-model-parallel-size 8 " in train
    assert "--actor-num-nodes 1 " in train
    # the Adam state still streams through NVMe, and the actor goes to disk while the engines generate
    assert "--stream-optimizer-state-to-disk " in train and "--offload-train-target disk " in train


def test_the_full_model_bf16_engine_uses_dp_attention_within_budget_on_h200(monkeypatch, tmp_path):
    h200 = _run(monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16")[-1]
    b300 = _run(monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16", hardware="B300")[-1]
    mxfp4 = _run(
        monkeypatch, tmp_path, "execute", model_name="MiMo-V2.6-Flash-RL-bf16", sglang_precision="mxfp4_w4a16_linear"
    )[-1]

    # one engine per H200 node: attention TP4 x DP2, EP8
    assert (
        "--rollout-num-gpus-per-engine 8 --sglang-enable-dp-attention --sglang-dp-size 2 --sglang-ep-size 8 "
        "--sglang-enable-dp-lm-head "
    ) in h200
    # the memory budget: weights + KV at 0.72 for the rollout, 16384 tokens per GPU for the train step
    assert "--sglang-mem-fraction-static 0.72 " in h200 and "--max-tokens-per-gpu 16384 " in h200
    assert "--sglang-enable-dp-attention" not in b300 and "--sglang-enable-dp-attention" not in mxfp4
    assert "--sglang-mem-fraction-static 0.8 " in b300 and "--max-tokens-per-gpu 9216 " in b300


def test_rl_batch_and_response_length_are_configurable(monkeypatch, tmp_path):
    train = _run(
        monkeypatch,
        tmp_path,
        "execute",
        model_name="mimo26-p4-bf16",
        rollout_batch_size=32,
        n_samples_per_prompt=16,
        rollout_max_response_len=2048,
    )[-1]

    assert "--rollout-batch-size 32 --n-samples-per-prompt 16 --rollout-max-response-len 2048 " in train


def test_rollouts_sample_the_full_vocabulary(monkeypatch, tmp_path):
    """top-p 1.0 and top-k -1 (the Miles defaults): no sampling-support replay."""
    train = _run(monkeypatch, tmp_path, "execute")[-1]

    assert "--rollout-top-p" not in train and "--rollout-top-k" not in train


def test_prepare_converts_the_w4a16_checkpoint_from_the_download_and_skips_finished_work(monkeypatch, tmp_path):
    models = tmp_path / "models"
    for name in ("MiMo-V2.6-Flash-RL", "MiMo-V2.6-Flash-RL-bf16"):
        (models / name).mkdir(parents=True)
        (models / name / "model.safetensors.index.json").write_text("{}")
    kwargs = {"model_name": "MiMo-V2.6-Flash-RL-bf16", "sglang_precision": "mxfp4_w4a16_linear", "prompt_data": "x"}
    commands = _run(monkeypatch, tmp_path, "prepare", **kwargs)
    (conversion,) = [c for c in commands if "convert_mimo_v2_to_bf16.py" in c]
    assert f"--model-dir {models}/MiMo-V2.6-Flash-RL --save-dir {models}/MiMo-V2.6-Flash-RL-w4a16 " in conversion
    assert conversion.endswith("--device cuda --keep-quant --bf16-linears")

    (models / "MiMo-V2.6-Flash-RL-w4a16").mkdir()
    (models / "MiMo-V2.6-Flash-RL-w4a16" / "model.safetensors.index.json").write_text("{}")
    assert _run(monkeypatch, tmp_path, "prepare", **kwargs) == []


def test_the_full_w4a8_engine_serves_the_download_without_converting_it(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "MiMo-V2.6-Flash-RL-bf16").mkdir(parents=True)
    (models / "MiMo-V2.6-Flash-RL-bf16" / "model.safetensors.index.json").write_text("{}")
    commands = _run(
        monkeypatch,
        tmp_path,
        "prepare",
        model_name="MiMo-V2.6-Flash-RL-bf16",
        sglang_precision="mxfp4_w4a8_linear",
        prompt_data="x",
    )

    assert not any("convert_mimo_v2_to_bf16.py" in c for c in commands)
    # `hf download` runs even when the index exists: it resumes an interrupted download
    assert [c for c in commands if c.startswith("hf download")] == [
        f"hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir {models}/MiMo-V2.6-Flash-RL"
    ]


def test_sft_rejects_an_engine_precision(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match="only applies to RL"):
        _run(monkeypatch, tmp_path, "execute", mode="sft", sglang_precision="mxfp4_w4a16_linear")


def test_the_mxfp4_engines_serve_the_full_model_only(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match="full model only"):
        _run(monkeypatch, tmp_path, "execute", model_name="mimo26-p4-bf16", sglang_precision="mxfp4_w4a16_linear")
