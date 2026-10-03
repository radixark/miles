import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

_SCRIPT = REPO_ROOT / "scripts/run_mimo_v2_6_flash.py"


def _run(monkeypatch, tmp_path, entrypoint, **overrides):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(_SCRIPT)
    call_entrypoint(module, entrypoint, {"model_dir": str(tmp_path / "models"), **overrides}, sandbox=tmp_path)
    return recording.commands


@pytest.mark.parametrize(
    ("model_name", "precision", "engine_name"),
    [
        ("mimo26-p4-bf16", "mxfp4_w4a8_linear", "mimo26-p4-native"),
        ("MiMo-V2.6-Flash-RL-bf16", "mxfp4_w4a8_linear", "MiMo-V2.6-Flash-RL"),
        ("mimo26-p4-bf16", "mxfp4_w4a16_linear", "mimo26-p4-w4a16"),
        ("MiMo-V2.6-Flash-RL-bf16", "mxfp4_w4a16_linear", "MiMo-V2.6-Flash-RL-w4a16"),
    ],
)
def test_the_mxfp4_engines_serve_their_checkpoint_while_the_trainer_loads_bf16(
    monkeypatch, tmp_path, model_name, precision, engine_name
):
    models = tmp_path / "models"
    train = _run(monkeypatch, tmp_path, "execute", model_name=model_name, sglang_precision=precision)[-1]

    assert f"--hf-checkpoint {models}/{engine_name} --ref-load {models}/{model_name} " in train
    assert "--sglang-moe-runner-backend marlin" in train
    assert "--sglang-attention-backend" not in train
    # the fused qkv_proj slices into 4 kv-head shards, so the engine's attention TP must divide 4
    assert "--rollout-num-gpus-per-engine 4 " in train
    # only FP8 block scales come back inexact from the BF16 weights; w4a16 has none
    assert ("--check-weight-update-allow-quant-error" in train) == (precision == "mxfp4_w4a8_linear")


def test_the_bf16_engine_serves_the_bf16_conversion(monkeypatch, tmp_path):
    train = _run(monkeypatch, tmp_path, "execute", model_name="mimo26-p4-bf16")[-1]

    assert f"--hf-checkpoint {tmp_path}/models/mimo26-p4-bf16 --megatron-to-hf-mode bridge" in train
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


def test_rollouts_sample_the_full_vocabulary(monkeypatch, tmp_path):
    """top-p 1.0 and top-k -1 (the Miles defaults): no sampling-support replay."""
    train = _run(monkeypatch, tmp_path, "execute", model_name="mimo26-p4-bf16")[-1]

    assert "--rollout-top-p" not in train and "--rollout-top-k" not in train


@pytest.mark.parametrize(
    ("precision", "engine_name", "flags"),
    [
        ("mxfp4_w4a8_linear", "mimo26-p4-native", "--keep-quant"),
        ("mxfp4_w4a16_linear", "mimo26-p4-w4a16", "--keep-quant --bf16-linears"),
    ],
)
def test_prepare_converts_the_partial_engine_checkpoint_and_skips_finished_work(
    monkeypatch, tmp_path, precision, engine_name, flags
):
    commands = _run(monkeypatch, tmp_path, "prepare", model_name="mimo26-p4-bf16", sglang_precision=precision)
    conversions = [c for c in commands if "convert_mimo_v2_to_bf16.py" in c]
    assert [c.endswith("--layers 0,1,5,6") for c in conversions] == [True, False]
    assert conversions[1].endswith(f"--layers 0,1,5,6 {flags}")
    assert f"--save-dir {tmp_path}/models/{engine_name} " in conversions[1]

    for name in ("mimo26-p4-bf16", engine_name):
        (tmp_path / "models" / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / name / "model.safetensors.index.json").write_text("{}")
    commands = _run(
        monkeypatch, tmp_path, "prepare", model_name="mimo26-p4-bf16", sglang_precision=precision, prompt_data="x"
    )
    assert commands == []


def test_prepare_converts_the_full_w4a16_checkpoint_from_the_download(monkeypatch, tmp_path):
    models = tmp_path / "models"
    for name in ("MiMo-V2.6-Flash-RL", "MiMo-V2.6-Flash-RL-bf16"):
        (models / name).mkdir(parents=True)
        (models / name / "model.safetensors.index.json").write_text("{}")
    commands = _run(
        monkeypatch,
        tmp_path,
        "prepare",
        model_name="MiMo-V2.6-Flash-RL-bf16",
        sglang_precision="mxfp4_w4a16_linear",
        prompt_data="x",
    )
    (conversion,) = [c for c in commands if "convert_mimo_v2_to_bf16.py" in c]
    assert f"--model-dir {models}/MiMo-V2.6-Flash-RL --save-dir {models}/MiMo-V2.6-Flash-RL-w4a16 " in conversion
    assert conversion.endswith("--device cuda --keep-quant --bf16-linears")


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
