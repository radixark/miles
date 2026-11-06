"""Exports read back to the same HF-keyed tensors; conversions refuse what would change the training."""

import json
from dataclasses import replace

import pyrage
import pytest
import torch
import transformers
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from safetensors.torch import load_file, save_file

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import save_sampler_export, write_peft_export
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import (
    has_plain_peft_form,
    write_external_adapter,
    write_peft_adapter,
    write_sampler_adapter,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    map_hf_keys_to_peft_export_keys,
    map_peft_keys_to_hf_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    PARAMETRIZATION_FILENAME,
    AdapterParametrization,
    base_model_fingerprint,
    read_parametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    ConversionOp,
    apply_parametrization_conversion,
    assert_parametrization_matches_run,
    plan_parametrization_conversion,
)
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    ExternalWeightsReader,
    MilesAdapterDirReader,
    PeftAdapterReader,
    _read_peft_tensors,
    classify_adapter_source,
    read_source_lora_alpha,
)
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

L = "model.layers.0"
QKV = (f"{L}.linear_attn.in_proj_q", f"{L}.linear_attn.in_proj_k", f"{L}.linear_attn.in_proj_v")
Q_PROJ = f"{L}.self_attn.q_proj"
EXPERTS = (f"{L}.mlp.experts.gate_proj", f"{L}.mlp.experts.up_proj", f"{L}.mlp.experts.down_proj")
QWEN3 = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
# GDN v has more rows than q / k, as in Qwen3.5 (more value heads than key heads)
QWEN35 = get_hf_lora_targets(
    {
        "model_type": "qwen3_5_text",
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 2,
    }
)
QKV_ROWS = (4, 4, 8)


def _run(**overrides):
    fields = dict(
        base_model="m",
        lora_rank=2,
        lora_alpha=4.0,
        lora_type="canonical_lora",
        target_modules=(*QKV, *EXPERTS),
        experts_shared_outer_loras=False,
    )
    return AdapterParametrization(**(fields | overrides))


def _weights(shared_qkv: bool):
    g = torch.Generator().manual_seed(0)
    w = {}
    for m, rows in zip(QKV, QKV_ROWS, strict=True):
        if not shared_qkv or m == min(QKV):
            w[f"{m}.lora_A"] = torch.randn(2, 8, generator=g)
        w[f"{m}.lora_B"] = torch.randn(rows, 2, generator=g)
    for m in EXPERTS:
        w[f"{m}.lora_A"] = torch.randn(3, 2, 8, generator=g)
        w[f"{m}.lora_B"] = torch.randn(3, 8, 2, generator=g)
    return w


def test_peft_export_reads_back_bitwise(tmp_path):
    run = _run(shared_lora_a_groups=(QKV,))
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(True).items()}
    write_peft_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    names = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert any("in_proj_qkv" in n for n in names), "shared-A GDN components go back to their HF name"
    assert classify_adapter_source(tmp_path, QWEN35, load_optimizer=False) == AdapterSourceKind.PEFT
    reader = PeftAdapterReader(tmp_path, run, QWEN35)
    assert reader.read_weights().keys() == weights.keys()
    assert all(torch.equal(reader.read_weights()[k], v) for k, v in weights.items())
    assert reader.source.parametrization.shared_lora_a_groups == run.shared_lora_a_groups


def test_only_fused_components_sharing_one_a_have_a_plain_peft_form():
    assert not has_plain_peft_form(_run(), QWEN35)
    assert has_plain_peft_form(_run(shared_lora_a_groups=(tuple(sorted(QKV)),)), QWEN35)
    # a lone component of a fused module has no HF module either
    assert not has_plain_peft_form(_run(target_modules=(QKV[0],)), QWEN35)
    assert has_plain_peft_form(_run(target_modules=EXPERTS), QWEN35)


def test_hf_peft_loads_the_peft_export(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    modules = (f"{L}.self_attn.q_proj", f"{L}.mlp.down_proj")
    g = torch.Generator().manual_seed(0)
    weights = {}
    for m in modules:
        rows, cols = model.get_submodule(m).weight.shape
        weights[f"{m}.lora_A"] = torch.randn(2, cols, generator=g).to(torch.bfloat16)
        weights[f"{m}.lora_B"] = torch.randn(rows, 2, generator=g).to(torch.bfloat16)
    write_peft_adapter(tmp_path, weights, _run(lora_type="lora", target_modules=modules), QWEN3, lora_dropout=0.0)
    loaded = peft.PeftModel.from_pretrained(model, tmp_path).base_model.model
    for key, value in weights.items():
        module, factor = key.rsplit(".", 1)
        assert torch.equal(getattr(loaded.get_submodule(module), factor)["default"].weight.to(torch.bfloat16), value)


def test_a_fused_gdn_module_keeps_its_a_under_the_sorted_first_component(tmp_path):
    write_peft_adapter(tmp_path, _weights(True), _run(shared_lora_a_groups=(QKV,)), QWEN35, lora_dropout=0.0)
    weights, fused_groups = _read_peft_tensors(tmp_path, map_peft_keys_to_hf_keys, QWEN35)
    assert fused_groups == [QKV] and min(QKV) != QKV[0]
    assert {k for k in weights if k.endswith(".lora_A") and k.split(".lora_")[0] in QKV} == {f"{min(QKV)}.lora_A"}


def test_sampler_export_reads_back_through_the_adapter_dir_reader(tmp_path):
    run = _run(shared_lora_a_groups=(QKV,))
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(True).items()}
    write_sampler_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    assert classify_adapter_source(tmp_path, QWEN35, load_optimizer=False) == AdapterSourceKind.ADAPTER_DIR
    reader = MilesAdapterDirReader(tmp_path, QWEN35)
    assert not reader.source.has_optimizer_state
    assert all(torch.equal(reader.read_weights()[k], v) for k, v in weights.items())


def test_sampler_export_target_modules_name_hf_modules_not_sglang_tensors(tmp_path):
    write_sampler_adapter(
        tmp_path, {k: v.to(torch.bfloat16) for k, v in _weights(False).items()}, _run(), QWEN35, lora_dropout=0.0
    )
    names = set(load_file(str(tmp_path / "adapter_model.safetensors")))
    assert any(".experts.w1." in n for n in names) and not any(".experts.gate_proj." in n for n in names)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["target_modules"] == sorted((*QKV, *EXPERTS))


def test_splitting_a_shared_a_is_weights_only():
    source, target = _run(shared_lora_a_groups=(QKV,)), _run()
    steps = plan_parametrization_conversion(source, target, source_dtype=torch.float32)
    assert [s.op for s in steps] == [ConversionOp.SPLIT_SHARED_LORA_A]
    assert_parametrization_matches_run(source, target, steps, load_optimizer=False, has_optimizer_state=True)
    with pytest.raises(AdapterSourceError, match="needs the source's own parametrization"):
        assert_parametrization_matches_run(source, target, steps, load_optimizer=True, has_optimizer_state=True)
    split = apply_parametrization_conversion(_weights(True), steps)
    assert torch.equal(split[f"{QKV[1]}.lora_A"], split[f"{QKV[0]}.lora_A"])


def test_the_same_shared_a_groups_in_another_order_need_no_conversion():
    hf_order = _run(shared_lora_a_groups=(QKV, EXPERTS[:2]))
    reordered = _run(shared_lora_a_groups=(EXPERTS[1::-1], (QKV[2], QKV[0], QKV[1])))
    assert reordered == hf_order
    assert plan_parametrization_conversion(reordered, hf_order, source_dtype=torch.float32) == []


def test_merging_requires_equal_copies():
    steps = plan_parametrization_conversion(_run(), _run(shared_lora_a_groups=(QKV,)), source_dtype=torch.float32)
    with pytest.raises(AdapterSourceError, match="cannot share A"):
        apply_parametrization_conversion(_weights(False), steps)


def test_shared_outer_broadcasts_for_weights_and_collapses_only_if_equal():
    source, target = _run(experts_shared_outer_loras=True), _run()
    steps = plan_parametrization_conversion(source, target, source_dtype=torch.float32)
    w = _weights(False)
    for key in (f"{EXPERTS[0]}.lora_A", f"{EXPERTS[1]}.lora_A", f"{EXPERTS[2]}.lora_B"):
        w[key] = w[key][:1]
    out = apply_parametrization_conversion(w, steps)
    assert out[f"{EXPERTS[0]}.lora_A"].shape[0] == 3
    back = plan_parametrization_conversion(target, source, source_dtype=torch.float32)
    with pytest.raises(AdapterSourceError, match="experts differ"):
        apply_parametrization_conversion(_weights(False), back)


@pytest.mark.parametrize("field, value", [("base_model", "other"), ("lora_rank", 4)])
def test_mismatches_are_refused(field, value):
    with pytest.raises(AdapterSourceError):
        assert_parametrization_matches_run(
            _run(**{field: value}), _run(), [], load_optimizer=False, has_optimizer_state=False
        )


@pytest.mark.parametrize("shared_qkv", [False, True], ids=["canonical", "fused_gdn"])
def test_external_export_reads_back_through_the_external_reader(tmp_path, shared_qkv):
    # experts' Tinker names are sglang's too; Tinker's head name is what marks the export as Tinker's
    run = _run(target_modules=(*QKV, *EXPERTS, "lm_head"), shared_lora_a_groups=(QKV,) if shared_qkv else ())
    weights = _weights(shared_qkv) | {"lm_head.lora_A": torch.randn(2, 8), "lm_head.lora_B": torch.randn(16, 2)}
    write_external_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert all(t.dtype == torch.float32 for t in tensors.values())
    assert any(".experts.w1." in name for name in tensors), "gated experts take Tinker's names"
    assert classify_adapter_source(tmp_path, QWEN35, load_optimizer=False) == AdapterSourceKind.EXTERNAL_WEIGHTS
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    official = {
        "base_model_name_or_path": None,
        "target_modules": "all-linear",
        "modules_to_save": None,
        "target_parameters": None,
    }
    assert {key: config[key] for key in official} == official, "as the official export"
    read = ExternalWeightsReader(tmp_path, run, QWEN35).read_weights()
    for module in run.target_modules:
        # each member carries its own copy of a shared A
        source = run.shared_lora_a_groups[0][0] if shared_qkv and module in QKV else module
        assert torch.equal(read[f"{module}.lora_A"], weights[f"{source}.lora_A"])
        assert torch.equal(read[f"{module}.lora_B"], weights[f"{module}.lora_B"])


def _ssh_identity() -> tuple[pyrage.ssh.Identity, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    private = key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption())
    public = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
    return pyrage.ssh.Identity.from_buffer(private), public


@pytest.mark.parametrize("key_type", ["x25519", "ssh"])
def test_an_encrypted_external_export_decrypts_to_the_plain_one(tmp_path, key_type):
    if key_type == "ssh":
        identity, recipient = _ssh_identity()
    else:
        identity = pyrage.x25519.Identity.generate()
        recipient = str(identity.to_public())
    write_external_adapter(tmp_path / "plain", _weights(False), _run(), QWEN35, lora_dropout=0.0)
    write_external_adapter(
        tmp_path / "age", _weights(False), _run(), QWEN35, age_encryption_recipients=[recipient], lora_dropout=0.0
    )
    assert sorted(p.name for p in (tmp_path / "age").iterdir()) == [
        "adapter_config.json.age",
        "adapter_model.safetensors.age",
    ]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        decrypted = pyrage.decrypt((tmp_path / "age" / f"{name}.age").read_bytes(), [identity])
        assert decrypted == (tmp_path / "plain" / name).read_bytes()


def test_an_encrypted_external_export_never_writes_plaintext(tmp_path, monkeypatch):
    identity = pyrage.x25519.Identity.generate()
    real_encrypt = pyrage.encrypt

    def encrypt(data, recipients):
        assert not [p.name for p in (tmp_path / "age").glob("*") if p.suffix != ".age"], "plaintext on disk"
        return real_encrypt(data, recipients)

    monkeypatch.setattr(pyrage, "encrypt", encrypt)
    write_external_adapter(
        tmp_path / "age",
        _weights(False),
        _run(),
        QWEN35,
        age_encryption_recipients=[str(identity.to_public())],
        lora_dropout=0.0,
    )


def test_tinker_names_classify_as_external(tmp_path):
    save_file(
        {"base_model.model.model.unembed_tokens.lora_A.weight": torch.zeros(2, 2)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    (tmp_path / "adapter_config.json").write_text("{}")
    assert classify_adapter_source(tmp_path, QWEN3, load_optimizer=False) == AdapterSourceKind.EXTERNAL_WEIGHTS


def test_a_head_adapter_reads_back(tmp_path):
    run = _run(target_modules=("lm_head",))
    weights = {
        "lm_head.lora_A": torch.ones(2, 8, dtype=torch.bfloat16),
        "lm_head.lora_B": torch.ones(16, 2, dtype=torch.bfloat16),
    }
    write_peft_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    assert PeftAdapterReader(tmp_path, run, QWEN3).read_weights().keys() == weights.keys()


@pytest.mark.parametrize(
    "edit, refusal",
    [
        ({"use_dora": True}, "use_dora"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out"),
        ({"init_lora_weights": "pissa_niter_4"}, "init_lora_weights"),
        ({"alora_invocation_tokens": [1, 2]}, "alora_invocation_tokens"),
        ({"lora_alpha": None}, r"lacks \['lora_alpha'\]"),
    ],
    ids=["dora", "fan_in_fan_out", "base_rewriting_init", "activated_lora", "no_alpha"],
)
def test_a_peft_config_whose_tensors_mean_something_else_is_refused(tmp_path, edit, refusal):
    write_peft_adapter(
        tmp_path,
        {f"{Q_PROJ}.lora_A": torch.ones(2, 8), f"{Q_PROJ}.lora_B": torch.ones(4, 2)},
        _run(target_modules=(Q_PROJ,)),
        QWEN35,
        lora_dropout=0.0,
    )
    config = json.loads((tmp_path / "adapter_config.json").read_text()) | edit
    (tmp_path / "adapter_config.json").write_text(json.dumps({k: v for k, v in config.items() if v is not None}))
    with pytest.raises(AdapterSourceError, match=refusal):
        PeftAdapterReader(tmp_path, _run(), QWEN3)


@pytest.mark.parametrize(
    "key, run_config, kind",
    [
        ("decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight", True, "megatron_bridge_checkpoint"),
        ("model.layers.0.attention.wq.lora_a.weight", False, "torchtitan_checkpoint"),
        ("decoder.layers.0.self_attention.linear_qkv.weight", True, None),
        ("model.layers.0.self_attn.q_proj.lora_A.weight", False, None),
    ],
    ids=["bridge", "torchtitan", "bridge_without_adapter", "miles_adapter_state"],
)
def test_dcp_sources_are_classified_by_their_keys(tmp_path, key, run_config, kind):
    import torch.distributed.checkpoint as dcp

    dcp.save({key: torch.zeros(2)}, checkpoint_id=tmp_path, no_dist=True)
    if run_config:
        (tmp_path / "run_config.yaml").write_text("{}")
    if kind is None:
        with pytest.raises(AdapterSourceError, match="neither"):
            classify_adapter_source(tmp_path, QWEN3, load_optimizer=True)
    else:
        assert classify_adapter_source(tmp_path, QWEN3, load_optimizer=True).value == kind


def test_a_shared_outer_peft_export_has_every_experts_copy_and_reads_back_into_a_shared_outer_run(tmp_path):
    up, down = f"{L}.mlp.experts.up_proj", f"{L}.mlp.experts.down_proj"
    run = _run(target_modules=(up, down), experts_shared_outer_loras=True)
    g = torch.Generator().manual_seed(0)
    weights = {
        f"{up}.lora_A": torch.randn(1, 2, 8, generator=g),
        f"{up}.lora_B": torch.randn(3, 8, 2, generator=g),
        f"{down}.lora_A": torch.randn(3, 2, 8, generator=g),
        f"{down}.lora_B": torch.randn(1, 8, 2, generator=g),
    }
    write_peft_adapter(tmp_path, weights, run, QWEN3, lora_dropout=0.0)

    names = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert names == {
        map_hf_keys_to_peft_export_keys(f"{module}.lora_{factor}", e)
        for module in (up, down)
        for factor in "AB"
        for e in range(3)
    }, "HF PEFT matches only per-expert names"
    reader = PeftAdapterReader(tmp_path, run, QWEN3)
    steps = plan_parametrization_conversion(reader.source.parametrization, run, source_dtype=torch.bfloat16)
    assert ConversionOp.COLLAPSE_SHARED_EXPERT_SIDE in {s.op for s in steps}
    read = apply_parametrization_conversion(reader.read_weights(), steps)
    for key, value in weights.items():
        assert torch.equal(read[key], value.bfloat16().float()), key


def test_shared_outer_flag_is_ignored_without_routed_experts():
    dense = _run(target_modules=QKV, experts_shared_outer_loras=True)
    assert (
        plan_parametrization_conversion(
            dense, replace(dense, experts_shared_outer_loras=False), source_dtype=torch.float32
        )
        == []
    )


def test_a_fused_gdn_module_inside_a_larger_shared_a_group_round_trips(tmp_path):
    z = f"{L}.linear_attn.in_proj_z"
    run = _run(target_modules=(*QKV, z), shared_lora_a_groups=((*QKV, z),))
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(True).items() if k.split(".lora_")[0] in QKV}
    weights[f"{z}.lora_B"] = torch.ones(4, 2, dtype=torch.bfloat16)
    write_sampler_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    names = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert sorted(n.rsplit(".", 1)[-1] for n in names) == ["in_proj_qkv", "in_proj_z"]
    read = MilesAdapterDirReader(tmp_path, QWEN35).read_weights()
    assert read.keys() == weights.keys() and all(torch.equal(read[k], v) for k, v in weights.items())


GPT_OSS = get_hf_lora_targets({"model_type": "gpt_oss", "num_hidden_layers": 1})


def test_gpt_oss_packed_experts_round_trip_through_hf_components(tmp_path):
    gate, up, down = (f"{L}.mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
    run = _run(target_modules=(gate, up, down), shared_lora_a_groups=((gate, up),))
    g = torch.Generator().manual_seed(0)
    weights = {
        f"{gate}.lora_A": torch.randn(3, 2, 8, generator=g),
        f"{gate}.lora_B": torch.randn(3, 4, 2, generator=g),
        f"{up}.lora_B": torch.randn(3, 4, 2, generator=g),
        f"{down}.lora_A": torch.randn(3, 2, 4, generator=g),
        f"{down}.lora_B": torch.randn(3, 8, 2, generator=g),
    }
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    write_peft_adapter(tmp_path, weights, run, GPT_OSS, lora_dropout=0.0)
    packed = load_file(str(tmp_path / "adapter_model.safetensors"))
    # HF gpt-oss interleaves gate and up rows in gate_up_proj
    experts = f"base_model.model.{L}.mlp.experts"
    b = packed[f"{experts}.gate_up_proj.lora_B.weight"]
    assert torch.equal(b[:, 0::2], weights[f"{gate}.lora_B"]) and torch.equal(b[:, 1::2], weights[f"{up}.lora_B"])
    assert packed[f"{experts}.down_proj.lora_A.weight"].shape == (3, 2, 4), "packed experts stay 3D"
    read = PeftAdapterReader(tmp_path, run, GPT_OSS)
    assert read.source.parametrization.shared_lora_a_groups == ((gate, up),)
    assert all(torch.equal(read.read_weights()[k], v) for k, v in weights.items())


def test_a_sampler_export_of_a_w1_w2_w3_family_reads_back_under_its_own_names(tmp_path):
    targets = get_hf_lora_targets({"model_type": "deepseek_v41", "num_hidden_layers": 1, "n_shared_experts": 0})
    w1 = "layers.0.ffn.experts.w1"
    run = _run(target_modules=(w1,))
    weights = {
        f"{w1}.lora_A": torch.ones(3, 2, 8, dtype=torch.bfloat16),
        f"{w1}.lora_B": torch.ones(3, 8, 2, dtype=torch.bfloat16),
    }
    write_sampler_adapter(tmp_path, weights, run, targets, lora_dropout=0.0)
    assert MilesAdapterDirReader(tmp_path, targets).read_weights().keys() == weights.keys()


def test_a_sampler_export_reads_back_when_the_group_starts_at_a_later_gdn_component(tmp_path):
    reordered = (QKV[1], QKV[0], QKV[2])
    run = _run(shared_lora_a_groups=(reordered,))
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(True).items()}
    write_sampler_adapter(tmp_path, weights, run, QWEN35, lora_dropout=0.0)
    read = MilesAdapterDirReader(tmp_path, QWEN35).read_weights()
    assert read.keys() == weights.keys() and all(torch.equal(read[k], v) for k, v in weights.items())


def test_gpt_oss_gate_and_up_with_their_own_a_have_no_plain_peft_form():
    # HF gpt-oss has one packed experts.gate_up_proj; canonical gate / up with their own A have no module there
    gate, up, down = (f"{L}.mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
    assert not has_plain_peft_form(_run(target_modules=(gate, up, down)), GPT_OSS)
    assert has_plain_peft_form(_run(target_modules=(gate, up, down), shared_lora_a_groups=((gate, up),)), GPT_OSS)


def test_gpt_oss_sampler_export_keeps_packed_hf_names(tmp_path):
    gate, up, down = (f"{L}.mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
    run = _run(target_modules=(gate, up, down), shared_lora_a_groups=((gate, up),))
    weights = (
        {f"{gate}.lora_A": torch.ones(3, 2, 8)}
        | {f"{m}.lora_B": torch.ones(3, 8, 2) for m in (gate, up, down)}
        | {f"{down}.lora_A": torch.ones(3, 2, 8)}
    )
    write_sampler_adapter(tmp_path, weights, run, GPT_OSS, lora_dropout=0.0)
    names = set(load_file(str(tmp_path / "adapter_model.safetensors")))
    assert names == {f"{L}.mlp.experts.{p}.lora_{f}.weight" for p in ("gate_up_proj", "down_proj") for f in "AB"}


def test_gpt_oss_down_proj_alone_stays_one_packed_3d_tensor(tmp_path):
    """HF gpt-oss has no per-expert down_proj; adapting only it must not split it into experts.N.down_proj."""
    down = f"{L}.mlp.experts.down_proj"
    weights = {f"{down}.lora_A": torch.ones(3, 2, 4), f"{down}.lora_B": torch.ones(3, 8, 2)}
    write_peft_adapter(tmp_path / "peft", weights, _run(target_modules=(down,)), GPT_OSS, lora_dropout=0.0)
    write_sampler_adapter(tmp_path / "sampler", weights, _run(target_modules=(down,)), GPT_OSS, lora_dropout=0.0)
    for export, prefix in (("peft", "base_model.model."), ("sampler", "")):
        tensors = load_file(str(tmp_path / export / "adapter_model.safetensors"))
        assert set(tensors) == {f"{prefix}{down}.lora_A.weight", f"{prefix}{down}.lora_B.weight"}, export
        assert tensors[f"{prefix}{down}.lora_A.weight"].shape == (3, 2, 4)


def _dcp_with(path, key):
    import torch.distributed.checkpoint as dcp

    dcp.save({key: torch.zeros(2)}, checkpoint_id=path, no_dist=True)


def test_a_megatron_bridge_source_records_its_alpha_in_run_config(tmp_path):
    _dcp_with(tmp_path, "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight")
    (tmp_path / "run_config.yaml").write_text("peft:\n  alpha: 32\n")
    assert read_source_lora_alpha(tmp_path) == 32.0


def test_a_megatron_bridge_full_fine_tune_is_no_lora_source(tmp_path):
    _dcp_with(tmp_path, "decoder.layers.0.self_attention.linear_qkv.weight")
    (tmp_path / "run_config.yaml").write_text("peft: null\n")
    with pytest.raises(ValueError, match="neither Megatron-Bridge nor torchtitan LoRA keys"):
        read_source_lora_alpha(tmp_path)


def test_a_bridge_checkpoint_with_adapter_keys_but_no_peft_config_is_refused(tmp_path):
    _dcp_with(tmp_path, "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight")
    (tmp_path / "run_config.yaml").write_text("peft: null\n")
    with pytest.raises(AssertionError, match="adapter keys but its run_config.yaml has no peft"):
        read_source_lora_alpha(tmp_path)


class _StackedDownExport:
    """Bridge's export of a packed checkpoint: experts.down_proj comes as one 3D tensor, not one per expert."""

    bf16_adapter_key_style = "peft"

    def __init__(self):
        self.a, self.b = torch.randn(3, 2, 4), torch.randn(3, 4, 2)

    def export_bf16_adapter(self, slot):
        down = f"{L}.mlp.experts.down_proj"
        yield f"{down}.lora_A.weight", self.a.clone()
        yield f"{down}.lora_B.weight", self.b.clone()


def test_an_expert_module_the_export_stacks_stays_stacked(tmp_path):
    """The written form follows Bridge's export, not the model family: a packed base has no per-expert names."""
    down = f"{L}.mlp.experts.down_proj"
    export = _StackedDownExport()
    write_peft_export(tmp_path, export, 0, _run(target_modules=(down,)), QWEN3, lora_dropout=0.0)

    written = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(written) == {f"base_model.model.{down}.lora_A.weight", f"base_model.model.{down}.lora_B.weight"}
    assert torch.equal(written[f"base_model.model.{down}.lora_B.weight"], export.b.to(torch.bfloat16))


class _PerExpertDownExport:
    """Bridge's per-expert export: one 2D tensor per expert."""

    bf16_adapter_key_style = "peft"

    def __init__(self):
        self.a, self.b = torch.randn(3, 2, 4), torch.randn(3, 4, 2)

    def export_bf16_adapter(self, slot):
        for e in range(3):
            yield f"{L}.mlp.experts.{e}.down_proj.lora_A.weight", self.a[e].clone()
            yield f"{L}.mlp.experts.{e}.down_proj.lora_B.weight", self.b[e].clone()


def test_a_sampler_export_is_written_uncommitted_with_its_tenant_and_parametrization(tmp_path):
    down = f"{L}.mlp.experts.down_proj"
    run = _run(target_modules=(down,))
    export = _PerExpertDownExport()

    written = save_sampler_export(tmp_path, "v1", export, 0, run, QWEN3, tenant_digest="digest", lora_dropout=0.0)

    assert written.path.parent == tmp_path and not (tmp_path / "v1").exists(), "the committer links it"
    tensors = load_file(str(written.path / "adapter_model.safetensors"))
    # sglang's stacked expert names, every expert in one 3D tensor
    assert set(tensors) == {f"{L}.mlp.experts.w2.lora_A.weight", f"{L}.mlp.experts.w2.lora_B.weight"}
    assert torch.equal(tensors[f"{L}.mlp.experts.w2.lora_B.weight"], export.b.to(torch.bfloat16))
    assert json.loads((written.path / "tenant.json").read_text()) == {"tenant_digest": "digest"}
    assert MilesAdapterDirReader(written.path, QWEN3).source.parametrization == run


class _BridgeSharedOuterExport:
    """Bridge's shared-outer export (peft_bridge.py, expand_shared_outer=False): one unindexed ``[1, r, in]`` A for
    every expert, and a 2D B per expert."""

    bf16_adapter_key_style = "peft"

    def __init__(self, num_experts):
        self.a = torch.randn(1, 2, 4)
        self.b = [torch.randn(8, 2) for _ in range(num_experts)]

    def export_bf16_adapter(self, slot):
        gate = f"{L}.mlp.experts.gate_proj"
        yield f"{gate}.lora_A.weight", self.a.clone()
        for expert, b in enumerate(self.b):
            yield f"{L}.mlp.experts.{expert}.gate_proj.lora_B.weight", b.clone()


def test_a_bridge_shared_outer_side_is_written_per_expert_like_its_pair(tmp_path):
    """A shared [1, ...] side is not a packed checkpoint's stacked tensor: HF PEFT needs it under every expert."""
    gate = f"{L}.mlp.experts.gate_proj"
    export = _BridgeSharedOuterExport(num_experts=3)
    write_peft_export(
        tmp_path, export, 0, _run(target_modules=(gate,), experts_shared_outer_loras=True), QWEN3, lora_dropout=0.0
    )

    written = load_file(str(tmp_path / "adapter_model.safetensors"))
    expert = f"base_model.model.{L}.mlp.experts.{{}}.gate_proj"
    assert set(written) == {f"{expert.format(e)}.lora_{f}.weight" for e in range(3) for f in "AB"}
    for e in range(3):
        assert torch.equal(written[f"{expert.format(e)}.lora_A.weight"], export.a[0].to(torch.bfloat16))
        assert torch.equal(written[f"{expert.format(e)}.lora_B.weight"], export.b[e].to(torch.bfloat16))


_GLM5 = get_hf_lora_targets(
    {"model_type": "glm_moe_dsa", "mlp_layer_types": ["sparse"], "n_shared_experts": 0, "q_lora_rank": 4}
)
_QWEN35_MOE = get_hf_lora_targets(
    {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "layer_types": ["full_attention"],
            "shared_expert_intermediate_size": 0,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 2,
        },
    }
)
_INKLING = get_hf_lora_targets({"model_type": "inkling_model", "mlp_layer_types": ["sparse"], "n_shared_experts": 0})


@pytest.mark.parametrize(
    "targets, layer",
    [(_GLM5, "model.layers.0"), (_QWEN35_MOE, "model.language_model.layers.0"), (_INKLING, "model.layers.0")],
    ids=["glm_moe_dsa", "qwen3_5_moe", "inkling"],
)
def test_mains_native_export_with_sglang_expert_names_reads_back(tmp_path, targets, layer):
    """main's native --save-hf stacks experts under sglang's w1 / w3 / w2 and keeps HF names otherwise."""
    g = torch.Generator().manual_seed(0)
    hf_by_sglang = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    tensors = {
        f"{layer}.self_attn.o_proj.lora_A.weight": torch.randn(2, 8, generator=g),
        f"{layer}.self_attn.o_proj.lora_B.weight": torch.randn(8, 2, generator=g),
    }
    for sglang in hf_by_sglang:
        tensors[f"{layer}.mlp.experts.{sglang}.lora_A.weight"] = torch.randn(3, 2, 8, generator=g)
        tensors[f"{layer}.mlp.experts.{sglang}.lora_B.weight"] = torch.randn(3, 8, 2, generator=g)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4.0}))

    assert classify_adapter_source(tmp_path, targets, load_optimizer=False) == AdapterSourceKind.PEFT
    weights = PeftAdapterReader(tmp_path, _run(), targets).read_weights()

    def hf_key(name: str) -> str:
        for sglang, hf in hf_by_sglang.items():
            name = name.replace(f".experts.{sglang}.", f".experts.{hf}.")
        return name.removesuffix(".weight")

    expected = {hf_key(name): value for name, value in tensors.items()}
    assert weights.keys() == expected.keys()
    assert all(torch.equal(weights[k], v) for k, v in expected.items())


_QWEN3_4B = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151936}


def _check_bases(source, target):
    assert_parametrization_matches_run(source, target, [], load_optimizer=False, has_optimizer_state=False)


def test_a_base_read_from_another_mount_matches_by_its_fingerprint():
    fingerprint = base_model_fingerprint(_QWEN3_4B | {"_name_or_path": "/root/models/Qwen3-4B"})
    saved = _run(base_model="/root/models/Qwen3-4B", base_model_fingerprint=fingerprint)
    _check_bases(saved, _run(base_model="/lustre/models/Qwen3-4B", base_model_fingerprint=fingerprint))
    _check_bases(saved, _run(base_model="Qwen/Qwen3-4B", base_model_fingerprint=fingerprint))


def test_a_base_of_another_architecture_is_refused_whatever_its_name():
    saved = _run(base_model="Qwen/Qwen3-4B", base_model_fingerprint=base_model_fingerprint(_QWEN3_4B))
    other = _run(
        base_model="Qwen/Qwen3-4B", base_model_fingerprint=base_model_fingerprint(_QWEN3_4B | {"hidden_size": 4096})
    )
    with pytest.raises(AdapterSourceError, match="is another model than run base"):
        _check_bases(saved, other)


def test_a_side_without_a_fingerprint_compares_where_the_base_was_read_from():
    run = _run(base_model="/root/models/Qwen3-4B", base_model_fingerprint=base_model_fingerprint(_QWEN3_4B))
    _check_bases(_run(base_model="/root/models/Qwen3-4B"), run)
    with pytest.raises(AdapterSourceError, match="adapter base '/lustre/models/Qwen3-4B' != run base"):
        _check_bases(_run(base_model="/lustre/models/Qwen3-4B"), run)


@pytest.mark.parametrize(
    "edit, refusal",
    [
        (lambda fields: fields | {"format_version": 2}, "format version 2; this miles reads version 1"),
        (
            lambda fields: fields | {"lora_dropout": 0.1},
            r"unknown fields \['lora_dropout'\].*load it with the miles version that wrote it",
        ),
        (lambda fields: {k: v for k, v in fields.items() if k != "lora_rank"}, r"lacks \['lora_rank'\]"),
    ],
    ids=["newer_format", "unknown_field", "missing_field"],
)
def test_a_parametrization_file_this_miles_cannot_read_exactly_is_refused(tmp_path, edit, refusal):
    run = _run(target_modules=QKV[:1])
    write_parametrization(tmp_path, run)
    assert read_parametrization(tmp_path) == run
    fields = json.loads((tmp_path / PARAMETRIZATION_FILENAME).read_text())
    (tmp_path / PARAMETRIZATION_FILENAME).write_text(json.dumps(edit(fields)))
    with pytest.raises(AdapterSourceError, match=refusal):
        read_parametrization(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        # config.json names the expert count num_experts; Qwen3MoeConfig.to_dict() names it num_local_experts
        dict(
            model_type="qwen3_moe",
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=128,
            tie_word_embeddings=False,
        ),
        # config.json leaves head_dim out; to_dict() derives it (for MLA not from hidden_size / heads)
        dict(
            model_type="llama",
            vocab_size=128256,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=False,
        ),
        dict(
            model_type="deepseek_v3",
            vocab_size=129280,
            hidden_size=7168,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_hidden_layers=61,
            num_attention_heads=128,
            num_key_value_heads=128,
            n_routed_experts=256,
            tie_word_embeddings=False,
        ),
    ],
    ids=lambda raw: raw["model_type"],
)
def test_config_json_and_its_to_dict_have_one_fingerprint(raw):
    resolved = transformers.AutoConfig.for_model(**raw).to_dict()
    assert base_model_fingerprint(raw) == base_model_fingerprint(resolved)


def test_a_remote_code_config_resolves_its_expert_count_aliases():
    common = {"model_type": "not_in_transformers", "hidden_size": 8}
    assert base_model_fingerprint(common | {"n_routed_experts": 4}) == base_model_fingerprint(
        common | {"num_local_experts": 4}
    )


def test_a_vl_wrapper_has_its_language_models_fingerprint():
    assert base_model_fingerprint({"model_type": "qwen3_vl", "text_config": _QWEN3_4B}) == base_model_fingerprint(
        _QWEN3_4B
    )


def test_a_source_s_alpha_is_read_from_the_kind_it_classifies_as(tmp_path):
    """A legacy Tinker checkpoint holding a stray run_config.yaml is legacy Tinker, so its alpha is META.json's."""
    (tmp_path / ".metadata").write_bytes(b"")
    (tmp_path / "META.json").write_text(json.dumps({"lora_alpha": 32}))
    (tmp_path / "run_config.yaml").write_text("peft:\n  alpha: 16\n")

    assert read_source_lora_alpha(tmp_path) == 32


def test_one_local_base_under_another_path_string_matches(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "model")
    source, run = _run(base_model=f"{tmp_path}/model/"), _run(base_model=str(tmp_path / "link"))
    assert_parametrization_matches_run(source, run, [], load_optimizer=False, has_optimizer_state=False)
