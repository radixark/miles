"""CPU checks for the Muse Spark raw provider and the Miles forward boundary."""

import json
import math
import os
from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
pytest.importorskip("transformer_engine.pytorch")

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from tests.fast.backends.megatron_utils.muse_spark_test_utils import build_model_config

from miles.backends.megatron_utils.parallel import get_packed_seq_params
from miles.backends.training_utils import parallel as training_parallel
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles_plugins.models.muse_spark import provider
from miles_plugins.models.muse_spark.config import MuseSpark1xConfig, parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.model import MuseSpark1xGPTModel
from miles_plugins.models.muse_spark.ops.moe import MuseSpark1xTopKRouter
from miles_plugins.models.muse_spark.ops.norms import MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale
from miles_plugins.models.muse_spark.spec import get_muse_spark_1x_spec


def _checkpoint_config(num_hidden_layers, **overrides):
    """Minimal ``MuseSparkForConditionalGeneration`` config (all layers RoPE + global attention, MoE)."""
    hidden, head_dim = 64, 16
    dtype = overrides.pop("dtype", "bfloat16")
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": dtype,
        "text_config": {
            "model_type": "muse_spark_text",
            "hidden_size": hidden,
            "num_hidden_layers": num_hidden_layers,
            "vocab_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": head_dim,
            "max_position_embeddings": 32,
            "intermediate_size": 2 * hidden,
            "layer_types": ["full_attention"] * num_hidden_layers,
            "layer_rope_theta": [500000.0] * num_hidden_layers,
            "mlp_layer_types": ["sparse"] * num_hidden_layers,
            "moe_hidden_size": hidden // 2,
            "moe_intermediate_size": 32,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "output_multiplier": (hidden / 256) ** -0.5,
            "qk_scale_factor": 43.7840518911 / math.sqrt(head_dim),
            "residual_gate_temperature": 0.3,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            **overrides,
        },
    }


@pytest.fixture
def spark():
    return MuseSpark1xConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
    )


def test_builds_core_config_without_bridge(spark):
    config = build_model_config(spark, params_dtype=torch.bfloat16)
    assert config.spark is spark
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.float32
    assert config.expert_tensor_parallel_size == 1
    assert config.gradient_accumulation_fusion is False
    assert config.apply_rope_fusion is True
    assert config.cross_entropy_loss_fusion is False
    assert all(not base.__module__.startswith("megatron.bridge") for base in type(config).__mro__)


def test_rejects_fp32_model_parameters(spark):
    config = build_model_config(spark)
    config.params_dtype = torch.float32
    with pytest.raises(ValueError, match="requires BF16 parameters"):
        provider.validate_model_config(config)
    for dtype in ("fp32", "float32"):
        with pytest.raises(ValueError, match="parameter_dtype must be bfloat16"):
            parse_muse_spark_1x_config(_checkpoint_config(2, dtype=dtype))


def test_rejects_non_fp32_combine(spark):
    """Without the FP32 combine, probs are applied inside the experts, before post_expert_norm."""
    spark = replace(spark, moe_is_enabled=True, num_local_experts=4, num_experts_per_tok=2, moe_router_in_fp32=True)
    config = build_model_config(spark)
    config.moe_combine_in_fp32 = False
    with pytest.raises(ValueError, match="moe_combine_in_fp32 differs from MuseSpark1xConfig"):
        provider.validate_model_config(config)


@pytest.mark.parametrize(
    "temperature",
    [
        {"attn_temperature_tuning_floor_scale": 8192},
        {"attn_temperature_tuning_layers": "global"},
        {"attn_temperature_tuning_q_scale_constant": 0.1},
        {
            "attn_temperature_tuning_floor_scale": 8192,
            "attn_temperature_tuning_layers": "global",
            "attn_temperature_tuning_q_scale_constant": 0.1,
        },
    ],
)
def test_rejects_position_temperature(temperature):
    with pytest.raises(ValueError, match="unsupported text_config fields: \\['attn_temperature_tuning_"):
        parse_muse_spark_1x_config(_checkpoint_config(2, **temperature))
    with pytest.raises(ValueError, match="position temperature is not supported"):
        MuseSpark1xConfig(**temperature)


def test_runtime_does_not_override_checkpoint_semantics(spark):
    with pytest.raises(ValueError, match="unsupported Spark runtime fields"):
        build_model_config(spark, hidden_size=128)
    config = build_model_config(spark)
    config.hidden_size = 128
    with pytest.raises(ValueError, match="hidden_size differs from MuseSpark1xConfig"):
        provider.validate_model_config(config)


def test_tp_requires_sequence_parallel(spark):
    with pytest.raises(ValueError, match="requires sequence parallelism"):
        build_model_config(spark, tensor_model_parallel_size=2)
    config = build_model_config(spark, tensor_model_parallel_size=2, sequence_parallel=True)
    assert config.tensor_model_parallel_size == 2
    assert config.sequence_parallel is True


def test_miles_accepts_sp_thd(spark):
    provider._validate_miles_args(
        SimpleNamespace(qkv_format="thd", sequence_parallel=True, tensor_model_parallel_size=2), spark
    )


def test_pipeline_spec_keeps_global_norm_indices_and_independent_residual_gates(spark):
    config = build_model_config(
        replace(
            spark,
            use_residual_gate=True,
            norm_zero_centered_gamma=True,
            post_norm_gain_center_type="inverse_residual_index_v2",
        ),
        pipeline_model_parallel_size=2,
    )
    layer = get_muse_spark_1x_spec(config, pp_rank=1).layer_specs[0].submodules
    attn = build_module(layer.self_attn_bda)
    ffn = build_module(layer.mlp_bda)
    assert attn.gate is not ffn.gate
    with torch.no_grad():
        attn.gate.fill_(1)
    assert torch.equal(ffn.gate, torch.zeros_like(ffn.gate))
    for spec, center in ((layer.post_self_attn_layernorm, 1 / 4), (layer.post_mlp_layernorm, 1 / 5)):
        norm = build_module(spec, config=config, hidden_size=spark.hidden_size, eps=spark.rms_norm_eps)
        assert norm.weight.shape == (spark.hidden_size,)
        assert norm.weight_offset == center
        assert norm.eps == spark.rms_post_norm_eps


def test_uneven_pp8_covers_all_62_layers_with_global_norm_indices(spark):
    config = build_model_config(
        replace(
            spark,
            num_hidden_layers=62,
            norm_zero_centered_gamma=True,
            post_norm_gain_center_type="inverse_residual_index_v2",
        ),
        pipeline_model_parallel_size=8,
        num_layers_in_first_pipeline_stage=7,
        num_layers_in_last_pipeline_stage=7,
    )
    covered = []
    for rank, count in enumerate((7, 8, 8, 8, 8, 8, 8, 7)):
        offset = get_transformer_layer_offset(config, pp_rank=rank)
        specs = get_muse_spark_1x_spec(config, pp_rank=rank).layer_specs
        assert len(specs) == count
        for index, layer in enumerate(specs, start=offset):
            covered.append(index)
            assert layer.submodules.post_self_attn_layernorm.params["weight_offset"] == 1 / (2 * index + 2)
            assert layer.submodules.post_mlp_layernorm.params["weight_offset"] == 1 / (2 * index + 3)
    assert covered == list(range(62))


def test_provider_forwards_decoder_first_and_last_layer_counts(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(_checkpoint_config(62)))
    monkeypatch.setattr(
        provider,
        "get_args",
        lambda: SimpleNamespace(
            hf_checkpoint=str(tmp_path),
            pipeline_model_parallel_size=8,
            decoder_first_pipeline_num_layers=7,
            decoder_last_pipeline_num_layers=7,
        ),
    )

    class ConfigCaptured(Exception):
        pass

    def capture_config(args, config_class):
        assert args.pipeline_model_parallel_size == 8
        assert args.decoder_first_pipeline_num_layers == 7
        assert args.decoder_last_pipeline_num_layers == 7
        raise ConfigCaptured

    monkeypatch.setattr(provider, "core_transformer_config_from_args", capture_config)
    with pytest.raises(ConfigCaptured):
        provider.muse_spark_1x_model_provider()


@pytest.mark.parametrize("input_scale", [False, True])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("output_norm_in_fp32", [False, True])
@pytest.mark.parametrize("centered", [False, True])
def test_norm_specs_preserve_formulas_and_dtype_boundaries(spark, input_scale, affine, output_norm_in_fp32, centered):
    spark = replace(
        spark,
        norm_affine=affine,
        post_norm_affine=affine,
        norm_zero_centered_gamma=centered,
        use_ffn_learnable_scales=input_scale,
        output_norm_in_fp32=output_norm_in_fp32,
    )
    config = build_model_config(spark, params_dtype=torch.bfloat16)
    spec = get_muse_spark_1x_spec(config, pp_rank=0)
    layer = spec.layer_specs[0].submodules
    slots = (
        layer.input_layernorm,
        layer.post_self_attn_layernorm,
        layer.pre_mlp_layernorm,
        layer.post_mlp_layernorm,
        spec.layer_norm,
    )
    norms = [
        build_module(slot, config=config, hidden_size=spark.hidden_size, eps=spark.rms_norm_eps) for slot in slots
    ]
    assert [type(norm) for norm in norms] == [
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale if input_scale else MuseSpark1xRMSNorm,
        MuseSpark1xRMSNorm,
    ]
    assert [norm.weight is not None for norm in norms] == [affine, affine, affine, affine or input_scale, affine]
    assert [norm.weight_offset for norm in norms] == ([1, 1, 1, 1, 0] if centered else [0] * 5)
    torch.manual_seed(37)
    stream = torch.randn(2, 3, spark.hidden_size)
    branch = stream.bfloat16()
    inputs = (stream, branch, stream, branch, stream)
    for norm, value in zip(norms, inputs, strict=True):
        value = value.clone().requires_grad_()
        output = norm(value)
        assert output.dtype == torch.bfloat16
        output.float().square().sum().backward()
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert list(norm.state_dict()) == (["weight"] if norm.weight is not None else [])
        if norm.weight is not None:
            assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
            assert norm.weight.sequence_parallel is config.sequence_parallel
    pre_ffn = norms[2]
    pre_ffn_weight = None if pre_ffn.weight is None else (pre_ffn.weight + pre_ffn.weight_offset).float()
    expected_pre_ffn = torch.nn.functional.rms_norm(
        branch.float(), (spark.hidden_size,), pre_ffn_weight, spark.rms_norm_eps
    ).bfloat16()
    torch.testing.assert_close(pre_ffn(stream), expected_pre_ffn, rtol=0, atol=0)
    final_input = stream if output_norm_in_fp32 else branch
    expected = torch.nn.functional.rms_norm(
        final_input.float(), (spark.hidden_size,), eps=spark.rms_norm_eps
    ).bfloat16()
    torch.testing.assert_close(norms[-1](stream), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["unfused", AttnBackend.unfused])
def test_requested_attention_backend_controls_actual_mcore_selection(spark, monkeypatch, backend):
    names = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    config = build_model_config(spark, attention_backend=backend)
    assert config.attention_backend is AttnBackend.unfused
    LanguageModule._set_attention_backend(SimpleNamespace(config=config))
    assert [os.environ[name] for name in names] == ["0", "0", "1"]


def test_invalid_attention_backend_is_rejected(spark):
    with pytest.raises(ValueError, match="unsupported Spark attention backend"):
        build_model_config(spark, attention_backend="invalid")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("megatron_to_hf_mode", "bridge", "raw"),
        ("padded_vocab_size", 256, "padded_vocab_size differs"),
    ],
)
def test_miles_rejects_unsupported_contracts(spark, field, value, message):
    with pytest.raises(ValueError, match=message):
        provider._validate_miles_args(SimpleNamespace(**{field: value}), spark)


def test_context_parallel_uses_backend_validation(spark):
    provider._validate_miles_args(SimpleNamespace(context_parallel_size=2), spark)
    runtime = dict(params_dtype=torch.bfloat16, attention_backend="fused", context_parallel_size=2)
    config = build_model_config(spark, cp_comm_type="a2a", **runtime)
    assert config.context_parallel_size == 2
    with pytest.raises(ValueError, match="context parallelism"):
        build_model_config(spark, cp_comm_type="p2p", **runtime)
    runtime["attention_backend"] = "auto"
    assert build_model_config(spark, cp_comm_type="a2a", **runtime).context_parallel_size == 2
    with pytest.raises(ValueError, match="context parallelism"):
        build_model_config(spark, cp_comm_type="p2p", **runtime)
    runtime["attention_backend"] = "unfused"
    with pytest.raises(ValueError, match="context parallelism"):
        build_model_config(spark, cp_comm_type="a2a", **runtime)
    with pytest.raises(ValueError, match="Transformer Engine backend"):
        build_model_config(spark, attention_backend="local")


def test_entrypoint_reads_checkpoint_and_current_groups(tmp_path, monkeypatch):
    source = _checkpoint_config(2)
    (tmp_path / "config.json").write_text(json.dumps(source))
    args = SimpleNamespace(
        hf_checkpoint=tmp_path, params_dtype=torch.bfloat16, seq_length=8, attention_backend=AttnBackend.unfused
    )
    group = SimpleNamespace(size=lambda: 1)
    groups = SimpleNamespace(**{name: group for name in ("tp", "pp", "cp", "ep", "expt_tp")})
    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(provider.ProcessGroupCollection, "use_mpu_process_groups", lambda: groups)
    monkeypatch.setattr(provider, "MuseSpark1xGPTModel", lambda config, **kwargs: (config, kwargs))
    monkeypatch.setattr(
        provider,
        "core_transformer_config_from_args",
        lambda args, config_class: build_model_config(
            provider.parse_muse_spark_1x_config(source), attention_backend=args.attention_backend
        ),
    )
    config, kwargs = provider.muse_spark_1x_model_provider()
    assert config.spark.num_hidden_layers == 2
    assert config.spark.max_position_embeddings == 32
    assert config.attention_backend is AttnBackend.unfused
    assert kwargs == {"pg_collection": groups, "pre_process": True, "post_process": True, "vp_stage": None}


def _forward_shell(spark):
    model = MuseSpark1xGPTModel.__new__(MuseSpark1xGPTModel)
    torch.nn.Module.__init__(model)
    model.config = build_model_config(spark)
    model.max_sequence_length = spark.max_position_embeddings
    return model


@pytest.mark.parametrize("name", ["inference_context", "inference_params", "sequence_len_offset", "output_processor"])
def test_rejects_inference_at_model_entry(spark, name):
    model = _forward_shell(spark)
    with pytest.raises(ValueError, match="supports full sequences without a KV cache"):
        model(input_ids=torch.tensor([[1, 2, 3]]), **{name: 0})


@pytest.mark.parametrize("fp32_output", [False, True])
def test_miles_bshd_forward_defaults_to_local_vocabulary(spark, monkeypatch, fp32_output):
    model = _forward_shell(spark)
    captured = {}

    def forward(self, input_ids, position_ids, attention_mask, decoder_input, labels, **kwargs):
        captured.update(position_ids=position_ids, **kwargs)
        return "output"

    monkeypatch.setattr(GPTModel, "forward", forward)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss_mask = torch.ones_like(tokens)
    assert model(input_ids=tokens, position_ids=None, loss_mask=loss_mask, fp32_output=fp32_output) == "output"
    assert captured["position_ids"] is None
    assert captured["runtime_gather_output"] is False
    assert captured["loss_mask"] is loss_mask
    assert "fp32_output" not in captured


def _native_preprocess_shell(model, monkeypatch, hidden=None, pre_process=True):
    model.pre_process = pre_process
    model.position_embedding_type = "rope"
    model.decoder = SimpleNamespace(input_tensor=None if pre_process else hidden)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    model.rotary_pos_emb = RotaryEmbedding(
        model.config.kv_channels,
        rotary_percent=1.0,
        rotary_interleaved=True,
        use_cpu_initialization=True,
        cp_group=group,
    )
    # Exercise native RoPE arithmetic on CPU, including its cached forward path.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    model.embedding_norm = torch.nn.Identity()

    class Embedding:
        scatter_to_sequence_parallel = True

        def __call__(self, **kwargs):
            return hidden

    model.embedding = Embedding()


def test_native_thd_without_padding_metadata_uses_full_rope_table(spark, monkeypatch):
    model = _forward_shell(spark)
    _native_preprocess_shell(model, monkeypatch, torch.zeros(8, 1, spark.hidden_size))
    cu = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=3, max_seqlen_kv=3)
    result = model._preprocess(torch.ones((1, 8), dtype=torch.long), None, packed_seq_params=packed)
    torch.testing.assert_close(result[1][:, 0, 0, 0], torch.arange(3, dtype=torch.float32))


def _miles_packed_batch(monkeypatch, lengths, tp_size):
    state = SimpleNamespace(tp=SimpleNamespace(size=tp_size), cp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(training_parallel, "_parallel_state", state)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    rollout = {
        "tokens": [torch.arange(length) for length in lengths],
        "loss_masks": [torch.ones(length - 1) for length in lengths],
        "total_lengths": list(lengths),
        "response_lengths": [length - 1 for length in lengths],
    }
    batch = get_batch(
        DataIterator(rollout, micro_batch_size=len(lengths)),
        list(rollout),
        pad_multiplier=8 // tp_size,
        qkv_format="thd",
        get_position_ids=True,
    )
    packed = get_packed_seq_params(batch, SimpleNamespace(qkv_format="thd"))
    return batch, packed


@pytest.mark.parametrize("lengths", [(4, 4), (3, 5), (3, 3), (2, 1)])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_miles_thd_uses_native_rope_for_packed_boundaries(spark, monkeypatch, lengths, tp_size):
    batch, packed = _miles_packed_batch(monkeypatch, lengths, tp_size)
    model = _forward_shell(spark)
    model.config = build_model_config(spark, tensor_model_parallel_size=tp_size, sequence_parallel=tp_size > 1)
    _native_preprocess_shell(model, monkeypatch, torch.zeros(8 // tp_size, 1, spark.hidden_size))

    def forward(self, tokens, positions, *args, **kwargs):
        return self._preprocess(tokens, positions, packed_seq_params=kwargs["packed_seq_params"])[1]

    monkeypatch.setattr(GPTModel, "forward", forward)
    assert packed.cu_seqlens_host[: len(lengths) + 1] == tuple(accumulate(lengths, initial=0))
    assert batch["packed_seq_params"] is packed
    implicit = model(input_ids=batch["tokens"], packed_seq_params=packed)
    explicit = model(input_ids=batch["tokens"], position_ids=batch["position_ids"], packed_seq_params=packed)
    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)
    torch.testing.assert_close(implicit[:, 0, 0, 0], torch.arange(packed.max_seqlen_q, dtype=torch.float32))
    assert not batch["input_loss_masks"][0, sum(lengths) :].any()
    assert all(batch["input_loss_masks"][0, sum(lengths[:index])] == 0 for index in range(len(lengths)))


@pytest.mark.parametrize("pre_process", [False, True])
def test_sp_thd_rope_table_covers_dummy_tail_on_each_pipeline_stage(spark, monkeypatch, pre_process):
    batch, packed = _miles_packed_batch(monkeypatch, (2, 1), 2)
    model = _forward_shell(spark)
    model.config = build_model_config(spark, tensor_model_parallel_size=2, sequence_parallel=True)
    hidden = torch.zeros((4, 1, spark.hidden_size))
    _native_preprocess_shell(model, monkeypatch, hidden, pre_process=pre_process)
    result = model._preprocess(batch["tokens"], None, packed_seq_params=packed)
    assert result[0] is None if not pre_process else result[0].shape == hidden.shape
    torch.testing.assert_close(result[1][:, 0, 0, 0], torch.arange(5, dtype=torch.float32))


def test_full_recompute_retains_each_packs_metadata_after_model_forward(spark, monkeypatch):
    model = _forward_shell(spark)
    model.config = build_model_config(
        spark,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    _native_preprocess_shell(model, monkeypatch)
    calls = []

    def layer(*, hidden_states, rotary_pos_emb, packed_seq_params, **kwargs):
        boundaries = packed_seq_params.cu_seqlens_host
        calls.append((boundaries, rotary_pos_emb[:, 0, 0, 0].tolist()))
        return hidden_states * (rotary_pos_emb[-1, 0, 0, 0] + boundaries[1]), None

    block = SimpleNamespace(config=model.config, num_layers_per_pipeline_rank=1, _get_layer=lambda index: layer)
    # Keep the actual MCore checkpoint/autograd path; the CPU test has only CPU RNG state.
    monkeypatch.setattr(tensor_parallel_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),))
    monkeypatch.setattr(tensor_parallel_random, "_set_all_rng_states", torch.set_rng_state)

    def forward(self, tokens, positions, attention_mask, decoder_input, labels, **kwargs):
        hidden, rotary = self._preprocess(
            tokens, positions, decoder_input, packed_seq_params=kwargs["packed_seq_params"]
        )[:2]
        return TransformerBlock._checkpointed_forward(
            block, hidden, attention_mask, None, None, rotary, None, kwargs["packed_seq_params"], False
        )

    monkeypatch.setattr(GPTModel, "forward", forward)
    inputs, outputs = [], []
    for lengths in ((3, 3), (4, 4), (3, 3)):
        batch, packed = _miles_packed_batch(monkeypatch, lengths, 2)
        hidden = torch.ones((4, 1, spark.hidden_size), requires_grad=True)
        inputs.append(hidden)
        outputs.append(model(input_ids=batch["tokens"], decoder_input=hidden, packed_seq_params=packed))
    for output in outputs:
        output.sum().backward()
    # TP2 pads each pack to a multiple of 8 tokens, so (3, 3) carries a 2-token tail segment
    assert calls == [
        ((0, 3, 6, 8), [0, 1, 2]),
        ((0, 4, 8), [0, 1, 2, 3]),
        ((0, 3, 6, 8), [0, 1, 2]),
        ((0, 3, 6, 8), [0, 1, 2]),
        ((0, 4, 8), [0, 1, 2, 3]),
        ((0, 3, 6, 8), [0, 1, 2]),
    ]
    torch.testing.assert_close(inputs[0].grad, torch.full_like(inputs[0], 5))
    torch.testing.assert_close(inputs[1].grad, torch.full_like(inputs[1], 7))
    torch.testing.assert_close(inputs[2].grad, torch.full_like(inputs[2], 5))


def test_output_transform_preserves_fp32_logits_and_local_vocabulary(spark):
    model = _forward_shell(replace(spark, output_soft_cap_temp=1.3))
    hidden = torch.tensor([[[0.125, -0.25]], [[0.5, 0.125]]], dtype=torch.bfloat16, requires_grad=True)
    weight = torch.tensor([[0.125, 0.5], [-0.5, 0.25]], dtype=torch.bfloat16)
    captures = []

    def output_layer(value, *, weight, runtime_gather_output):
        captures.append(runtime_gather_output)
        return torch.nn.functional.linear(value, weight), None

    logits = model._compute_language_model_output(
        hidden_states=hidden,
        output_layer=output_layer,
        output_weight=weight,
        labels=None,
        compute_language_model_loss=None,
        runtime_gather_output=False,
    )
    expected = (1.3 * torch.tanh(torch.nn.functional.linear(hidden, weight).float() / 1.3)).transpose(0, 1)
    assert logits.dtype == torch.float32
    assert captures == [False]
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    logits.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_float16_module_keeps_router_tensors_fp32(spark, monkeypatch):
    # The real BF16 wrapping path must skip every FP32 router tensor, including the token-count buffer.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    spark = replace(spark, moe_is_enabled=True, num_local_experts=4, num_experts_per_tok=2, moe_router_in_fp32=True)
    config = build_model_config(spark, params_dtype=torch.bfloat16, perform_initialization=False)
    groups = SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
    router = MuseSpark1xTopKRouter(config, groups)
    with torch.no_grad():
        router.weight.fill_(0.1234567)
        router.expert_bias.copy_(torch.tensor([0.01234567, -0.02345678, 0.03456789, -0.04567891]))
        router.local_tokens_per_expert.fill_(257.0)
    tensors = dict([*router.named_parameters(), *router.named_buffers()])
    assert {"weight", "expert_bias", "local_tokens_per_expert"} <= set(tensors)
    for name in ("weight", "expert_bias", "local_tokens_per_expert"):
        assert not torch.equal(tensors[name], tensors[name].bfloat16().float()), name
    before = {name: tensor.clone() for name, tensor in tensors.items()}

    model = torch.nn.ModuleDict({"router": router, "linear": torch.nn.Linear(4, 4)})
    Float16Module(config, model)
    assert model["linear"].weight.dtype == torch.bfloat16
    for name, tensor in [*router.named_parameters(), *router.named_buffers()]:
        assert tensor.dtype == torch.float32, name
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("topk,gate_weight", [(1, 2.0), (2, -80.0)])
def test_mcore_router_normalization_and_backward(spark, monkeypatch, topk, gate_weight):
    # Exercise the real router lifecycle with CPU allocations and the unfused Torch GEMM fallback.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(moe_utils, "te_general_gemm", None)
    spark = replace(
        spark,
        moe_is_enabled=True,
        num_local_experts=4,
        num_experts_per_tok=topk,
        moe_router_in_fp32=True,
        moe_router_score_division_eps=1e-15,
        metap_enabled=True,
        metap_router_has_one_inf_dim=True,
        metap_base_width=16.0,
    )
    config = build_model_config(spark, perform_initialization=False)
    assert config.moe_router_enable_expert_bias and config.freeze_e_score_correction_bias
    assert not config.moe_router_use_torch_mm and not config.moe_router_fusion
    groups = SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
    router = MuseSpark1xTopKRouter(config, groups)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:, 0] = gate_weight
        router.expert_bias.copy_(torch.tensor([2.0, 1.0, 0.0, 0.0]))
    x = torch.zeros(1, 1, spark.hidden_size)
    x[..., 0] = 1.0
    x.requires_grad_()
    probs, routing_map = router(x)
    assert routing_map.tolist() == [[index < topk for index in range(4)]]
    score = torch.sigmoid(torch.tensor(gate_weight * 0.5))
    expected = score if topk == 1 else score / (2 * score + 1e-20)
    torch.testing.assert_close(probs[0, :topk], expected.expand(topk))
    assert probs[0, topk:].count_nonzero() == 0
    probs.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(router.weight.grad).all()
    assert router.weight.grad[:topk, 0].count_nonzero() == topk
    assert router.weight.grad[topk:].count_nonzero() == 0
    if topk == 1:
        torch.testing.assert_close(router.weight.grad[0, 0], 0.5 * score * (1 - score))
    assert router.expert_bias.grad is None
    torch.testing.assert_close(router.expert_bias, torch.tensor([2.0, 1.0, 0.0, 0.0]))

    router.zero_grad(set_to_none=True)
    empty = torch.empty(0, 1, spark.hidden_size, requires_grad=True)
    empty_probs, empty_map = router(empty)
    assert empty_probs.shape == empty_map.shape == (0, 4)
    empty_probs.sum().backward()
    assert empty.grad.shape == empty.shape
    assert router.weight.grad.count_nonzero() == 0
