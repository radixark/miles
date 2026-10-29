"""Build the checkpoint's text model directly with Megatron Core."""

import json
from pathlib import Path

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from miles_plugins.models.muse_spark.config import map_to_megatron_fields, parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.model import MuseSpark1xGPTModel
from miles_plugins.models.muse_spark.transformer_config import MuseSpark1xTransformerConfig

_UNSUPPORTED_CONFIG_FIELDS = (
    "fp8",
    "fp4",
    "mtp_num_layers",
    "overlap_moe_expert_parallel_comm",
    "overlap_dispatch_backward_with_experts_wgrad",
    "tp_comm_overlap",
    "virtual_pipeline_model_parallel_size",
    "pipeline_model_parallel_layout",
    "overlap_p2p_comm",
    "moe_shared_expert_overlap",
    "cross_entropy_loss_fusion",
    "gradient_accumulation_fusion",
    "bias_activation_fusion",
    "bias_dropout_fusion",
    "masked_softmax_fusion",
    "fused_single_qkv_rope",
)


def validate_model_config(config):
    """Validate the pure MCore model contract, including checkpoint load callers."""
    spark = config.spark
    if config.params_dtype != torch.bfloat16:
        raise ValueError("Spark provider requires BF16 parameters")
    if config.attention_backend == AttnBackend.local:
        raise ValueError("Spark attention requires a Transformer Engine backend")
    if config.context_parallel_size > 1:
        if config.attention_backend == AttnBackend.unfused:
            raise ValueError("Spark context parallelism requires auto, fused, or flash attention")
        _validate_fused_context_parallel(config)
    if config.tensor_model_parallel_size > 1:
        if not config.sequence_parallel:
            raise ValueError("Spark tensor parallel training requires sequence parallelism")
        if spark.use_fp32_for_output:
            raise ValueError("Spark FP32 output head is not enabled with tensor parallelism")
        if spark.num_key_value_heads % config.tensor_model_parallel_size:
            raise ValueError("Spark KV heads must divide tensor parallel size")
    if config.recompute_granularity is not None:
        if (
            config.recompute_granularity != "full"
            or config.recompute_method != "uniform"
            or not config.recompute_num_layers
        ):
            raise ValueError("Spark recompute requires full uniform layer checkpointing")
    unsupported = [name for name in _UNSUPPORTED_CONFIG_FIELDS if getattr(config, name)]
    if unsupported:
        raise ValueError(f"unsupported parallel, fused, or auxiliary Spark mode: {unsupported}")
    if config.cuda_graph_impl != "none":
        raise ValueError("Spark provider requires CUDA graphs disabled")
    if not config.apply_rope_fusion:
        raise ValueError("Spark provider requires the validated fused RoPE path")
    if config.bf16 != (config.params_dtype == torch.bfloat16):
        raise ValueError("Spark provider BF16 flag differs from the parameter dtype")
    if config.hidden_dropout or config.attention_dropout:
        raise ValueError("Spark provider requires dropout disabled")
    expected_pipeline_dtype = torch.float32 if spark.residual_dtype == "fp32" else config.params_dtype
    if config.pipeline_dtype != expected_pipeline_dtype:
        raise ValueError("Spark pipeline dtype differs from the residual dtype")
    if config.qk_layernorm != spark.use_qk_norm:
        raise ValueError("Spark qk_layernorm differs from the checkpoint")
    for name, expected in map_to_megatron_fields(spark).items():
        value = getattr(config, name)
        if name == "moe_layer_freq" and isinstance(value, int):
            value = [int(value > 0 and index % value == 0) for index in range(config.num_layers)]
        if value != expected:
            raise ValueError(
                f"Spark provider field {name} differs from MuseSpark1xConfig: got {value!r}, expected {expected!r}"
            )


def _validate_fused_context_parallel(config):
    """Require a2a communication for Miles' THD packing."""
    comm = config.cp_comm_type
    comm_types = set(comm) if isinstance(comm, (list, tuple)) else {comm}
    if not comm_types <= {"a2a"}:
        # THD packing (Miles' training format) is not supported by TE's all_gather CP; p2p rejects sliding windows.
        raise ValueError("Spark context parallelism runs Miles' THD packing only with --cp-comm-type a2a")


def _validate_miles_args(args, spark):
    if getattr(args, "megatron_to_hf_mode", "raw") != "raw":
        raise ValueError("Muse Spark requires --megatron-to-hf-mode raw")
    for name in (
        "multi_latent_attention",
        "heterogeneous_layers_config_path",
        "rotary_seq_len_interpolation_factor",
        "use_rope_scaling",
        "enable_mtp_training",
        "enable_witness",
        "multi_lora_n_adapters",
        "use_lora",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Muse Spark does not support {name}")
    for name, expected in (
        ("vocab_size", spark.vocab_size),
        ("padded_vocab_size", spark.vocab_size),
        ("rotary_base", spark.rope_theta),
        ("rotary_percent", 1.0),
        ("position_embedding_type", "rope" if spark.use_rope else "none"),
        ("untie_embeddings_and_output_weights", not spark.tie_word_embeddings),
    ):
        value = getattr(args, name, None)
        if value is not None and value != expected:
            raise ValueError(f"Muse Spark argument {name} differs from the checkpoint")
    if (getattr(args, "seq_length", 0) or 0) > spark.max_position_embeddings:
        raise ValueError("Muse Spark sequence length exceeds the checkpoint position limit")


def _validate_process_groups(config, groups):
    for name, field_name in (
        ("tp", "tensor_model_parallel_size"),
        ("pp", "pipeline_model_parallel_size"),
        ("cp", "context_parallel_size"),
        ("ep", "expert_model_parallel_size"),
        ("expt_tp", "expert_tensor_parallel_size"),
    ):
        group = getattr(groups, name)
        if group is None or group.size() != getattr(config, field_name):
            raise ValueError(f"Spark {name} process group differs from the model config")


def muse_spark_1x_model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Miles custom provider; initial weights arrive through a native checkpoint."""
    args = get_args()
    with (Path(args.hf_checkpoint) / "config.json").open() as source:
        spark = parse_muse_spark_1x_config(json.load(source))
    _validate_miles_args(args, spark)
    config = core_transformer_config_from_args(args, config_class=MuseSpark1xTransformerConfig)
    config.spark = spark
    validate_model_config(config)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    _validate_process_groups(config, groups)
    return MuseSpark1xGPTModel(
        config,
        pg_collection=groups,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )


def muse_spark_1x_omni_model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Multimodal provider: the text model plus the frozen NaViT vision tower (Inkling pattern).

    A separate entry point instead of a CLI switch; VLM launch scripts pass it as
    --custom-model-provider-path. The tower is built on pre_process ranks from
    args.hf_checkpoint and merged into decoder_input by ``vision.mm_towers``.
    """
    from miles_plugins.models.muse_spark.vision.mm_towers import wire_mm_towers

    model = muse_spark_1x_model_provider(pre_process, post_process, vp_stage)
    wire_mm_towers(model, get_args().hf_checkpoint)
    return model
