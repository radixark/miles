"""TOP block for Qwen3.8-Flash-Next (HF ``qwen4_exp``): trainer TP1/CP1/PP1, experts over EP.

Starts from the model's own block spec, so parameter names and the checkpoint bridge are unchanged, and binds every
numerical operation to the rollout program: serving's chunk-prefill GDN and QSA full attention
(``TopQwen38Attention``), the Qwen DeepEP/DeepGEMM experts with the gated shared expert, the gated residual
(hyper-connection), and the PLE n-gram increment. The QSA, gated residual and PLE backwards differentiate FP32
references at the forward's inputs.
"""

from megatron.core.transformer.spec_utils import ModuleSpec

from miles.utils.hf_utils.config import register_hf_config_aliases


def qwen38_block(args, config, vp_stage, stamp, *, contract):
    from miles_plugins.models.qwen3_8_next.qwen3_8_next import get_qwen3_8_next_spec
    from miles_plugins.top import qwen38_hc, qwen38_ple, qwen38_qsa
    from miles_plugins.top.binding import enable_batch_invariant
    from miles_plugins.top.decode_split import bind_split
    from miles_plugins.top.gdn_spec import _bind_gated_shared_expert, gdn_text_config
    from miles_plugins.top.moe_ep import bind_moe_layer, validate_moe_config
    from miles_plugins.top.qwen38_attention import TopQwen38Attention

    if getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        raise NotImplementedError("[top] Qwen3.8-Flash-Next TOP does not train MTP")
    register_hf_config_aliases()
    hf = gdn_text_config(args, config)
    if hf.model_type != "qwen4_exp_text":
        raise ValueError("[top] the Qwen3.8-Flash-Next TOP spec needs a qwen4_exp text checkpoint")
    # 0: the full-attention layers run QSA, which takes no FA decode split.
    bind_split(hf.model_type, stamp)

    # SequentialMLP owns the per-expert parameters; TOP supplies the grouped DeepGEMM arithmetic.
    config.moe_grouped_gemm = False
    if config.moe_permute_fusion:
        config.moe_permute_fusion = False
        stamp.append(("moe_permute_fusion", "True", "False (slot metadata requires unfused permutation)"))
    validate_moe_config(config, hf.model_type)

    block = get_qwen3_8_next_spec(args, config, vp_stage)
    for layer_spec in block.layer_specs:
        layer_spec.submodules.self_attention = ModuleSpec(module=TopQwen38Attention, params={"hf_config": hf})
        layer_stamp = []
        bind_moe_layer(layer_spec, layer_stamp)
        _bind_gated_shared_expert(layer_spec)
    stamp.extend(layer_stamp)
    stamp.append(("hybrid_attention", "delegated", "serving's chunk prefill GDN / QSA full attention"))
    stamp.append(("hyper_connection", "delegate", qwen38_hc.install()))
    stamp.append(("ple", "delegate", qwen38_ple.install()))
    stamp.append(("qsa", "delegate", qwen38_qsa.install(**qwen38_qsa.rope_params_from_hf(args.hf_checkpoint))))
    enable_batch_invariant()
    return block
