from miles.utils.hf_config import load_hf_config


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert (
        args.target_modules is None and args.exclude_modules is None
    ), "Tinker uses --tinker-train-attn/mlp/unembed; --target-modules and --exclude-modules are not supported"
    modules = _resolve_target_modules(
        load_hf_config(args.hf_checkpoint),
        train_attn=args.tinker_train_attn,
        train_mlp=args.tinker_train_mlp,
        train_unembed=args.tinker_train_unembed,
    )
    # The common LoRA validator parses and validates this before trainer/engine initialization.
    args.target_modules = ",".join(modules)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True


def _resolve_target_modules(hf_config, *, train_attn, train_mlp, train_unembed):
    # Other architectures need their own complete attention/MLP mapping.
    assert hf_config.model_type in (
        "qwen3",
        "qwen3_moe",
    ), f"Tinker target layout is not defined for model_type={hf_config.model_type!r}"
    modules = []
    if train_attn:
        modules.extend(("q_proj", "k_proj", "v_proj", "o_proj"))
    if train_mlp:
        modules.extend(("gate_proj", "up_proj", "down_proj"))
    if train_unembed:
        modules.append("lm_head")
    assert modules, "Tinker requires at least one trainable LoRA module group"
    return modules
