from miles.utils.args.schema import A, Arg, BaseConfig


class LoraConfig(BaseConfig):
    """Add LoRA-related arguments for Megatron backend."""

    lora_rank: A[int, Arg(help="LoRA rank. Set to 0 to disable LoRA (default: 0)")] = 0
    lora_alpha: A[int, Arg(help="LoRA alpha for scaling (default: 16)")] = 16
    lora_dropout: A[float, Arg(help="LoRA dropout rate (default: 0.0)")] = 0.0
    lora_type: A[
        str,
        Arg(
            choices=["lora", "canonical_lora"],
            help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
        ),
    ] = "lora"
    target_modules: A[
        str | None,
        Arg(
            help=(
                "LoRA targets: omit or use 'all-linear' for the HF model defaults, or provide "
                "comma-separated groups (attn,mlp,unembed), HF targets, or a mix of both. "
                "Megatron module names are also accepted by the Bridge backend."
            )
        ),
    ] = None
    exclude_modules: A[str | None, Arg(help="Modules to exclude from LoRA (comma-separated)")] = None
    lora_adapter_path: A[str | None, Arg(help="Path to load pre-trained LoRA adapter weights (default: None)")] = None
    lora_sync_from_tensor: A[
        bool,
        Arg(help="Sync LoRA weights via tensor instead of file (more efficient)"),
    ] = False
    lora_base_cpu_backup: A[
        bool,
        Arg(
            help=(
                "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
                "and skip per-step base sync. Trades host RAM for faster "
                "onload/offload. Ignored unless --colocate and LoRA are both on."
            )
        ),
    ] = False
    lora_train_only: A[
        bool,
        Arg(
            help=(
                "Train LoRA adapters in Megatron but keep rollout engines on the frozen "
                "base policy: SGLang LoRA serving and adapter weight sync are disabled "
                "(only the base weights are synced). For models without SGLang LoRA "
                "support (e.g. Inkling native LoRA)."
            )
        ),
    ] = False
    experts_shared_outer_loras: A[
        bool,
        Arg(
            help=(
                "Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
                "down lora_B shared across experts, expert_dim=1). Matches SGLang "
                "PR #21466's experts_shared_outer_loras=True serving contract."
            )
        ),
    ] = False
    multi_lora_n_adapters: A[
        int,
        Arg(
            help="Maximum number of concurrent adapter slots for multi-LoRA. Set to 0 to disable multi-LoRA (default: 0)"
        ),
    ] = 0
    check_lora_weight_equal: A[
        bool,
        Arg(
            help=(
                "Verify the megatron->sglang LoRA adapter weight-sync on the colocated "
                "(from_tensors) path: on every sync the trainer ships a per-tensor sha256 "
                "manifest of the adapter it sends, and each rollout engine hashes the "
                "tensors it received and fails the load on any mismatch/missing/extra "
                "name. The LoRA analogue of --check-weight-update-equal, which only "
                "covers base weights."
            )
        ),
    ] = False
