import json
from typing import Any

from miles.utils.args.schema import A, Arg, BaseConfig


class TrainConfig(BaseConfig):
    trainer_id: str
    trainer_model_id: str | None

    train_backend: A[str, Arg(choices=["megatron", "fsdp"], help="The backend for training.")] = "megatron"
    qkv_format: A[str, Arg(choices=["thd", "bshd"], help="The qkv layout.")] = "thd"
    linear_attention_backend: A[
        str,
        Arg(
            choices=["fla", "flashqla"],
            help=(
                "Backend for Qwen GDN linear-attention layers. "
                "'fla' (flash-linear-attention) is portable and runs on any supported GPU. "
                "'flashqla' (FlashQLA) requires NVIDIA SM90 (Hopper) or newer, CUDA 12.8+, and PyTorch 2.8+."
            ),
        ),
    ] = "fla"
    miles_dsa_topk_backend: A[
        str,
        Arg(choices=["torch", "flashinfer"], help="Top-k backend for Miles DSA indexer."),
    ] = "torch"
    true_on_policy_mode: A[bool, Arg(help="Whether to enable true-on-policy mode.")] = False
    recompute_logprobs_via_prefill: A[
        bool,
        Arg(
            help=(
                "Recompute rollout logprobs via SGLang prefill instead of decode kernels. "
                "Only needed for models whose prefill and decode paths are not numerically identical."
            )
        ),
    ] = False
    train_env_vars: A[
        Any,
        Arg(
            type_parser=json.loads,
            help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
        ),
    ] = "{}"
    train_memory_margin_bytes: A[
        int,
        Arg(help="Add margin for train memory allocation. By default we will reserve 1GB as margin."),
    ] = (
        1024**3
    )
    debug_skip_weight_update: A[
        bool,
        Arg(
            help=(
                "Debug-only: preserve the train/rollout offload-onload schedule, "
                "but skip the actual actor-to-rollout weight update."
            )
        ),
    ] = False
    debug_disable_optimizer: A[
        bool,
        Arg(
            help=(
                "Debug-only: do not initialize the Megatron optimizer or LR scheduler. "
                "Training still runs rollout, log-prob forward, and actor forward/backward, "
                "but skips optimizer state allocation and optimizer updates."
            )
        ),
    ] = False
    rematerialize_param_from_master_weight: A[
        bool,
        Arg(
            help=(
                "Colocate CPU memory optimization. Drop the actor's parameter weight backup "
                "during inference, and rebuild it from the optimizer's master weights on the "
                "next train step. Reduces peak CPU memory by 2*param per rank (bf16 training). "
                "Works with both the GPU optimizer and the CPU optimizer, but is not compatible "
                "with --use-precision-aware-optimizer on GPU. ref/teacher tags keep their "
                "backups. Recommended for Grace GPU colocate training."
            )
        ),
    ] = False
    check_rematerialize_param_from_master_weight: A[
        bool,
        Arg(help="Debug: SHA256-verify the first two rematerialize cycles are bit-identical."),
    ] = False
    megatron_to_hf_mode: A[
        str,
        Arg(
            type_parser=None,
            choices=["raw", "bridge"],
            help="The method to convert megatron weights to hugging face weights for SGLang.",
        ),
    ] = "raw"
    dsa_attention_backend: A[
        str,
        Arg(
            type_parser=None,
            choices=["megatron", "tilelang"],
            help=(
                "DSA sparse-MLA kernel backend for GLM (glm_moe_dsa) under --megatron-to-hf-mode bridge. "
                "'tilelang' (default) uses the fused TileLang kernels (SparseMLA + lighting_indexer, vendored from slime) for "
                "rollout<->train numerical parity; 'megatron' uses the portable unfused megatron-core "
                "kernels. 'tilelang' requires --qkv-format thd and the optional tilelang dep, and is "
                "training/forward-only (no KV cache, cannot serve inference). Both support GLM-5.1 and "
                "GLM-5.2, full or LoRA. No effect on non-DSA models or the 'raw' path."
            ),
        ),
    ] = "tilelang"
    extra_high_precision_layers_hf: A[
        list[str] | tuple[str, ...],
        Arg(
            type_parser=str,
            nargs="*",
            help=("Extra substrings for HF weight names to skip quantization " "(e.g. .kv_b_proj.)."),
        ),
    ] = ()
    extra_high_precision_layers_megatron: A[
        list[str] | tuple[str, ...],
        Arg(
            type_parser=str,
            nargs="*",
            help=(
                "Extra substrings for Megatron weight names to skip quantization in Megatron-to-HF paths "
                "(e.g. .linear_kv_up_proj.)."
            ),
        ),
    ] = ()
    custom_model_provider_path: A[
        str | None,
        Arg(
            help=(
                "Path to a custom model provider function. "
                "If set, we will use this function instead of the default model provider. "
                "The function should have the signature "
                "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                "Example: 'my_module.my_model_provider'."
            )
        ),
    ] = None
    recompute_loss_function: A[
        bool,
        Arg(help="Whether to enable recompute loss function to save memory during training."),
    ] = False
    log_probs_chunk_size: A[int, Arg(help="Chunk size to compute log probs to save memory")] = -1
    indep_dp: A[
        bool,
        Arg(
            help="Launch each DP replica as an independent Megatron instance instead of using Megatron-internal data parallelism."
        ),
    ] = False
    delay_split_train_data_by_dp: A[
        bool,
        Arg(
            help=(
                "Split the rollout batch across DP ranks on the training side instead of the rollout side, "
                "using the training side's own DP size."
            )
        ),
    ] = False
    allgather_cp: A[bool, Arg()] = False
    low_memory_resume: A[
        bool,
        Arg(
            reset=True,
            help=("Allocate optimizer states on CPU during checkpoint loading to prevent GPU OOM on memory spike. "),
        ),
    ] = False
    mfu_peak_tflops: A[
        float | None,
        Arg(
            help=(
                "Peak dense BF16 TFLOP/s of one training GPU — the denominator of perf/actor_train_mfu. "
                "Defaults to a built-in table keyed on the device name; set this for a device the table "
                "does not know, or to report MFU against another precision's peak. With neither available "
                "the MFU metric is not logged."
            )
        ),
    ] = None
