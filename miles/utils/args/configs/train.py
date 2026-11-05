import json
from typing import Any, Literal

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import A, Arg, BaseConfig


class TrainConfig(BaseConfig):
    trainer_id: str
    trainer_model_id: str | None
    entry: Literal["train", "serve"] = "train"

    train_backend: A[str, Arg(choices=["megatron", "fsdp", "torchtitan"], help="The backend for training.")] = (
        "megatron"
    )
    qkv_format: A[str, Arg(choices=["thd", "bshd"], help="The qkv layout.")] = "thd"
    linear_attention_backend: A[
        str,
        Arg(
            choices=["fla", "flashqla", "cake"],
            help=(
                "Backend for Qwen GDN linear-attention layers. "
                "'fla' (flash-linear-attention) is portable and runs on any supported GPU. "
                "'flashqla' (FlashQLA) requires NVIDIA SM90 (Hopper) or newer, CUDA 12.8+, and PyTorch 2.8+. "
                "'cake' is the Cake chunked forward/backward from the miles-ext-kernels submodule: "
                "--cake-gdn-variant det (bit-deterministic, the only variant with context parallelism) or fast (the "
                "Cake non-deterministic package, fixed-order; the default outside --deterministic-mode). It requires "
                "--use-miles-ext-kernels, NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin) and K = V = 128."
            ),
        ),
    ] = "fla"
    linear_attention_input_norm: A[
        str,
        Arg(
            choices=("te", "hf", "cake"),
            help=(
                "Input RMSNorm of the Qwen3.5 / Qwen3-Next GDN layers, zero-centred ((1 + w) scale, fp32 math) "
                "either way. 'te' is Transformer Engine's fused kernel; 'hf' is the eager HF Qwen3NextRMSNorm "
                "these layers used before, kept to reproduce its numerics (same math, slower). 'cake' is the Cake "
                "fused RMSNorm from the miles-ext-kernels submodule (--cake-rmsnorm-variant det or fast); it "
                "requires --use-miles-ext-kernels, NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin) and a hidden size "
                "the kernels serve; the layer refuses others when it is built (for example Qwen3.5-4B's 2560)."
            ),
        ),
    ] = "te"
    kda_backend: A[
        str,
        Arg(
            choices=("fla", "cake"),
            help=(
                "Backend for the KDA (Kimi Delta Attention) linear-attention layers of Kimi-K3 and GLM-5.3-Flash. "
                "'fla' (flash-linear-attention) is portable. 'cake' is the deterministic Cake chunked "
                "forward/backward from the miles-ext-kernels submodule (raw forget gate and q/k L2 norm in the "
                "kernels, BF16 beta, 1 to 64 heads per rank); it requires --use-miles-ext-kernels, --train-backend "
                "megatron, --qkv-format thd, no context parallelism and NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin)."
            ),
        ),
    ] = "fla"
    linear_attention_conv_backend: A[
        str,
        Arg(
            choices=("fla", "cake"),
            help=(
                "Backend for the causal depthwise short convolution (SiLU, kernel size 4) of the GDN and KDA "
                "linear-attention layers. 'fla' is flash-linear-attention's kernel. 'cake' is the Cake kernel from "
                "the miles-ext-kernels submodule (--cake-conv1d-variant det or fast, "
                "--cake-conv1d-silu-grad-preactivation bf16 or fp32); it requires --use-miles-ext-kernels, no "
                "context parallelism and NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin)."
            ),
        ),
    ] = "fla"
    linear_attention_output_norm_backend: A[
        str,
        Arg(
            choices=("fla", "cake"),
            help=(
                "Backend for the SiLU-gated output RMSNorm (RMSNorm(x) * w * SiLU(z), head dimension 128) of the GDN "
                "linear-attention layers. 'fla' is flash-linear-attention's kernel. 'cake' is the Cake kernel from "
                "the miles-ext-kernels submodule (--cake-gated-rmsnorm-variant det or fast); it requires "
                "--use-miles-ext-kernels and NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin). Layers whose output "
                "norm has a sigmoid gate (KDA, a sigmoid-gated Qwen3.8-Next config) refuse it when they are built."
            ),
        ),
    ] = "fla"
    mhc_backend: A[
        str,
        Arg(
            choices=("megatron", "cake"),
            help=(
                "Backend for the mHC (manifold-constrained hyper-connection) stream mixing of GLM-5.3-Flash and "
                "DeepSeek-V4. 'megatron' is Megatron's hyper-connection module. 'cake' is the Cake four-stream "
                "kernels from the miles-ext-kernels submodule (--cake-mhc-variant det or fast); it requires "
                "--use-miles-ext-kernels, NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin), the Megatron backend, "
                "hidden size 4096, 5120 or 7168 (4096 on SM107a), --hidden-dropout 0, no 'mhc' in "
                "--recompute-modules, no CUDA graphs and no --overlap-moe-expert-parallel-comm."
            ),
        ),
    ] = "megatron"
    full_attention_backend: A[
        str,
        Arg(
            choices=("megatron", "cake"),
            help=(
                "Backend for the core attention of the head-dim-256 full-attention layers of the Qwen3.5 / "
                "Qwen3-Next models (their MTP layer too). 'megatron' is Transformer Engine's attention. 'cake' is "
                "the Cake packed causal GQA forward/backward from the miles-ext-kernels submodule; it requires "
                "--use-miles-ext-kernels, NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin), the Megatron backend, "
                "--qkv-format thd, no context parallelism, no attention dropout or --window-size, and an even "
                "number of query heads per key/value head on each tensor-parallel rank. It sums dQ with FP32 "
                "atomic adds, so --deterministic-mode rejects it."
            ),
        ),
    ] = "megatron"
    true_on_policy_mode: A[bool, Arg(help="Whether to enable true-on-policy mode.")] = False
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
            choices=["megatron", "tilelang", "cake"],
            help=(
                "DSA sparse-MLA kernel backend for GLM (glm_moe_dsa) under --megatron-to-hf-mode bridge. "
                "'tilelang' (default) uses the fused TileLang kernels (SparseMLA + lighting_indexer, vendored from slime) for "
                "rollout<->train numerical parity; 'megatron' uses the portable unfused megatron-core "
                "kernels. 'tilelang' requires --qkv-format thd and the optional tilelang dep, and is "
                "training/forward-only (no KV cache, cannot serve inference). Both support GLM-5.1 and "
                "GLM-5.2, full or LoRA. Neither has an effect on non-DSA models or the 'raw' path, whose miles DSA "
                "layers keep their measured default kernels. 'cake' is for the 'raw' path only: the miles DSA "
                "plugin layers (GLM-5 / DeepSeek-V3.2 and GLM-5.3 with --dsa-impl miles, DeepSeek-V4 / V4.1 with "
                "--dsv4-impl miles) run their sparse attention and indexer scores on the Cake kernels from the "
                "miles-ext-kernels submodule (--cake-dsa-variant det or fast; the scores are always det). It "
                "requires --use-miles-ext-kernels, NVIDIA SM100a/SM103a/SM107a (Blackwell, Rubin), the Megatron "
                "backend, a 512-wide latent with a 0 or 64 RoPE tail and 8/16/32/64 indexer heads of dimension 128."
            ),
        ),
    ] = "tilelang"
    use_miles_ext_kernels: A[
        bool,
        Arg(
            help=(
                "Enable the kernels of the miles-ext-kernels submodule (3rdparty/miles-ext-kernels) and "
                "register their arguments. Required by every 'cake' backend."
            )
        ),
    ] = False
    # The values of the per-kernel arguments the miles-ext-kernels submodule registers under
    # --use-miles-ext-kernels, by destination name (None without the flag). The submodule owns their names; miles hands
    # them to miles_ext_kernels.configure() where it builds a 'cake' layer.
    miles_ext_kernels_args: dict[str, Any] | None = None
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
        CustomFunctionConfig | None,
        Arg(
            help=(
                "Path to a custom model provider function. "
                "If set, we will use this function instead of the default model provider. "
                "The function should have the signature "
                "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                "Example: 'my_module.my_model_provider'."
            ),
        ),
    ] = None
    recompute_loss_function: A[
        bool,
        Arg(help="Whether to enable recompute loss function to save memory during training."),
    ] = False
    log_probs_backend: A[
        str,
        Arg(
            choices=["torch", "fused"],
            help=(
                "How per-token log-probs and entropy are computed from the logits. 'torch' upcasts each "
                "response chunk to fp32 and keeps its softmax for backward. 'fused' streams the logits through "
                "Triton kernels and writes the gradient in place; --log-probs-chunk-size does not apply to it. "
                "Not only an implementation choice: 'fused' normalizes over the true vocabulary (--vocab-size),"
                " as the rollout engine does, while 'torch' normalizes over Megatron's padded vocabulary "
                "(except for --loss-type score_centering), so with vocab padding (TP > 1) the two give slightly"
                " different log-probs, ratios and KL. 'fused' does not support --true-on-policy-mode or "
                "sampling-support replay (--rollout-top-p below 1 or a positive --rollout-top-k)."
            ),
        ),
    ] = "torch"
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


class TrainRolloutOnlyConfig(BaseConfig):
    recompute_logprobs_via_prefill: A[
        bool,
        Arg(
            help=(
                "Recompute rollout logprobs via SGLang prefill instead of decode kernels. "
                "Only needed for models whose prefill and decode paths are not numerically identical."
            )
        ),
    ] = False
