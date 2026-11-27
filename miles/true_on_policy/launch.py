"""Launch arguments of a TOP run, for scripts and recipes.

No admission here: the op registry admits the trainer layout while arguments are parsed and the serving
layout when an engine launches. ``--true-on-policy-contract`` also selects the engines' contract.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

from miles.true_on_policy.gdn import GDN_CHUNKED, GDN_MODE_ENV

_TRAINER = (
    "--true-on-policy-mode",
    "--deterministic-mode",
    "--true-on-policy-contract",
    TRUE_ON_POLICY_V1,
    "--spec",
    "miles_plugins.top.spec",
    "get_top_spec",
    "--transformer-impl",
    "local",
    "--use-cpu-initialization",
    "--no-bias-swiglu-fusion",
)
_SERVING = ("--sglang-enable-deterministic-inference",)
_GDN_TRAINER = ("--mtp-num-layers", "0", "--attention-backend", "flash", "--megatron-to-hf-mode", "raw")
_GDN_SERVING = (
    "--sglang-linear-attn-backend",
    "triton",
    "--sglang-mamba-ssm-dtype",
    "float32",
    "--sglang-page-size",
    "128",
    "--sglang-mamba-radix-cache-strategy",
    "extra_buffer",
    # A finished turn caches its state at the last decode snapshot, and the next turn re-prefills from there: a
    # snapshot every page instead of SGLang's default 256 halves the rows a turn prefills again.
    "--sglang-mamba-track-interval",
    "128",
    "--sglang-sampling-defaults",
    "openai",
)
_DEEPEP_TRAINER = ("--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", "deepep")
_DEEPEP_SERVING = (
    "--sglang-moe-a2a-backend",
    "deepep",
    "--sglang-deepep-mode",
    "auto",
    "--sglang-deepep-dispatcher-output-dtype",
    "bf16",
)
# GLM's DSA: serving's sparse FlashMLA on a cache of 64-token pages (FP8 E4M3 for a block-FP8 checkpoint, else BF16), FlashInfer's deterministic top-k (the
# trainer's selection reads the same tie-break settings), no indexer or top-k fusion, no dense-attention prefill.
_GLM_TRAINER = ("--miles-dsa-topk-backend", "flashinfer", "--attention-backend", "flash")
_GLM_SERVING = (
    "--sglang-page-size",
    "64",
    "--sglang-dsa-prefill-backend",
    "flashmla_sparse",
    "--sglang-dsa-decode-backend",
    "flashmla_sparse",
    "--sglang-dsa-topk-backend",
    "flashinfer",
)
_GLM_ENV = {
    "SGLANG_DISABLE_DSA_INDEXER_FUSION": "1",
    "SGLANG_DSA_FUSE_TOPK": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": "1",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "small",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
}
_GDN_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe", "qwen4_exp"})
# Qwen3.8-Flash-Next: the model's own provider publishes the PLE n-gram ids of each forward, and the raw weight
# converter is named by --model-name.
_QWEN38_TRAINER = (
    "--model-name",
    "qwen4_exp",
    "--custom-model-provider-path",
    "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider",
)
_QWEN38_ENV = {
    # Under attention DP every rank looks its own tokens up in a whole n-gram table (no cross-rank gather).
    "SGLANG_USE_ATTN_TP_NGRAM": "1",
    # Each DP rank and each trainer rank pins its own host n-gram table (~92 GB for Flash-Next); SGLang's default
    # membind to the GPU's NUMA node would put a node's ranks on one node's memory.
    "SGLANG_AUTO_NUMA_BIND": "0",
}
# DeepSeek-V4 (block-FP8 only): FP8 experts and the paged C4 indexer, the program validate_dsv4 admits. wo_a stays as
# serving resolves it (true_on_policy.dsv4.fp8_wo_a_program; the trainer reads the same decision).
_DSV4_ENV = {"SGLANG_DSV4_FP4_EXPERTS": "0", "SGLANG_OPT_DSV4_NONPAGED_INDEXER": "0"}
# DeepSeek-V4.1 (validate_dsv41): the release's MXFP4 experts cast to 128x128 FP8 at load (DeepGEMM's FP4 GEMM is
# SM100/SM120-only), eager prefill, radix reuse at whole pages (validate_dsv41_radix), prefill chunks covering the
# context; its layers are not recomputed.
_DSV41_ENV = {"SGLANG_DSV4_FP4_DEQUANT": "1", "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1"}
_DSV41_SERVING = (
    "--sglang-kv-cache-dtype",
    "fp8_e4m3",
    "--sglang-cuda-graph-backend-prefill",
    "disabled",
)
# GLM-5.3-Flash (true_on_policy.kda_chunked, glm5_next): the pinned KDA program in engines and trainer, Triton KDA
# kernels on FP32 states, radix reuse on the mamba track (every radix state on the 64-token grid), BF16 KV; the trainer
# loads the HF checkpoint directly and its converter is named by --model-name.
_GLM53_TRAINER = ("--mtp-num-layers", "0", "--megatron-to-hf-mode", "raw", "--model-name", "glm5_next")
_GLM53_SERVING = (
    "--sglang-linear-attn-backend",
    "triton",
    "--sglang-mamba-ssm-dtype",
    "float32",
    "--sglang-mamba-radix-cache-strategy",
    "extra_buffer",
    # One decode snapshot per 256-token page, SGLang's default interval, stated for the gate's hit report.
    "--sglang-mamba-track-interval",
    "256",
    "--sglang-kv-cache-dtype",
    "bfloat16",
    # The DSA program (true_on_policy.glm5_next_dsa): TileLang sparse MLA on both phases, 64-row index pages of 4-token
    # pooled keys.
    "--sglang-dsa-prefill-backend",
    "tilelang",
    "--sglang-dsa-decode-backend",
    "tilelang",
    "--sglang-page-size",
    "256",
)
_GLM53_ENV = {"SGLANG_TOP_KDA_MODE": "chunked", "FLA_DISABLE_BACKEND_DISPATCH": "1", "SGLANG_DSA_FUSE_TOPK": "0"}
_DEEPEP_MODEL_TYPES = frozenset(
    {"qwen3_5_moe", "qwen4_exp", "glm_moe_dsa", "glm5_next", "deepseek_v4", "deepseek_v41"}
)


# How TOP recipes run the trainer, and so the e2e gates: rollout log-probs are the old policy (bitwise the trainer's),
# so the actor runs no forward-only pass, and full activation recompute rebuilds the scored activations in backward.
TOP_RECIPE_TRAIN_ARGS = (
    "--use-rollout-logprobs",
    "--recompute-granularity",
    "full",
    "--recompute-method",
    "uniform",
    "--recompute-num-layers",
    "1",
)


class TopLaunch(NamedTuple):
    train_args: tuple[str, ...]
    env_vars: dict[str, str]

    @property
    def kv_cache_dtype(self) -> str | None:
        """The engine's KV cache dtype as these flags set it; None leaves SGLang's default."""
        args = list(self.train_args)
        return args[args.index("--sglang-kv-cache-dtype") + 1] if "--sglang-kv-cache-dtype" in args else None


def top_launch(*, model_type: str, engine_gpus: int = 1, fp8: bool = False) -> TopLaunch:
    """The flags and environment a TOP run of ``model_type`` needs, with ``engine_gpus`` GPUs per engine; ``fp8``
    for a block-FP8 checkpoint.

    Families with their own backends add theirs here. MoE trainers run EP over DeepEP; a one-GPU engine runs every
    expert locally, and an engine wider than one GPU runs one DP attention rank per GPU and EP across the engine.
    """
    args = list(_TRAINER + _SERVING)
    env = {
        "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    if model_type in _GDN_MODEL_TYPES:
        args += _GDN_TRAINER + _GDN_SERVING
        env[GDN_MODE_ENV] = GDN_CHUNKED
        env["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
        if model_type == "qwen4_exp":
            args += _QWEN38_TRAINER
            env.update(_QWEN38_ENV)
    elif model_type == "glm_moe_dsa":
        args += _GLM_TRAINER + _GLM_SERVING + ("--sglang-kv-cache-dtype", "fp8_e4m3" if fp8 else "bfloat16")
        env.update(_GLM_ENV)
    elif model_type == "glm5_next":
        args += _GLM53_TRAINER + _GLM53_SERVING
        env.update(_GLM53_ENV)
    elif model_type == "deepseek_v4":
        args += ("--sglang-kv-cache-dtype", "fp8_e4m3")
        env.update(_DSV4_ENV)
    elif model_type == "deepseek_v41":
        args += _DSV41_SERVING
        env.update(_DSV41_ENV)
    if model_type in _DEEPEP_MODEL_TYPES:
        args += ("--sglang-moe-runner-backend", "deep_gemm") + _DEEPEP_TRAINER
        if engine_gpus > 1:
            args += _DEEPEP_SERVING + ("--sglang-ep-size", str(engine_gpus))
        # The shared expert runs outside the routed slots; serving must not fuse it into the routed experts.
        args.append("--sglang-disable-shared-experts-fusion")
    if engine_gpus > 1:
        args += (
            "--sglang-enable-dp-attention",
            "--sglang-enable-dp-lm-head",
            "--sglang-dp-size",
            str(engine_gpus),
            "--sglang-moe-dense-tp-size",
            "1",
        )
    return TopLaunch(train_args=tuple(args), env_vars=env)
