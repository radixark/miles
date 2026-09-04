"""Default ``slot -> HubKernelSpec`` mapping for ``--kernel-backend hub``.

A slot is a role in the FSDP forward, not a repo: ``models/qwen3_5.py`` asks for
``SLOT_CAUSAL_CONV1D`` and gets whatever repo the active mapping points that slot at. Swap the
whole mapping with ``--kernel-mapping-path``; the signature is ``(args) -> dict[str, HubKernelSpec]``.

Every repo here is under ``kernels-community``, which the Hub marks as a trusted kernel publisher,
so ``kernels.get_kernel`` accepts them without ``trust_remote_code``.
"""

from miles.backends.fsdp_utils.kernels.hub import HubKernelSpec

# Feeds GatedDeltaNet's linear-attention recurrence (Qwen3-Next / Qwen3.5 / Qwen3.6). This is the
# slot with the nastiest native failure mode: without `flash-linear-attention`, transformers binds
# `torch_chunk_gated_delta_rule`, whose signature ends in `**kwargs` -- so the `cu_seqlens` that
# `models/qwen3_5.py` injects is accepted and then silently ignored, and the recurrence runs across
# the whole packed row with no per-document reset and no warning.
SLOT_GATED_DELTA_RULE = "gated_delta_rule"

# Feeds GatedDeltaNet's short causal convolution (Qwen3-Next / Qwen3.5 / Qwen3.6). transformers
# binds these as instance attributes from the `causal_conv1d` wheel; without them it falls back to
# `F.silu(self.conv1d(...))`, which takes no `seq_idx` and so cannot reset conv state per packed
# document. transformers already routes nemotron_h at this same repo through `lazy_load_kernel`,
# but not the GatedDeltaNet architectures.
SLOT_CAUSAL_CONV1D = "causal_conv1d"

# Feeds the NemotronH attention mixer's varlen path. Without it the packed-document attention
# patch in `models/nemotron_h.py` returns the unpatched forward, and attention runs dense across
# document boundaries.
SLOT_FLASH_ATTN_VARLEN = "flash_attn_varlen"

# Pure Triton, so the Hub build is a single `torch-cuda` variant with no torch/CUDA/ABI matrix.
FLA = HubKernelSpec(
    repo_id="kernels-community/fla",
    version=1,
    functions=("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)

CAUSAL_CONV1D = HubKernelSpec(
    repo_id="kernels-community/causal-conv1d",
    version=1,
    functions=("causal_conv1d_fn", "causal_conv1d_update"),
)

FLASH_ATTN2 = HubKernelSpec(
    repo_id="kernels-community/flash-attn2",
    version=2,
    functions=("flash_attn_varlen_func",),
)


def default_module_kernels(args) -> dict[str, HubKernelSpec]:
    """The mapping miles ships. Empty under the deterministic run modes, which own their numerics."""
    if getattr(args, "true_on_policy_mode", False) or getattr(args, "deterministic_mode", False):
        return {}
    return {
        SLOT_GATED_DELTA_RULE: FLA,
        SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D,
        SLOT_FLASH_ATTN_VARLEN: FLASH_ATTN2,
    }
