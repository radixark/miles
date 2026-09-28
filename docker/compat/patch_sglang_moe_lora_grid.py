"""Make SGLang's MoE-LoRA mapping safe for coalesced request segments.

The GLM-5.3 inference image carries an SGLang fork where request segments may
span more tokens than ``max(extend_seq_lens_cpu)``.  In that case the fast
Triton launch is too small.  Fall back to the existing torch/searchsorted path;
it is slower only for the affected prefill and, unlike enlarging the Triton
grid, cannot address a segment past ``seg_indptr``.
"""

from pathlib import Path


SGLANG_BACKEND = Path(
    "/sgl-workspace/sglang/python/sglang/srt/lora/backend/base_backend.py"
)

BEFORE = """\
        tiles_per_segment = triton.cdiv(max_len, block_size)
        grid_size = tiles_per_segment * weight_indices.numel()
        assert grid_size * block_size >= num_tokens, (
            f\"MoE LoRA token-mapping launch under-covers tokens: \"
            f\"{grid_size=} {block_size=} {num_tokens=}\"
        )
        _compute_moe_lora_info_kernel[(grid_size,)](
            seg_indptr,
            lora_ranks,
            weight_indices,
            adapter_enabled,
            token_lora_mapping,
            weight_indices.numel(),
            max_len,
            BLOCK_SIZE=block_size,
        )
        return adapter_enabled, token_lora_mapping
"""

AFTER = """\
        tiles_per_segment = triton.cdiv(max_len, block_size)
        grid_size = tiles_per_segment * weight_indices.numel()
        if grid_size * block_size >= num_tokens:
            _compute_moe_lora_info_kernel[(grid_size,)](
                seg_indptr,
                lora_ranks,
                weight_indices,
                adapter_enabled,
                token_lora_mapping,
                weight_indices.numel(),
                max_len,
                BLOCK_SIZE=block_size,
            )
            return adapter_enabled, token_lora_mapping
        # Coalesced request segments can be longer than max_len. Enlarging this
        # grid would make pid_seg run past seg_indptr, so use the safe
        # searchsorted implementation below for these uncommon large prefills.
"""


def patch_backend(path: Path = SGLANG_BACKEND) -> None:
    source = path.read_text()
    if AFTER in source:
        return
    if BEFORE not in source:
        raise RuntimeError(f"SGLang MoE-LoRA launch block not found in {path}")
    path.write_text(source.replace(BEFORE, AFTER, 1))


if __name__ == "__main__":
    patch_backend()
