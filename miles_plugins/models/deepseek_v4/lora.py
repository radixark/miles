"""LoRA helpers for DeepSeek-V4's grouped output projection."""

import torch


def apply_grouped_wo_a(
    module,
    input_: torch.Tensor,
    *,
    num_groups: int,
    group_output_dim: int,
) -> torch.Tensor:
    """Apply ``wo_a`` as independent projections for each output group.

    DeepSeek-V4 stores the group weights in one flattened column-parallel
    matrix, while the forward pass consumes only the matching input/output
    group. A standard LoRA wrapper broadcasts that flattened projection across
    every input group, so its delta must be reduced to the same group diagonal.
    """

    base_module = getattr(module, "to_wrap", module)
    weight = base_module.weight
    expected_output_dim = num_groups * group_output_dim
    if weight.ndim != 2 or weight.shape[0] != expected_output_dim:
        raise RuntimeError(
            "DeepSeek-V4 grouped wo_a weight mismatch: "
            f"got {tuple(weight.shape)}, expected [{expected_output_dim}, input_dim]"
        )
    if input_.shape[-2:] != (num_groups, weight.shape[1]):
        raise RuntimeError(
            "DeepSeek-V4 grouped wo_a input mismatch: "
            f"got {tuple(input_.shape)}, expected [..., {num_groups}, {weight.shape[1]}]"
        )

    grouped_weight = weight.view(num_groups, group_output_dim, weight.shape[1])
    output = torch.einsum("...gd,grd->...gr", input_, grouped_weight)

    adapter = getattr(module, "adapter", None)
    if adapter is None or not getattr(module, "_adapter_enabled", True):
        return output

    adapter_output = module.adapter_forward(adapter, input_.contiguous())
    expected_shape = (*input_.shape[:-1], expected_output_dim)
    if tuple(adapter_output.shape) != expected_shape:
        raise RuntimeError(
            "DeepSeek-V4 grouped wo_a LoRA output mismatch: "
            f"got {tuple(adapter_output.shape)}, expected {expected_shape}"
        )
    adapter_output = adapter_output.view(*input_.shape[:-2], num_groups, num_groups, group_output_dim)
    adapter_output = torch.diagonal(adapter_output, dim1=-3, dim2=-2).movedim(-1, -2)
    return output + adapter_output
