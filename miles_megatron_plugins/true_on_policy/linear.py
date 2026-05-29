from __future__ import annotations

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from .matmul import sglang_reference_matmul
from .runtime import ensure_batch_invariant_mode_from_config


class SGLangColumnParallelLinear(ColumnParallelLinear):
    """Megatron local column-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
        ensure_batch_invariant_mode_from_config(self.config)
        bias = kwargs.pop("bias", None)
        return sglang_reference_matmul(
            input,
            weight,
            bias,
            gradient_accumulation_fusion=kwargs.pop("gradient_accumulation_fusion"),
            allreduce_dgrad=kwargs.pop("allreduce_dgrad"),
            sequence_parallel=kwargs.pop("sequence_parallel"),
            grad_output_buffer=kwargs.pop("grad_output_buffer", None),
            wgrad_deferral_limit=kwargs.pop("wgrad_deferral_limit", None),
            tp_group=kwargs.pop("tp_group", None),
            row_parallel=False,
        )


class SGLangRowParallelLinear(RowParallelLinear):
    """Megatron local row-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
        ensure_batch_invariant_mode_from_config(self.config)
        bias = kwargs.pop("bias", None)
        return sglang_reference_matmul(
            input,
            weight,
            bias,
            gradient_accumulation_fusion=kwargs.pop("gradient_accumulation_fusion"),
            allreduce_dgrad=kwargs.pop("allreduce_dgrad"),
            sequence_parallel=kwargs.pop("sequence_parallel"),
            grad_output_buffer=kwargs.pop("grad_output_buffer", None),
            wgrad_deferral_limit=kwargs.pop("wgrad_deferral_limit", None),
            tp_group=kwargs.pop("tp_group", None),
            row_parallel=True,
        )
