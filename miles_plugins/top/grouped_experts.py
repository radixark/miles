"""Delegate expert arithmetic while SequentialMLP retains parameter ownership."""

from miles_plugins.top.kernels.deepgemm_experts import deepgemm_expert_rows


def build_top_grouped_mlp():
    from megatron.core.transformer.moe.experts import SequentialMLP

    class TopGroupedMLP(SequentialMLP):
        def forward(
            self,
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            **kwargs,
        ):
            if kwargs:
                raise NotImplementedError(f"[top] grouped experts got unexpected {sorted(kwargs)}")
            if hasattr(self, "fp8_masters"):  # config-access-exempt: only TOP FP8 modules carry fp8_masters
                from miles_plugins.top.fp8_forward import trainable_expert_rows

                rows = trainable_expert_rows(
                    permuted_local_hidden_states,
                    self.fp8_masters,
                    tokens_per_expert,
                    swiglu_limit=getattr(
                        self, "top_swiglu_limit", None
                    ),  # config-access-exempt: only TOP FP8 modules carry top_swiglu_limit
                )
                return rows, None
            w1 = tuple(expert.linear_fc1.weight for expert in self.local_experts)
            w2 = tuple(expert.linear_fc2.weight for expert in self.local_experts)
            # Applying permuted_probs here would round weighted rows before the FP32 combine.
            rows = deepgemm_expert_rows(
                permuted_local_hidden_states.contiguous(),
                w1,
                w2,
                tokens_per_expert,
                gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
                swiglu_limit=getattr(
                    self, "top_swiglu_limit", None
                ),  # config-access-exempt: only TOP modules with a clamped SwiGLU carry top_swiglu_limit
            )
            return rows, None

    return TopGroupedMLP
