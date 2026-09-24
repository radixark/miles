"""The delta-rule recurrence as each model family calls fla: gate math, beta squashing and the kernel
flags, behind one ``DeltaRule`` interface the attention module drives."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from miles.kernels.attention.delta_rule.backend import get_chunk_gated_delta_rule, get_chunk_kda
from miles.kernels.attention.delta_rule.heads import DeltaRuleHeads


class DeltaRule(ABC):
    norm_activation: str
    dt_bias_dtype: torch.dtype | None = None

    @abstractmethod
    def dt_bias_size(self, heads: DeltaRuleHeads) -> int: ...

    @abstractmethod
    def __call__(self, q, k, v, beta_logits, decay, A_log, dt_bias, *, cu_seqlens, cp_context) -> torch.Tensor:
        """q/k ``[b, s, Gl, hk]``, v ``[b, s, Hl, hv]`` (fla groups value heads per key head) -> ``[b, s, Hl, hv]``."""


class GatedDeltaRule(DeltaRule):
    def __init__(self, backend: str = "fla", norm_activation: str = "silu"):
        self.backend = backend
        self.kernel = get_chunk_gated_delta_rule(backend)
        self.norm_activation = norm_activation

    def dt_bias_size(self, heads):
        return heads.num_v_heads

    def __call__(self, q, k, v, beta_logits, decay, A_log, dt_bias, *, cu_seqlens, cp_context):
        if cp_context is not None and self.backend != "fla":
            raise NotImplementedError(f"GDN context parallelism requires the 'fla' backend, got {self.backend!r}.")
        beta = beta_logits.sigmoid()
        g = -A_log.float().exp() * F.softplus(decay.float() + dt_bias)
        if self.backend == "flashqla":
            q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
        out, _ = self.kernel(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            **({"cp_context": cp_context} if cp_context is not None else {}),
        )
        return out


class KimiDeltaRule(DeltaRule):
    norm_activation = "sigmoid"
    dt_bias_dtype = torch.float32

    def __init__(self, gate_lower_bound: float):
        self.kernel = get_chunk_kda()
        self.gate_lower_bound = gate_lower_bound

    def dt_bias_size(self, heads):
        return heads.value_dim

    def __call__(self, q, k, v, beta_logits, decay, A_log, dt_bias, *, cu_seqlens, cp_context):
        boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
        out, _ = self.kernel(
            q=q,
            k=k,
            v=v,
            g=decay.reshape(v.shape),
            beta=beta_logits.float().sigmoid(),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            **boundaries,
        )
        return out
