"""Qwen3.8-Flash-Next (``qwen4_exp``) for torchtitan.

The decoder is Qwen3.5's hybrid stack -- GatedDeltaNet linear attention, gated full
attention, a 512-expert MoE with a gated shared expert -- with three changes, each
reusing the Triton kernels and selection math the Megatron path already validated
against sglang:

- **Hyper-connections.** The residual is ``hc_count`` streams wide. Each block site
  reads one width-``dim`` input through a gated mean over the streams and writes its
  output back into every stream, weighted per stream. The checkpoint has no block
  norms: each hyper-connection's ``hc_norm`` is the pre-block norm, and a read-only
  one in the ``norm`` slot contracts the streams before ``lm_head``.
- **QSA.** Full-attention layers attend only the ``indexer_budget`` keys an indexer
  picks per query, plus the query's own partial block.
- **PLE.** One layer adds a per-layer n-gram embedding to the widened residual before
  its attention read; see ``ple.py``.

Tensor, context and pipeline parallelism are rejected: the PLE conv, the QSA indexer
and the hyper-connection streams assume each rank holds whole sequences, and the
pipeline path has never been run.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import VarlenMetadata, create_varlen_metadata_for_document
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.qwen3.sharding import _GROUPED_EXPERTS_PARAM_LAYOUT
from torchtitan.models.qwen3_5.model import Qwen35Attention, RMSNormGated
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.qwen3_8_next.ple import PerLayerEmbedding
from miles.kernels.attention.qsa import qsa_block_sparse_attention
from miles.kernels.hyper_connection.hc import hc_combine, hc_mix_inject
from miles_plugins.models.qwen3_8_next.ops.qsa_selection import (
    PackedBlockLayout,
    build_block_form,
    merge_tail_and_clip,
    project_indexer_qk,
    score_indexer_blocks,
    select_indexer_tokens,
)


class SigmoidGatedRMSNorm(RMSNormGated):
    """GatedDeltaNet's output norm, gated by ``sigmoid`` (``output_gate_type``) instead of ``silu``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNormGated.Config):
        pass

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # one fp32 pass and one rounding, like sglang's and Megatron's fused gated-norm kernels
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x * torch.sigmoid(gate.float())).to(input_dtype)


class HyperConnection(Module):
    """One read site over the ``n``-stream residual: ``[..., n*dim] -> ([..., dim], h_post [..., n])``.

    With ``inject`` it also returns the per-stream write weights for ``hc_combine``;
    without it, it is the model's output contraction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n: int
        lowrank: int
        inject: bool = True
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.n = config.n
        self.eps = config.eps
        wide = config.n * config.dim
        self.hc_norm = nn.Parameter(torch.empty(wide))
        self.w_down = nn.Parameter(torch.empty(config.lowrank, wide))
        self.w_up = nn.Parameter(torch.empty(wide, config.lowrank))
        self.w_inject = nn.Parameter(torch.empty(config.n, wide)) if config.inject else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hc_mix_inject(x, self.hc_norm, self.w_down, self.w_up, self.w_inject, self.n, self.eps)


class HeadContraction(HyperConnection):
    """The ``norm`` slot: contracts the streams to ``dim`` ahead of ``lm_head``."""

    @dataclass(kw_only=True, slots=True)
    class Config(HyperConnection.Config):
        inject: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed, _ = super().forward(x)
        return mixed


class QSAAttention(Qwen35Attention):
    """Qwen3.5's gated full attention, reading only the keys a QSA indexer selects.

    ``inner_attention`` stays a varlen config only so the trainer builds document
    offsets for the model; the attention itself runs the QSA kernel. The indexer's
    weights get no gradient: selection runs under ``no_grad``, as in Megatron.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Attention.Config):
        index_qk_proj: Linear.Config
        indexer_n_heads: int
        indexer_head_dim: int
        indexer_budget: int
        indexer_compress_ratio: int
        indexer_rope_theta: float
        indexer_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_qk_proj = config.index_qk_proj.build()
        self.index_q_norm = nn.Parameter(torch.empty(config.indexer_head_dim))
        self.index_k_norm = nn.Parameter(torch.empty(config.indexer_head_dim))
        self.indexer_n_heads = config.indexer_n_heads
        self.indexer_head_dim = config.indexer_head_dim
        self.indexer_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.indexer_eps = config.indexer_eps
        self.indexer_rope_theta = config.indexer_rope_theta

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: VarlenMetadata,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        assert B == 1, f"QSA runs on one packed sequence buffer, got batch {B}"
        positions_T = positions.reshape(-1)
        cu_seqlens = attention_masks.cu_seq_q

        with torch.no_grad():
            selection = self._select(x_BLD[0], positions_T, cu_seqlens)
            selection, seq_start = merge_tail_and_clip(selection, positions_T, self.compress_ratio)
            block_form = build_block_form(selection, positions_T, seq_start, cu_seqlens, self.compress_ratio)

        xq_gate = self.wq(x_BLD).view(B, L, self.n_heads, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)
        xk = self.wk(x_BLD).view(B, L, self.n_kv_heads, self.head_dim)
        xv = self.wv(x_BLD).view(B, L, self.n_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        xq_rot, xk_rot = self.rope(xq[..., : self.rotary_dim], xk[..., : self.rotary_dim], positions)
        xq = torch.cat([xq_rot, xq[..., self.rotary_dim :]], dim=-1)
        xk = torch.cat([xk_rot, xk[..., self.rotary_dim :]], dim=-1)

        out = qsa_block_sparse_attention(xq[0], xk[0], xv[0], *block_form[:-1], self.scaling, block_form[-1])
        out = out.view(B, L, self.n_heads, self.head_dim) * torch.sigmoid(gate)
        return self.wo(out.reshape(B, L, -1))

    def _select(self, x_TD: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        layout = PackedBlockLayout(cu_seqlens, positions, self.compress_ratio) if cu_seqlens.numel() > 2 else None
        q, block_k = project_indexer_qk(
            self.index_qk_proj(x_TD),
            self.index_q_norm,
            self.index_k_norm,
            positions,
            self._indexer_rope_table(x_TD.shape[0], x_TD.device),
            n_heads=self.indexer_n_heads,
            head_dim=self.indexer_head_dim,
            compress_ratio=self.compress_ratio,
            eps=self.indexer_eps,
            layout=layout,
        )
        logits = score_indexer_blocks(
            q,
            block_k,
            positions,
            head_dim=self.indexer_head_dim,
            compress_ratio=self.compress_ratio,
            layout=layout,
        )
        return select_indexer_tokens(
            logits,
            positions,
            block_topk=self.indexer_budget // self.compress_ratio,
            token_topk=self.indexer_budget,
            compress_ratio=self.compress_ratio,
            layout=layout,
        )

    def _indexer_rope_table(self, num_positions: int, device: torch.device) -> torch.Tensor:
        """Megatron's ``[max_pos, 1, 1, rotary_dim]`` table, ``cat(freqs, freqs)``, which ``apply_indexer_rope`` reads."""
        exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim
        inv_freq = 1.0 / (self.indexer_rope_theta**exponents)
        freqs = torch.outer(torch.arange(num_positions, dtype=torch.float32, device=device), inv_freq)
        return torch.cat([freqs, freqs], dim=-1)[:, None, None, :]


class Qwen38NextBlock(Module):
    """``x`` is the ``n*dim``-wide residual; each site reads it, runs, and writes back."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: QSAAttention.Config | None = None
        delta_net: Module.Config | None = None
        moe: Module.Config
        attn_hc: HyperConnection.Config
        mlp_hc: HyperConnection.Config
        ple: PerLayerEmbedding.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None
        self.attn = config.attention.build() if self.full_attn else config.delta_net.build()
        # apply_fsdp_to_decoder looks for this to give routed experts their own mesh under EP
        self.moe_enabled = True
        self.moe = config.moe.build()
        self.attn_hc = config.attn_hc.build()
        self.mlp_hc = config.mlp_hc.build()
        self.ple = config.ple.build() if config.ple is not None else None
        self.n = config.attn_hc.n

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: dict[str, VarlenMetadata],
        positions: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masks = attention_masks["deltanet"]
        if self.ple is not None:
            if tokens is None:
                raise RuntimeError("the PLE layer hashes input token ids, but this stage was given none")
            x = self.ple(x, tokens, masks.cu_seq_q)

        h, h_post = self.attn_hc(x)
        h = self.attn(h, masks, positions) if self.full_attn else self.attn(h, masks)
        x = hc_combine(x, h, h_post, self.n)

        h, h_post = self.mlp_hc(x)
        return hc_combine(x, self.moe(h), h_post, self.n)


class Qwen38NextModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        hc_count: int

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            for name, degree in (
                ("tensor", parallelism.tensor_parallel_degree),
                ("context", parallelism.context_parallel_degree),
            ):
                if degree > 1:
                    raise NotImplementedError(
                        f"Qwen3.8-Next does not support {name} parallelism under torchtitan: the PLE conv, "
                        "the QSA indexer and the hyper-connection streams need every token of a sequence"
                    )
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Qwen3.8-Next does not support pipeline parallelism under torchtitan: stages would exchange "
                    "the hyper-connection-wide residual, and the padding miles adds to fix PP shapes would reach "
                    "the QSA indexer and the PLE conv as an extra document; neither path has been run"
                )
            for layer in self.layers:
                if layer.ple is not None:
                    layer.ple.hf_checkpoint = config.hf_assets_path
                if parallelism.expert_parallel_degree > 1:
                    # read at build time, so it has to be set here rather than in parallelize
                    set_moe_sharding_config(
                        layer.moe, enable_ep=True, enable_sp=False, expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT
                    )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            return sum(p.numel() for p in model.parameters()), 0

    def __init__(self, config: Config):
        super().__init__(config)
        self.hc_count = config.hc_count

    def get_attention_masks(self, positions: torch.Tensor) -> dict[str, VarlenMetadata]:
        """One set of document offsets serves the GDN kernels, the PLE conv and QSA."""
        return {"deltanet": create_varlen_metadata_for_document(positions, include_host_offsets=True)}

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: dict[str, VarlenMetadata] | None = None,
    ) -> torch.Tensor:
        if self.tok_embeddings is not None:
            h = self.tok_embeddings(tokens).repeat(1, 1, self.hc_count)
            ple_tokens = tokens
        else:
            h, ple_tokens = tokens, None

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, tokens=ple_tokens)

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
