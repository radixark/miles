from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
from torchtitan.models.base.module import Module
from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.linear import Linear

_RMS_EPS = 1e-6


class Inkling2RMSNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(out_dtype)


def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + _RMS_EPS)


def block_attn_res(
    blocks: list[torch.Tensor],
    partial: torch.Tensor | None,
    query: torch.Tensor,
    *,
    partial_addend: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depth-wise softmax over the completed residual blocks and the current partial one.

    Keys are the RMS-normalized sources, values are the RMS-normalized sources too
    (``attn_res_normalize_values``); the output is fp32.
    """
    sources = [*blocks, partial] if partial is not None else list(blocks)
    values = torch.stack([source.float() for source in sources])
    if partial_addend is not None:
        values = torch.cat([values[:-1], (values[-1] + partial_addend.float()).unsqueeze(0)])
    inv_rms = torch.rsqrt(values.square().mean(dim=-1) + _RMS_EPS)
    if values.shape[0] == 1:
        return values[0] * inv_rms[0].unsqueeze(-1)
    effective_query = query.float() * (values.shape[-1] ** -0.5)
    logits = torch.einsum("ntd,d,nt->tn", values, effective_query, inv_rms)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("tn,ntd->td", probabilities, values * inv_rms.unsqueeze(-1))


class ResidualQuery(Module):
    """A Block Attention Residual query (or ``count`` of them for full-attention Q/K/V inputs), kept in fp32."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        count: int = 1

    def __init__(self, config: Config):
        super().__init__()
        shape = (config.dim,) if config.count == 1 else (config.count, config.dim)
        self.query = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.query)

    def forward(self) -> torch.Tensor:
        # read through a call so FSDP unshards the query first; clone because FSDP frees the unsharded
        # storage as soon as this module's forward returns
        return self.query.clone()


class ShortConv(Module):
    """Causal depthwise convolution with a residual tap, scaled by ``(taps + 1) ** -0.5``.

    The dynamic variant (full-attention K/V) adds a per-token, per-head kernel predicted
    from the layer's normalized input to a static per-channel kernel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        channels: int
        kernel_size: int
        dynamic: bool = False
        n_heads: int = 1
        dynamic_input_dim: int = 0

    def __init__(self, config: Config):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.dynamic = config.dynamic
        self.n_heads = config.n_heads
        if config.dynamic:
            self.static_w = nn.Parameter(torch.empty(config.kernel_size, config.channels))
            self.weight_proj = nn.Parameter(torch.empty(config.n_heads * config.kernel_size, config.dynamic_input_dim))
            self.scale = (2 * config.kernel_size + 1) ** -0.5
        else:
            self.weight = nn.Parameter(torch.empty(config.channels, 1, config.kernel_size))
            self.scale = (config.kernel_size + 1) ** -0.5

    def reset_parameters(self) -> None:
        if self.dynamic:
            nn.init.zeros_(self.static_w)
            nn.init.zeros_(self.weight_proj)
        else:
            nn.init.zeros_(self.weight)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, dynamic_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_f = x.float()
        width = self.kernel_size
        if self.dynamic:
            dynamic = F.linear(dynamic_input.to(x.dtype), self.weight_proj.to(x.dtype)).float()
            dynamic = dynamic.view(x.shape[0], self.n_heads, width)
            static = self.static_w.float()
            head_dim = x.shape[-1] // self.n_heads
        else:
            static = self.weight.squeeze(1).to(x.dtype).float()
        acc = x_f
        for lag in range(width):
            shifted = x_f if lag == 0 else F.pad(x_f[:-lag], (0, 0, lag, 0))
            shifted = shifted * (positions >= lag).unsqueeze(-1)
            if self.dynamic:
                kernel = static[lag] + dynamic[:, :, lag].repeat_interleave(head_dim, dim=-1)
            else:
                kernel = static[:, width - 1 - lag]
            acc = acc + shifted * kernel
        return (acc * self.scale).to(x.dtype)


class QKL2Clip(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        threshold: float
        learnable_scale: bool

    def __init__(self, config: Config):
        super().__init__()
        self.threshold = config.threshold
        if config.learnable_scale:
            self.scale = nn.Parameter(torch.ones(config.num_heads, config.head_dim))
        else:
            self.scale = None

    def reset_parameters(self) -> None:
        if self.scale is not None:
            nn.init.ones_(self.scale)

    def forward(self, x_THD: torch.Tensor, output_scale_T: torch.Tensor | None = None) -> torch.Tensor:
        value = x_THD.float()
        if self.scale is not None:
            value = value * self.scale.float()
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        out = (value * (self.threshold / (norm + _RMS_EPS)).clamp(max=1.0)).to(x_THD.dtype)
        if output_scale_T is not None:
            out = (out.float() * output_scale_T.view(-1, 1, 1)).to(x_THD.dtype)
        return out


class RelLogitsProj(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        d_rel: int
        rel_extent: int

    def __init__(self, config: Config):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(config.d_rel, config.rel_extent))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj)

    def forward(self, r_THR: torch.Tensor) -> torch.Tensor:
        return torch.einsum("thd,de->the", r_THR, self.proj.to(r_THR.dtype))


@cache
def _compiled_flex():
    return torch.compile(flex_attention, dynamic=True)


@dataclass
class PackedDocs:
    """Document layout of one packed sequence: positions restart at 0 at every document."""

    positions: torch.Tensor
    doc_ids: torch.Tensor
    log_scaling_tau: torch.Tensor
    _masks: dict

    @classmethod
    def build(cls, positions: torch.Tensor, *, n_floor: int, alpha: float) -> "PackedDocs":
        doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1
        tau = 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))
        return cls(positions=positions, doc_ids=doc_ids, log_scaling_tau=tau, _masks={})

    def block_mask(self, window: int | None):
        if window not in self._masks:
            doc_ids = self.doc_ids

            def mask_mod(b, h, q_idx, kv_idx):
                keep = (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
                if window is not None:
                    keep = keep & (q_idx - kv_idx <= window)
                return keep

            length = doc_ids.shape[0]
            self._masks[window] = create_block_mask(mask_mod, None, None, length, length, device=doc_ids.device)
        return self._masks[window]


def relative_attention(
    q_THD: torch.Tensor,
    k_TKD: torch.Tensor,
    v_TKD: torch.Tensor,
    rel_THE: torch.Tensor,
    docs: PackedDocs,
    *,
    window: int | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(scale * q.k + rel[q, h, q - k]) v over causal same-document keys; returns (out fp32 [T,H,D], lse [T,H])."""
    extent = rel_THE.shape[-1]
    rel_HTE = rel_THE.permute(1, 0, 2).contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        bias = rel_HTE[h, q_idx, distance.clamp(0, extent - 1)]
        return torch.where((distance >= 0) & (distance < extent), score + bias.to(score.dtype), score)

    out, aux = _compiled_flex()(
        q_THD.permute(1, 0, 2).unsqueeze(0),
        k_TKD.permute(1, 0, 2).unsqueeze(0),
        v_TKD.permute(1, 0, 2).unsqueeze(0),
        score_mod=score_mod,
        block_mask=docs.block_mask(window),
        scale=scale,
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
    )
    return out[0].permute(1, 0, 2).float(), aux.lse[0].permute(1, 0)


class Inkling2Attention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        n_kv_heads: int
        head_dim: int
        d_rel: int
        is_local: bool
        sliding_window: int
        log_scaling_n_floor: int
        log_scaling_alpha: float
        wq_du: Linear.Config
        wk_dv: Linear.Config
        wv_dv: Linear.Config
        wr_du: Linear.Config
        gate_attn: Linear.Config
        attn_sink_proj: Linear.Config
        wo_ud: Linear.Config
        k_sconv: ShortConv.Config
        v_sconv: ShortConv.Config
        q_l2_clip: QKL2Clip.Config
        k_l2_clip: QKL2Clip.Config
        rel_l2_clip: QKL2Clip.Config
        rel_logits_proj: RelLogitsProj.Config
        v_norm_eps: float

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_rel = config.d_rel
        self.is_local = config.is_local
        self.window = config.sliding_window - 1 if config.is_local else None
        self.v_norm_eps = config.v_norm_eps
        self.wq_du = config.wq_du.build()
        self.wk_dv = config.wk_dv.build()
        self.wv_dv = config.wv_dv.build()
        self.wr_du = config.wr_du.build()
        self.gate_attn = config.gate_attn.build()
        self.attn_sink_proj = config.attn_sink_proj.build()
        self.wo_ud = config.wo_ud.build()
        self.k_sconv = config.k_sconv.build()
        self.v_sconv = config.v_sconv.build()
        self.q_l2_clip = config.q_l2_clip.build()
        self.k_l2_clip = config.k_l2_clip.build()
        self.rel_l2_clip = config.rel_l2_clip.build()
        self.rel_logits_proj = config.rel_logits_proj.build()

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, docs: PackedDocs) -> torch.Tensor:
        tokens = q_in.shape[0]
        q = self.wq_du(q_in)
        k = self.wk_dv(k_in)
        v = self.wv_dv(v_in)
        r = self.wr_du(q_in)
        gate = self.gate_attn(q_in)
        sinks = self.attn_sink_proj(q_in)
        dynamic_k, dynamic_v = (None, None) if self.is_local else (k_in, v_in)
        k = self.k_sconv(k, docs.positions, dynamic_k)
        v = self.v_sconv(v, docs.positions, dynamic_v)
        tau = None if self.is_local else docs.log_scaling_tau
        q = self.q_l2_clip(q.view(tokens, self.n_heads, self.head_dim), tau)
        k = self.k_l2_clip(k.view(tokens, self.n_kv_heads, self.head_dim))
        v = v.view(tokens, self.n_kv_heads, self.head_dim)
        v = F.rms_norm(v.float(), (self.head_dim,), eps=self.v_norm_eps).to(v.dtype)
        r = self.rel_l2_clip(r.view(tokens, self.n_heads, self.d_rel), tau)
        rel = self.rel_logits_proj(r)
        out, lse = relative_attention(q, k, v, rel, docs, window=self.window, scale=self.head_dim**-0.5)
        sink_scale = torch.sigmoid(lse.float() - sinks.float())
        out = out * torch.sigmoid(gate.float()).view_as(out) * sink_scale.unsqueeze(-1)
        return self.wo_ud(out.reshape(tokens, -1).to(q_in.dtype))


class SoftcappedSwiGLU(BinaryActivationFn):
    """softcap(gate) * sigmoid(gate) * softcap(up); the sigmoid sees the uncapped gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        gate_cap: float
        up_cap: float

    def __init__(self, config: Config) -> None:
        self.gate_cap = config.gate_cap
        self.up_cap = config.up_cap

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_f, up_f = gate.float(), up.float()
        capped_gate = self.gate_cap * torch.tanh(gate_f / self.gate_cap) if self.gate_cap > 0 else gate_f
        capped_up = self.up_cap * torch.tanh(up_f / self.up_cap) if self.up_cap > 0 else up_f
        return (capped_gate * torch.sigmoid(gate_f) * capped_up).to(gate.dtype)


class MessageController(Module):
    """Mixes a cross-pass message into a core layer input: x = (1 - tanh(forget)) x + tanh(gain) m."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        rank: int

    def __init__(self, config: Config):
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(config.rank, 2 * config.dim))
        self.gain_weight = nn.Parameter(torch.empty(config.dim, config.rank))
        self.forget_weight = nn.Parameter(torch.empty(config.dim, config.rank))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down_weight, std=0.02)
        nn.init.zeros_(self.gain_weight)
        nn.init.zeros_(self.forget_weight)

    def forward(self, destination: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        normalized = _rms_normalize(torch.stack((destination.float(), message.float()), dim=1))
        hidden = F.linear(normalized.flatten(1).bfloat16(), self.down_weight.bfloat16())
        gain = F.linear(hidden, self.gain_weight.bfloat16()).float()
        forget = F.linear(hidden, self.forget_weight.bfloat16()).float()
        return (1.0 - torch.tanh(forget)) * destination.float() + torch.tanh(gain) * message.float()
