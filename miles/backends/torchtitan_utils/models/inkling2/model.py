import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy, resolve_fsdp_mesh
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.moe import MoE

from miles.backends.torchtitan_utils.models.inkling2.layers import (
    Inkling2Attention,
    Inkling2RMSNorm,
    MessageController,
    PackedDocs,
    ResidualQuery,
    ShortConv,
    block_attn_res,
)
from miles.backends.torchtitan_utils.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter

_LOOP_MODULES = ("attn", "k", "v", "mlp")
_ROW_NORM_PARAM = re.compile(
    r"(?:^tok_embeddings\.weight|^lm_head\.weight"
    r"|attn\.(?:wq_du|wk_dv|wv_dv|wr_du|gate_attn|attn_sink_proj|wo_ud)\.weight"
    r"|feed_forward\.router\.gate\.weight|routed_experts\.(?:w13|w2)\.weight"
    r"|controllers\.(?:attn|k|v|mlp)\.down_weight)$"
)


class LoopQueries(Module):
    """Queries that read the previous core pass's residual blocks: one pair per core layer for the
    (attention-Q, MoE) inputs and one pair for the (K, V) inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_core_layers: int

    def __init__(self, config: Config):
        super().__init__()
        shape = (2 * config.num_core_layers, config.dim)
        self.queries = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.kv_queries = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.queries)
        nn.init.zeros_(self.kv_queries)

    def forward(self, bank: list[torch.Tensor], core_index: int, module: str) -> torch.Tensor:
        queries = self.queries if module in ("attn", "mlp") else self.kv_queries
        return block_attn_res(bank, None, queries[2 * core_index + int(module in ("mlp", "v"))])


class _Fp32OutMatmul(torch.autograd.Function):
    """x @ w.T in bf16 with an fp32 result; torch.mm(out_dtype=...) has no autograd formula."""

    @staticmethod
    def forward(ctx, x_TD: torch.Tensor, w_VD: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x_TD, w_VD)
        return torch.mm(x_TD, w_VD.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_TV: torch.Tensor):
        x_TD, w_VD = ctx.saved_tensors
        grad_TV = grad_TV.to(x_TD.dtype)
        grad_x = torch.mm(grad_TV, w_VD, out_dtype=torch.float32).to(x_TD.dtype)
        grad_w = torch.mm(grad_TV.t(), x_TD, out_dtype=torch.float32).to(w_VD.dtype)
        return grad_x, grad_w


class Fp32OutLinear(Module):
    """bf16 GEMM with an fp32 output, as the checkpoint's unembedding requests."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.out_features, config.in_features))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=self.weight.shape[1] ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Fp32OutMatmul.apply(x.reshape(-1, x.shape[-1]), self.weight.to(x.dtype)).view(*x.shape[:-1], -1)


class Inkling2Block(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        is_local: bool
        block_size: int
        # named as the module it builds, so LoRA targets (config paths) are state-dict paths
        attn: Inkling2Attention.Config
        input_norm: Inkling2RMSNorm.Config
        mlp_norm: Inkling2RMSNorm.Config
        feed_forward: MoE.Config
        attn_sconv: ShortConv.Config
        mlp_sconv: ShortConv.Config
        attn_query: ResidualQuery.Config | None
        mlp_query: ResidualQuery.Config
        controller: MessageController.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.is_local = config.is_local
        self.is_block_boundary = config.layer_id % config.block_size == 0
        self.attn = config.attn.build()
        if config.is_local:
            self.attn_norm = config.input_norm.build()
        else:
            self.q_input_norm = config.input_norm.build()
            self.k_input_norm = config.input_norm.build()
            self.v_input_norm = config.input_norm.build()
        self.mlp_norm = config.mlp_norm.build()
        self.feed_forward = config.feed_forward.build()
        self.attn_sconv = config.attn_sconv.build()
        self.mlp_sconv = config.mlp_sconv.build()
        self.attn_query = config.attn_query.build() if config.attn_query is not None else None
        self.mlp_query = config.mlp_query.build()
        self.controllers = None
        if config.controller is not None:
            self.controllers = ModuleDict({name: config.controller.build() for name in _LOOP_MODULES})

    def _attention_inputs(self, blocks: list[torch.Tensor], partial: torch.Tensor) -> list[torch.Tensor]:
        streams = 1 if self.is_local else 3
        if not blocks:
            return [partial] * streams
        query = self.attn_query()
        if self.is_local:
            return [block_attn_res(blocks, partial, query)]
        return [block_attn_res(blocks, partial, query[i]) for i in range(3)]

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial: torch.Tensor,
        docs: PackedDocs,
        messages: tuple[LoopQueries, list[torch.Tensor], int] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        streams = self._attention_inputs(blocks, partial)
        if messages is not None:
            loop_queries, bank, core_index = messages
            if self.is_local:
                streams = streams * 3
            streams = [
                self.controllers[name](stream, loop_queries(bank, core_index, name))
                for name, stream in zip(("attn", "k", "v"), streams, strict=True)
            ]
        if self.is_local and messages is None:
            normed = self.attn_norm(streams[0])
            q_in = k_in = v_in = normed
        elif self.is_local:
            q_in, k_in, v_in = (self.attn_norm(stream) for stream in streams)
        else:
            q_in = self.q_input_norm(streams[0])
            k_in = self.k_input_norm(streams[1])
            v_in = self.v_input_norm(streams[2])
        attention = self.attn_sconv(self.attn(q_in, k_in, v_in, docs), docs.positions)

        if self.is_block_boundary:
            mlp_input = block_attn_res([*blocks, partial], attention, self.mlp_query())
        else:
            mlp_input = block_attn_res(blocks, partial, self.mlp_query(), partial_addend=attention)
        if messages is not None:
            loop_queries, bank, core_index = messages
            mlp_input = self.controllers["mlp"](mlp_input, loop_queries(bank, core_index, "mlp"))
        mlp = self.mlp_sconv(self.feed_forward(self.mlp_norm(mlp_input)), docs.positions)

        if self.is_block_boundary:
            return [*blocks, partial], (attention.float() + mlp.float()).to(partial.dtype)
        return blocks, (partial.float() + attention.float() + mlp.float()).to(partial.dtype)


class Inkling2Model(Decoder):
    state_dict_adapter_cls = Inkling2StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        embed_norm: Inkling2RMSNorm.Config
        output_query: ResidualQuery.Config
        loop_queries: LoopQueries.Config | None
        core_start: int
        core_end: int
        num_passes: int
        unpadded_vocab_size: int
        final_logit_softcapping: float
        log_scaling_n_floor: int
        log_scaling_alpha: float
        activation_checkpoint: bool = True

        @property
        def first_full_attention(self) -> None:
            return None

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            core = self.core_end - self.core_start
            visits = len(self.layers) + (self.num_passes - 1) * core
            per_layer_active = (active_nparams - 2 * self.dim * self.vocab_size) / len(self.layers)
            return nparams, 6 * (per_layer_active * visits + self.dim * self.vocab_size)

    def __init__(self, config: Config):
        super().__init__(config)
        self.embed_norm = config.embed_norm.build()
        self.output_query = config.output_query.build()
        self.loop_queries = config.loop_queries.build() if config.loop_queries is not None else None
        self.core = range(config.core_start, config.core_end)
        self.num_passes = config.num_passes
        self.unpadded_vocab_size = config.unpadded_vocab_size
        self.final_logit_softcapping = config.final_logit_softcapping
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        self.activation_checkpoint = config.activation_checkpoint
        self._row_norm_targets: dict[str, torch.Tensor] = {}

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        for module in self.modules():
            if not isinstance(module, Module) or module._sharding_config is not None:
                continue
            states = [name for name, _ in module.named_parameters(recurse=False)]
            states += [name for name, _ in module.named_buffers(recurse=False)]
            if states:
                module._sharding_config = replicated_state_sharding_config(*states)
        super()._parallelize(parallelism_context)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        fp32_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
        )
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, parallelism_context.pp_enabled
        )
        offload_policy = CPUOffloadPolicy() if training.enable_cpu_offload else OffloadPolicy()
        fp32_modules = [self.output_query]
        if self.loop_queries is not None:
            fp32_modules.append(self.loop_queries)
        for block in self.layers.values():
            fp32_modules.append(block.mlp_query)
            if block.attn_query is not None:
                fp32_modules.append(block.attn_query)
        for module in fp32_modules:
            fully_shard(
                module,
                mesh=dp_mesh,
                dp_mesh_dims=dp_mesh_dims,
                mp_policy=fp32_policy,
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward,
            )
        super()._apply_fsdp(parallelism_context=parallelism_context, training=training, parallelism=parallelism)

    @torch.no_grad()
    def project_row_norms(self) -> None:
        """Rescale each pretrained-normalized projection row back to its norm at the first call.

        Rows are local under FSDP and EP (both shard dim 0), so this needs no communication. A frozen weight (the
        LoRA base) never moves, so it is skipped.
        """
        targets = self._row_norm_targets
        for name, param in self.named_parameters():
            if not param.requires_grad or not _ROW_NORM_PARAM.search(name):
                continue
            local = param.to_local() if isinstance(param, DTensor) else param
            if local.numel() == 0:
                continue
            norms = local.float().norm(dim=-1, keepdim=True)
            if name not in targets:
                targets[name] = norms
                continue
            local.mul_((targets[name] / norms.clamp(min=1e-12)).to(local.dtype))

    def _run_layer(self, layer, blocks, partial, docs, messages=None):
        if not (self.activation_checkpoint and torch.is_grad_enabled()):
            return layer(blocks, partial, docs, messages)
        count = len(blocks)
        bank = messages[1] if messages is not None else []

        def run(partial_, *tensors):
            blocks_ = list(tensors[:count])
            messages_ = None if messages is None else (messages[0], list(tensors[count:]), messages[2])
            new_blocks, new_partial = layer(blocks_, partial_, docs, messages_)
            return (new_partial, *new_blocks[count:])

        new_partial, *appended = checkpoint(run, partial, *blocks, *bank, use_reentrant=False)
        return [*blocks, *appended], new_partial

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata=None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert positions is not None and positions.ndim == 1, "Inkling2 trains on one packed sequence"
        docs = PackedDocs.build(positions, n_floor=self.log_scaling_n_floor, alpha=self.log_scaling_alpha)
        partial = self.embed_norm(self.tok_embeddings(tokens))
        blocks: list[torch.Tensor] = []
        layers = list(self.layers.values())
        for layer in layers[: self.core.start]:
            blocks, partial = self._run_layer(layer, blocks, partial, docs)
        prelude_blocks, prelude_partial = list(blocks), partial
        for pass_idx in range(self.num_passes):
            bank = None
            if pass_idx:
                bank = [*blocks[len(prelude_blocks) + 1 :], partial]
                blocks, partial = list(prelude_blocks), prelude_partial
            for core_index, layer in enumerate(layers[self.core.start : self.core.stop]):
                messages = None if bank is None else (self.loop_queries, bank, core_index)
                blocks, partial = self._run_layer(layer, blocks, partial, docs, messages)
        for layer in layers[self.core.stop :]:
            blocks, partial = self._run_layer(layer, blocks, partial, docs)
        hidden = self.norm(block_attn_res(blocks, partial, self.output_query()))
        if self._skip_lm_head:
            return hidden
        logits = self.lm_head(hidden)[..., : self.unpadded_vocab_size]
        cap = self.final_logit_softcapping
        return cap * torch.tanh(logits / cap) if cap else logits
