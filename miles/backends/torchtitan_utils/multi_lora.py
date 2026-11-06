"""Multi-LoRA slots on torchtitan projections: every projection ``LoRATransform`` adapts carries N slot adapters
pre-allocated at the max rank, each factor of each slot its own Parameter, and runs each slot on its contiguous range
of the (slot-sorted) tokens.

``lora_type="lora"`` keeps ``LoRATransform``'s fused adapter per slot (one A shared by q/k/v, one by gate/up);
``canonical_lora`` gives each HF projection of a fused ``wqkv`` / ``w13`` its own A and B, packed into the fused output.
"""

import functools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.lora import _LoRALinearMixin

from miles.backends.torchtitan_utils.lora_names import ComponentPacking


class _MultiLoRALinearMixin:
    """N slot adapters on a frozen Linear, each adding ``alpha / rank * B(A(x))`` to its slot's token rows.

    A slot claimed below the max rank keeps A's rows and B's columns past its rank at zero, and with them their
    gradients. A free slot has alpha 0. Every slot runs on every forward, an empty range included, so every slot
    parameter gets a gradient on every rank and FSDP reduces the same set everywhere.
    """

    lora_slots: nn.ModuleList

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]
        for param in nn.Module.parameters(self):  # type: ignore[arg-type]
            param.requires_grad_(False)
        self.packing: ComponentPacking | None = config.packing
        self.max_lora_rank = config.rank
        self.lora_ranks_by_slot = [config.rank] * config.num_slots
        self.lora_alphas_by_slot = [0.0] * config.num_slots
        self.token_counts_by_slot: tuple[int, ...] | None = None
        self.lora_slots = nn.ModuleList(_slot_adapter(config) for _ in range(config.num_slots))

    def _linear(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        base = super()._linear(input, weight, bias)  # type: ignore[misc]
        rows = input.reshape(-1, input.shape[-1])
        counts = self.token_counts_by_slot
        assert counts is not None, "multi-LoRA slots need AdapterSlots.route_tokens before a forward"
        assert sum(counts) == rows.shape[0], f"{rows.shape[0]} token rows, but {sum(counts)} routed"
        # as LoRATransform's adapters, the slots run inside this projection's remat region
        delta = torch.cat(
            [
                alpha / rank * self._slot_output(slot, slot_rows)
                for slot, slot_rows, alpha, rank in zip(
                    self.lora_slots,
                    rows.split(counts),
                    self.lora_alphas_by_slot,
                    self.lora_ranks_by_slot,
                    strict=True,
                )
            ]
        )
        return base + delta.view(base.shape)

    def _slot_output(self, slot: nn.Module, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            lora_b._linear(lora_a._linear(x, lora_a.weight, None), lora_b.weight, None)
            for lora_a, lora_b in zip(slot.lora_a, slot.lora_b, strict=True)
        ]
        return outputs[0] if self.packing is None else self.packing.pack(outputs)

    def slot_named_parameters(self, slot: int) -> Iterator[tuple[str, nn.Parameter]]:
        """``lora_a`` / ``lora_b`` for a fused adapter, ``lora_a.{i}`` / ``lora_b.{i}`` per canonical component."""
        for name, param in self.lora_slots[slot].named_parameters():
            factor, component, _ = name.split(".")
            yield (factor if self.packing is None else f"{factor}.{component}"), param

    @torch.no_grad()
    def init_slot(self, slot: int, lora_rank: int, lora_alpha: float) -> None:
        """Bind ``lora_rank`` / ``lora_alpha`` and zero A's rows and B's columns past ``lora_rank``."""
        assert 0 < lora_rank <= self.max_lora_rank, f"LoRA rank {lora_rank} is not in (0, {self.max_lora_rank}]"
        # alpha 0 marks a free slot
        assert lora_alpha > 0, f"LoRA alpha {lora_alpha} must be positive"
        for name, param in self.slot_named_parameters(slot):
            kept = torch.arange(self.max_lora_rank, device=param.device) < lora_rank
            mask = kept[:, None] if name.startswith("lora_a") else kept
            param.mul_(_placed_like(param, mask.to(param.dtype).expand(param.shape)))
        self.lora_ranks_by_slot[slot], self.lora_alphas_by_slot[slot] = lora_rank, float(lora_alpha)

    @torch.no_grad()
    def redraw_slot(self, slot: int, generator: torch.Generator) -> None:
        """A drawn at the max rank like a newly built adapter, B zero; every rank draws the whole A from
        ``generator``, so ranks holding the same rows agree."""
        for name, param in self.slot_named_parameters(slot):
            if name.startswith("lora_a"):
                drawn = torch.empty(param.shape, dtype=param.dtype, device=generator.device)
                nn.init.kaiming_uniform_(drawn, a=math.sqrt(5), generator=generator)
                param.copy_(_placed_like(param, drawn.to(param.device)))
            else:
                param.zero_()

    def clear_slot(self, slot: int, generator: torch.Generator) -> None:
        self.redraw_slot(slot, generator)
        self.lora_ranks_by_slot[slot], self.lora_alphas_by_slot[slot] = self.max_lora_rank, 0.0


def _placed_like(param: torch.Tensor, whole: torch.Tensor) -> torch.Tensor:
    """``whole`` (global-shaped, equal on every rank) placed like ``param``, without communication."""
    if not isinstance(param, DTensor):
        return whole
    return distribute_tensor(whole, param.device_mesh, param.placements, src_data_rank=None)


def _slot_adapter(config) -> nn.Module:
    """One slot's ``lora_a`` / ``lora_b`` ModuleLists: one fused adapter, or one per canonical component."""
    packing = config.packing
    if packing is None:
        if config.num_linears > 1:
            # as LoRATransform: a stacked projection's B inherits its sharding, its shared A is replicated
            a_sharding = _replicated() if config.sharding_config is not None else None
            b_sharding = (
                ShardingConfig(state_shardings=dict(config.sharding_config.state_shardings))
                if config.sharding_config is not None
                else None
            )
        else:
            a_sharding, b_sharding = _LoRALinearMixin._adapter_sharding(config.sharding_config)
        rows_by_component = [(config.out_features, config.num_linears)]
    else:
        # canonical components exist only for column-parallel fused projections: B shards over its own rows
        a_sharding = _replicated() if config.sharding_config is not None else None
        b_sharding = (
            ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.S(0))})
            if config.sharding_config is not None
            else None
        )
        rows_by_component = [(rows, 1) for rows in _component_rows(config, packing)]
    slot = nn.Module()
    slot.lora_a = nn.ModuleList(
        Linear.Config(
            in_features=config.in_features,
            out_features=config.rank,
            bias=False,
            inplace_wgrad_accum=config.inplace_wgrad_accum,
            sharding_config=a_sharding,
            param_init={"weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))},
        ).build()
        for _ in rows_by_component
    )
    slot.lora_b = nn.ModuleList(
        Linear.Config(
            in_features=config.rank,
            out_features=rows,
            num_linears=num_linears,
            bias=False,
            inplace_wgrad_accum=config.inplace_wgrad_accum,
            sharding_config=b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()
        for rows, num_linears in rows_by_component
    )
    return slot


def _replicated() -> ShardingConfig:
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


def _component_rows(config, packing: ComponentPacking) -> list[int]:
    if packing.kind == "stacked":
        return [config.out_features] * packing.num_components
    kv_rows = config.out_features // (packing.heads_per_kv + 2)
    return [kv_rows * packing.heads_per_kv, kv_rows, kv_rows]


@functools.cache
def get_multi_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """The multi-LoRA version of a Linear class, cached like torchtitan's ``get_lora_linear``."""

    class MultiLoRAProjection(_MultiLoRALinearMixin, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_cls.Config):  # type: ignore[misc, name-defined]
            rank: int
            num_slots: int
            packing: ComponentPacking | None = None

    MultiLoRAProjection.__name__ = f"MultiLoRA{parent_cls.__name__}"
    MultiLoRAProjection.__qualname__ = f"MultiLoRA{parent_cls.__name__}"
    return MultiLoRAProjection


def convert_to_slots(
    model_config: Module.Config, *, num_slots: int, canonical: bool, fused_projections: set[str]
) -> Module.Config:
    """``model_config`` after ``LoRATransform``, each adapted Linear turned into ``num_slots`` slot adapters.

    ``fused_projections``: the FQNs of projections holding several HF projections, which ``canonical`` splits.
    """
    assert num_slots > 0, f"multi-LoRA needs at least one slot, got {num_slots}"
    for fqn, cfg, parent, attr in list(model_config.traverse(Linear.Config)):
        owner = cfg._owner
        if not issubclass(owner, _LoRALinearMixin):
            continue
        # LoRATransform's class is (_LoRALinearMixin, parent class)
        base_cls = owner.__bases__[1]
        packing = _packing(fqn, cfg, parent, attr) if canonical and fqn in fused_projections else None
        values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init and f.name != "alpha"}
        new_cfg = get_multi_lora_linear(base_cls).Config(**values, num_slots=num_slots, packing=packing)
        if isinstance(parent, list):
            parent[attr] = new_cfg
        else:
            setattr(parent, attr, new_cfg)
    return model_config


def _packing(fqn: str, cfg: Linear.Config, parent, attr) -> ComponentPacking:
    if cfg.num_linears > 1:
        return ComponentPacking("stacked", cfg.num_linears)
    if isinstance(parent, QKVLinear.Config) and attr == "wqkv":
        return ComponentPacking(
            "qkv", 3, heads_per_kv=parent.n_heads // parent.n_kv_heads, head_dim=parent.head_dim
        )
    raise ValueError(f"canonical LoRA does not know how {fqn} fuses its HF projections")


def multi_lora_modules(model_parts: Sequence[nn.Module]) -> Iterator[tuple[str, _MultiLoRALinearMixin]]:
    for part in model_parts:
        for fqn, module in part.named_modules():
            if isinstance(module, _MultiLoRALinearMixin):
                yield fqn, module


def slot_adapter_parameters(
    model_parts: Sequence[nn.Module], slot: int
) -> tuple[dict[str, nn.Parameter], dict[str, ComponentPacking]]:
    """Slot ``slot``'s parameters keyed like a single LoRA's (``{projection}.lora_a.weight``, or
    ``{projection}.lora_a.{i}.weight`` per canonical component), and the canonical projections' packings."""
    params, packings = {}, {}
    for fqn, module in multi_lora_modules(model_parts):
        params |= {f"{fqn}.{name}.weight": param for name, param in module.slot_named_parameters(slot)}
        if module.packing is not None:
            packings[fqn] = module.packing
    return params, packings
