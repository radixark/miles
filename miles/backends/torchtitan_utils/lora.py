"""LoRA for the torchtitan backend: torchtitan's ``LoRATransform`` adapts exactly the projections miles' HF targets
name; ``lora_names`` names the trained adapters back as HF PEFT tensors.

Routed experts are adapted shared-outer (``lora_experts``) when the HF checkpoint stores each expert projection as one
stacked tensor (Inkling2's ``mlp.experts.w13_weight``); they are targeted as the packed ``experts.gate_up_proj`` /
``experts.down_proj``. A family whose checkpoint names every expert apart is not adapted.

The frozen base of every projection and embedding is stored in bf16: FSDP computes and all-gathers them in bf16
anyway, so only the fp32 copy they would otherwise keep is dropped. Adapters, norms and the fp32-policy residual
queries keep their dtypes."""

from dataclasses import fields

import torch
from torch import nn
from torchtitan.config.transform import LoRATransform
from torchtitan.config.transform.apply import transform_model_config_
from torchtitan.models.base import BaseModel
from torchtitan.models.base.module import Module
from torchtitan.models.common import Embedding
from torchtitan.models.common.linear import GroupedLinear, Linear

from miles.backends.torchtitan_utils.lora_experts import SharedOuterLoRAGroupedLinear
from miles.backends.torchtitan_utils.lora_names import GROUPED_LORA_RANK_MULTIPLE, hf_expert_target, hf_export_names
from miles.utils.lora.utils import matches_lora_target


def convert_model_config(
    model_config: BaseModel.Config,
    *,
    hf_targets: list[str],
    rank: int,
    alpha: float,
    experts_shared_outer: bool = False,
) -> BaseModel.Config:
    """``model_config`` with LoRA on exactly the projections ``hf_targets`` name, its frozen base in bf16."""
    sd_adapter = model_config._owner.state_dict_adapter_cls(model_config, None)
    hf_names_by_projection = {
        fqn: hf_export_names(sd_adapter, fqn, _weight_shape(cfg))
        for fqn, cfg, _, _ in model_config.traverse(Linear.Config)
    }
    experts = _stacked_expert_projections(model_config, sd_adapter)
    hf_names_by_projection |= {fqn: [target] for fqn, (target, _, _, _) in experts.items()}
    target_modules = _target_modules(hf_names_by_projection, hf_targets)
    adapted_experts = target_modules & experts.keys()
    if adapted_experts:
        _check_expert_lora(rank=rank, experts_shared_outer=experts_shared_outer)
    for fqn in adapted_experts:
        _, cfg, parent, attr = experts[fqn]
        setattr(parent, attr, _shared_outer_config(cfg, rank=rank, alpha=alpha))
    model_config = transform_model_config_(
        model_config,
        [LoRATransform(rank=rank, alpha=alpha, target_modules=sorted(target_modules - adapted_experts))],
    )
    _store_frozen_base_in_bf16(model_config)
    return model_config


def _check_expert_lora(*, rank: int, experts_shared_outer: bool) -> None:
    if not experts_shared_outer:
        raise ValueError(
            "torchtitan adapts routed experts shared-outer only (one lora_A for gate/up and one lora_B for down "
            "across the experts, the layout SGLang serves): pass --experts-shared-outer-loras, or leave the experts "
            "out of --target-modules"
        )
    if rank % GROUPED_LORA_RANK_MULTIPLE:
        raise ValueError(f"routed-expert LoRA needs --lora-rank divisible by {GROUPED_LORA_RANK_MULTIPLE}, got {rank}")


def _weight_shape(cfg: Linear.Config) -> tuple[int, ...]:
    if cfg.num_linears > 1:
        return (cfg.num_linears, cfg.out_features, cfg.in_features)
    return (cfg.out_features, cfg.in_features)


def _stacked_expert_projections(model_config: BaseModel.Config, sd_adapter) -> dict[str, tuple]:
    """fqn -> (packed HF target, config, parent, attr) of each routed-expert projection the checkpoint stacks."""
    found = {}
    for fqn, cfg, parent, attr in model_config.traverse(GroupedLinear.Config):
        shape = (
            cfg.group_size,
            *((cfg.num_linears,) if cfg.num_linears > 1 else ()),
            cfg.out_features,
            cfg.in_features,
        )
        names = hf_export_names(sd_adapter, fqn, shape)
        # experts the checkpoint names one by one would each need a slice of the shared factors
        if len(names) == 1:
            found[fqn] = (hf_expert_target(names[0], num_linears=cfg.num_linears), cfg, parent, attr)
    return found


def _shared_outer_config(cfg: GroupedLinear.Config, *, rank: int, alpha: float) -> SharedOuterLoRAGroupedLinear.Config:
    return SharedOuterLoRAGroupedLinear.Config(**_init_fields(cfg), rank=rank, alpha=alpha)


def _init_fields(cfg: Module.Config) -> dict:
    """The fields a config subclass is rebuilt from, as torchtitan's LoRATransform copies them."""
    return {
        f.name: getattr(cfg, f.name)  # config-access-exempt: copies a dataclass's own init fields
        for f in fields(cfg)
        if f.init
    }


def _is_targeted(hf_name: str, hf_targets: list[str]) -> bool:
    return any(matches_lora_target(hf_name, target) for target in hf_targets)


def _target_modules(hf_names_by_projection: dict[str, list[str]], hf_targets: list[str]) -> set[str]:
    """The projection FQNs whose HF projections are exactly the targeted ones.

    A fused projection shares one ``lora_A`` across the HF projections it holds, so it is adapted for all of them
    or for none; a target set that splits one fails here, before the model is built.
    """
    exported = [name for names in hf_names_by_projection.values() for name in names]
    missing = [target for target in hf_targets if not any(matches_lora_target(name, target) for name in exported)]
    if missing:
        raise ValueError(
            f"LoRA targets {missing} name no projection the torchtitan model exports: routed experts the HF "
            "checkpoint stores one per expert are not adapted (use --target-modules attn for those MoE models), "
            "and a tied lm_head has no weight of its own"
        )

    targeted = {
        fqn: names for fqn, names in hf_names_by_projection.items() if any(_is_targeted(n, hf_targets) for n in names)
    }
    split = {
        fqn: sorted(names) for fqn, names in targeted.items() if not all(_is_targeted(n, hf_targets) for n in names)
    }
    if split:
        fqn, names = next(iter(split.items()))
        raise ValueError(
            f"torchtitan fuses {names} into {fqn}, which shares one lora_A across them, so it cannot adapt only "
            f"some of them; target all of them or none (requested {hf_targets})"
        )
    return set(targeted)


_BF16_BASE_CONFIGS = (Linear.Config, GroupedLinear.Config, Embedding.Config)
_bf16_config_classes: dict[type, type] = {}


def _store_frozen_base_in_bf16(model_config: BaseModel.Config) -> None:
    # these configs are leaves, so replacing one never detaches a config the traversal still holds
    for _, cfg, parent, attr in list(model_config.traverse(Module.Config, recurse=True)):
        if parent is None or not isinstance(cfg, _BF16_BASE_CONFIGS):
            continue
        bf16_cls = _bf16_frozen_config_cls(type(cfg))
        new_cfg = bf16_cls(**_init_fields(cfg))
        if isinstance(parent, list):
            parent[attr] = new_cfg
        else:
            setattr(parent, attr, new_cfg)


def _bf16_frozen_config_cls(config_cls: type) -> type:
    if config_cls not in _bf16_config_classes:

        class Bf16FrozenConfig(config_cls):
            def build(self, **kwargs):
                module = config_cls.build(self, **kwargs)
                for name, param in list(module.named_parameters(recurse=False)):
                    if not param.requires_grad and param.is_floating_point():
                        setattr(module, name, nn.Parameter(param.detach().to(torch.bfloat16), requires_grad=False))
                return module

        Bf16FrozenConfig.__name__ = Bf16FrozenConfig.__qualname__ = f"Bf16{config_cls.__name__}"
        _bf16_config_classes[config_cls] = Bf16FrozenConfig
    return _bf16_config_classes[config_cls]
