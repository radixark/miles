"""LoRA for the torchtitan backend: torchtitan's ``LoRATransform`` adapts exactly the projections miles' HF targets
name; ``lora_names`` names the trained adapters back as HF PEFT tensors.

Routed experts are grouped projections; their adapters take the shared-outer layout the engine serves
(``--experts-shared-outer-loras``), and a target that reaches them without it is refused here."""

from torchtitan.config.transform import LoRATransform
from torchtitan.config.transform.apply import transform_model_config_
from torchtitan.models.base import BaseModel
from torchtitan.models.common.linear import GroupedLinear, Linear

from miles.backends.torchtitan_utils.lora_names import grouped_hf_export_names, hf_export_names
from miles.utils.lora.utils import matches_lora_target


def convert_model_config(
    model_config: BaseModel.Config,
    *,
    hf_targets: list[str],
    layout: list[str],
    rank: int,
    alpha: float,
    experts_shared_outer: bool,
) -> BaseModel.Config:
    """``model_config`` with LoRA on exactly the projections ``hf_targets`` name; ``layout`` is every HF
    projection the model's LoRA layout serves (``args.hf_lora_layout``)."""
    sd_adapter = model_config._owner.state_dict_adapter_cls(model_config, None)
    hf_names_by_projection = {
        fqn: hf_export_names(sd_adapter, fqn, _weight_shape(cfg))
        for fqn, cfg, _, _ in model_config.traverse(Linear.Config)
    }
    grouped = {
        fqn: grouped_hf_export_names(sd_adapter, fqn, _grouped_weight_shape(cfg))
        for fqn, cfg, _, _ in model_config.traverse(GroupedLinear.Config)
    }
    target_modules = _target_modules({**hf_names_by_projection, **grouped}, hf_targets, layout)
    transform = {"rank": rank, "alpha": alpha, "target_modules": sorted(target_modules)}
    if target_modules & set(grouped):
        if not experts_shared_outer:
            raise ValueError(
                "LoRA on routed experts takes the shared-outer layout the engine serves; pass "
                "--experts-shared-outer-loras (targets reach "
                f"{sorted(name for fqn in target_modules & set(grouped) for name in grouped[fqn])})"
            )
        transform["expert_adapter_layout"] = "shared_outer"
    return transform_model_config_(model_config, [LoRATransform(**transform)])


def _weight_shape(cfg: Linear.Config) -> tuple[int, ...]:
    if cfg.num_linears > 1:
        return (cfg.num_linears, cfg.out_features, cfg.in_features)
    return (cfg.out_features, cfg.in_features)


def _grouped_weight_shape(cfg: GroupedLinear.Config) -> tuple[int, ...]:
    return (cfg.group_size, *_weight_shape(cfg))


def _is_targeted(hf_name: str, hf_targets: list[str]) -> bool:
    return any(matches_lora_target(hf_name, target) for target in hf_targets)


def _target_modules(
    hf_names_by_projection: dict[str, list[str]], hf_targets: list[str], layout: list[str]
) -> set[str]:
    """The projection FQNs whose HF projections are exactly the targeted ones.

    A fused projection shares one ``lora_A`` across the HF projections it holds, so it is adapted for all of them
    or for none; a target set that splits one fails here, before the model is built. A bare selector such as
    ``gate_proj`` also reaches projections the model's LoRA layout leaves out because the engine cannot serve
    them (Muse Glimmer's attention output gate); those fail here too.
    """
    exported = [name for names in hf_names_by_projection.values() for name in names]
    missing = [target for target in hf_targets if not any(matches_lora_target(name, target) for name in exported)]
    if missing:
        raise ValueError(
            f"LoRA targets {missing} name no projection the torchtitan model exports; a tied lm_head has no "
            "weight of its own"
        )

    targeted = {
        fqn: names for fqn, names in hf_names_by_projection.items() if any(_is_targeted(n, hf_targets) for n in names)
    }
    unservable = sorted(name for names in targeted.values() for name in names if not _is_targeted(name, layout))
    if unservable:
        raise ValueError(
            f"LoRA targets {hf_targets} reach {unservable}, which the model's LoRA layout leaves out because the "
            "engine cannot serve adapters there; name the projections explicitly or use attn, mlp or all-linear"
        )
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
