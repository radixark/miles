"""LoRA for the torchtitan backend: torchtitan's ``LoRATransform`` adapts exactly the projections miles' HF targets
name; ``lora_names`` names the trained adapters back as HF PEFT tensors."""

from torchtitan.config.transform import LoRATransform
from torchtitan.config.transform.apply import transform_model_config_
from torchtitan.models.base import BaseModel
from torchtitan.models.common.linear import Linear

from miles.backends.torchtitan_utils.lora_names import hf_export_names
from miles.utils.lora.utils import matches_lora_target


def convert_model_config(model_config: BaseModel.Config, *, hf_targets: list[str], rank: int, alpha: float):
    """``model_config`` with LoRA on exactly the Linear projections ``hf_targets`` name."""
    sd_adapter = model_config._owner.state_dict_adapter_cls(model_config, None)
    hf_names_by_projection = {
        fqn: hf_export_names(sd_adapter, fqn, _weight_shape(cfg))
        for fqn, cfg, _, _ in model_config.traverse(Linear.Config)
    }
    target_modules = _target_modules(hf_names_by_projection, hf_targets)
    return transform_model_config_(
        model_config, [LoRATransform(rank=rank, alpha=alpha, target_modules=sorted(target_modules))]
    )


def _weight_shape(cfg: Linear.Config) -> tuple[int, ...]:
    if cfg.num_linears > 1:
        return (cfg.num_linears, cfg.out_features, cfg.in_features)
    return (cfg.out_features, cfg.in_features)


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
            f"LoRA targets {missing} name no Linear projection the torchtitan model exports: MoE routed experts "
            "are grouped weights the torchtitan backend does not adapt yet (use --target-modules attn for MoE "
            "models), and a tied lm_head has no weight of its own"
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
