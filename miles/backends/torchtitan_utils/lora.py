"""LoRA for the torchtitan backend: torchtitan's ``LoRAConverter`` adapts exactly the projections
miles' HF targets name; ``lora_names`` names the trained adapters back as HF PEFT tensors."""

from torchtitan.components.lora import LoRAConverter
from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
from torchtitan.models.common.linear import Linear

from miles.backends.torchtitan_utils.lora_names import hf_export_names, hf_projection_name
from miles.utils.lora.utils import matches_lora_target


def convert_model_config(model_spec, *, hf_targets: list[str], rank: int, alpha: float):
    """``model_spec.model`` with LoRA on exactly the Linear projections ``hf_targets`` name."""
    model_config = model_spec.model
    _unfuse_qkv_in_place(model_config)
    sd_adapter = model_spec.state_dict_adapter(model_config, None)
    target_modules = _converter_target_modules(model_config, sd_adapter, hf_targets)
    converter = LoRAConverter.Config(rank=rank, alpha=alpha, target_modules=sorted(target_modules)).build()
    return converter.convert(model_config)


def _unfuse_qkv_in_place(model_config) -> None:
    """Replace every fused ``wqkv`` in ``model_config`` with the stock separate projections.

    One adapter on a fused ``wqkv`` would share ``lora_A`` across q, k and v and
    has no HF name, while PEFT adapts each projection on its own. Both layouts
    checkpoint as ``wq``/``wk``/``wv``, so the base weights load unchanged.
    """
    for _, cfg, parent, attr in list(model_config.traverse(FusedQKVLinear.Config)):
        if type(cfg) is not FusedQKVLinear.Config:
            raise ValueError(f"LoRA cannot unfuse {type(cfg).__qualname__} into separate q/k/v projections")
        dim, bias = cfg.wqkv.in_features, cfg.wqkv.bias
        setattr(
            parent,
            attr,
            QKVLinear.Config(
                head_dim=cfg.head_dim,
                wq=Linear.Config(in_features=dim, out_features=cfg.n_heads * cfg.head_dim, bias=bias),
                wkv=Linear.Config(in_features=dim, out_features=cfg.n_kv_heads * cfg.head_dim, bias=bias),
            ),
        )


def _converter_target_modules(model_config, sd_adapter, hf_targets: list[str]) -> set[str]:
    """The ``LoRAConverter`` ``target_modules`` that adapt exactly the HF-targeted projections.

    The converter adapts every projection built from a ``Linear.Config`` whose last
    FQN segment it names, in every layer, so a target set that would adapt more or
    less than ``hf_targets`` fails here, before the model is built.
    """
    projections_by_config_name = [
        (fqn.rpartition(".")[2], _projections_built_by(fqn, parent, attr))
        for fqn, _, parent, attr in model_config.traverse(Linear.Config)
    ]
    hf_names_by_projection = {
        projection: hf_export_names(sd_adapter, projection)
        for _, projections in projections_by_config_name
        for projection in projections
    }
    exported_names = [name for names in hf_names_by_projection.values() for name in names]
    missing = [
        target for target in hf_targets if not any(matches_lora_target(name, target) for name in exported_names)
    ]
    if missing:
        raise ValueError(
            f"LoRA targets {missing} name no Linear projection the torchtitan model exports: MoE routed experts "
            "are grouped weights the pinned torchtitan cannot adapt (use --target-modules attn for MoE models), "
            "and a tied lm_head has no weight of its own"
        )

    targeted = {
        projection
        for projection, names in hf_names_by_projection.items()
        if any(matches_lora_target(name, target) for name in names for target in hf_targets)
    }
    target_modules = {name for name, projections in projections_by_config_name if set(projections) & targeted}
    adapted = {
        projection
        for name, projections in projections_by_config_name
        if name in target_modules
        for projection in projections
    }
    adapted_names = {projection: hf_projection_name(sd_adapter, projection) for projection in adapted}
    if adapted != targeted:
        extra = [adapted_names[projection] for projection in adapted - targeted]
        lacking = [name for projection in targeted - adapted for name in hf_names_by_projection[projection]]
        raise ValueError(
            "torchtitan's LoRAConverter adapts every projection built from a matched config name, in every "
            f"layer, so it cannot adapt exactly the requested HF targets {hf_targets}: it would also adapt "
            f"{_preview_hf_names(extra)} and miss {_preview_hf_names(lacking)}"
        )
    return target_modules


def _projections_built_by(config_fqn: str, parent: object, attr: str | int) -> list[str]:
    # QKVLinear builds both wk and wv from its one wkv config
    if isinstance(parent, QKVLinear.Config) and attr == "wkv":
        prefix = config_fqn.removesuffix(".wkv")
        return [f"{prefix}.wk", f"{prefix}.wv"]
    return [config_fqn]


def _preview_hf_names(names: list[str], limit: int = 4) -> str:
    shown = sorted(names) or ["nothing"]
    return ", ".join(shown[:limit]) + (f" (+{len(shown) - limit} more)" if len(shown) > limit else "")
