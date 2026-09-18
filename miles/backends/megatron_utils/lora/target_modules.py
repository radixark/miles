from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch.distributed as dist

from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_megatron


@dataclass(frozen=True)
class _TargetModule:
    megatron_module: str
    hf_modules: frozenset[str]


def resolve_hf_target_modules(hf_targets, mappings, *, canonical):
    """Resolve HF module patterns through Bridge, retaining fused-module boundaries."""
    targets = set(hf_targets)
    candidates = {}
    covered = set()
    for mapping in mappings:
        hf_params = mapping.hf_param
        hf_params = [hf_params] if isinstance(hf_params, str) else list(hf_params.values())
        if not all(name.endswith(".weight") for name in hf_params):
            continue
        hf_modules = {name.removesuffix(".weight") for name in hf_params}
        selected = targets & hf_modules
        if not selected:
            continue
        # Exact registry patterns preserve scope, including the expert wildcard in weight*.
        module, weight = mapping.megatron_param.rsplit(".", 1)
        assert weight in ("weight", "weight*"), f"Unsupported LoRA parameter: {mapping.megatron_param}"
        if canonical and len(hf_modules) > 1:
            assert module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1"), (
                f"CanonicalLoRA does not define split adapters for {module!r}"
            )
            for target in sorted(selected):
                leaf = convert_target_modules_to_megatron([target.rsplit(".", 1)[-1]])[0]
                name = f"{module.rsplit('.', 1)[0]}.{leaf}" if "." in module else leaf
                candidates[name] = _TargetModule(module, frozenset({target}))
        else:
            assert selected == hf_modules, (
                f"LoRA on fused module {module!r} requires all HF targets {sorted(hf_modules)}; "
                "use canonical_lora to select individual projections"
            )
            candidates[module] = _TargetModule(module, frozenset(selected))
        covered.update(selected)
    assert targets <= covered, f"HF LoRA targets have no Bridge mapping: {sorted(targets - covered)}"
    return candidates


def select_present_target_modules(model_chunks, candidates):
    local_names = {name for chunk in model_chunks for name, _ in chunk.named_modules()}
    present = {
        target
        for target, mapping in candidates.items()
        if any(fnmatchcase(name, mapping.megatron_module) for name in local_names)
    }
    # PP/EP ranks may own different projections; validate against the complete distributed model.
    present = _gather_set(present)
    expected = set().union(*(mapping.hf_modules for mapping in candidates.values()))
    covered = set().union(*(candidates[target].hf_modules for target in present))
    assert expected <= covered, f"HF LoRA targets have no Megatron modules: {sorted(expected - covered)}"
    return {target: mapping for target, mapping in candidates.items() if target in present}


def validate_hf_target_adapters(model_chunks, candidates):
    missing = set()
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if any(fnmatchcase(name, mapping.megatron_module) for mapping in candidates.values()):
                if not any(param.requires_grad for param in module.parameters()):
                    missing.add(name)
    missing = _gather_set(missing)
    assert not missing, f"LoRA injection skipped selected Megatron modules: {sorted(missing)}"


def _gather_set(local):
    if not dist.is_initialized():
        return local
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return set().union(*gathered)
