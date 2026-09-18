from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch.distributed as dist

from miles.utils.hf_lora_targets import matches_hf_lora_target


_CANONICAL_PROJECTIONS = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
}


@dataclass(frozen=True)
class _TargetModule:
    megatron_module: str
    selectors: frozenset[str]
    hf_modules: frozenset[str]


def _matches_megatron_target(module, target):
    return fnmatchcase(module if "." in target else module.rsplit(".", 1)[-1], target)


def _canonical_module(module, hf_target):
    leaf = _CANONICAL_PROJECTIONS[hf_target.rsplit(".", 1)[-1]]
    return f"{module.rsplit('.', 1)[0]}.{leaf}" if "." in module else leaf


def _match_target_modules(module, hf_modules, targets, *, canonical):
    selected, covered = set(), set()
    for target in targets:
        matched = {name for name in hf_modules if matches_hf_lora_target(name, target)}
        if _matches_megatron_target(module, target):
            matched.update(hf_modules)
        if canonical and len(hf_modules) > 1 and module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1"):
            matched.update(
                name for name in hf_modules if _matches_megatron_target(_canonical_module(module, name), target)
            )
        if matched:
            covered.add(target)
            selected.update(matched)
    return selected, covered


def resolve_megatron_lora_targets(targets, mappings, *, canonical, exclude_modules=()):
    """Map HF selectors or explicit Megatron names to adapters without widening fused selections."""
    candidates = {}
    covered = set()
    for mapping in mappings:
        module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        hf_params = mapping.hf_param
        hf_params = [hf_params] if isinstance(hf_params, str) else list(hf_params.values())
        # Packed expert mappings address HF parameters directly, without a .weight suffix.
        hf_modules = {name.removesuffix(".weight") for name in hf_params}
        selected, matched = _match_target_modules(module, hf_modules, targets, canonical=canonical)
        covered.update(matched)
        excluded, _ = _match_target_modules(module, hf_modules, exclude_modules, canonical=canonical)
        selected -= excluded
        if not selected:
            continue
        if canonical and len(hf_modules) > 1:
            assert module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1"), (
                f"CanonicalLoRA does not define split adapters for {module!r}"
            )
            for target in sorted(selected):
                candidates[_canonical_module(module, target)] = _TargetModule(
                    module, frozenset(matched), frozenset({target})
                )
        else:
            assert selected == hf_modules, (
                f"LoRA on fused module {module!r} requires all HF targets {sorted(hf_modules)}; "
                "use canonical_lora to select individual projections"
            )
            candidates[module] = _TargetModule(module, frozenset(matched), frozenset(selected))
    assert set(targets) <= covered, f"LoRA targets have no Bridge mapping: {sorted(set(targets) - covered)}"
    assert candidates, "No LoRA targets remain after applying --exclude-modules"
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
    # A leaf selector needs a match, not every optional layout declared by the registry.
    expected = set().union(*(mapping.selectors for mapping in candidates.values()))
    covered = set().union(*(candidates[target].selectors for target in present))
    assert expected <= covered, f"LoRA targets have no Megatron modules: {sorted(expected - covered)}"
    return {target: mapping for target, mapping in candidates.items() if target in present}


def validate_lora_target_adapters(model_chunks, candidates):
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


def configure_lora_targets(args):
    # Bridge is optional outside the Megatron backend.
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    model_bridge = bridge._model_bridge
    model_bridge.hf_pretrained = bridge.hf_pretrained
    candidates = resolve_megatron_lora_targets(
        args.target_modules,
        model_bridge.mapping_registry().get_all_mappings(),
        canonical=args.lora_type == "canonical_lora",
        exclude_modules=args.exclude_modules,
    )
    args.hf_lora_targets = sorted({target for module in candidates.values() for target in module.hf_modules})
