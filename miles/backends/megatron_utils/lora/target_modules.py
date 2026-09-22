import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch.distributed as dist

from miles.utils.hf_utils.lora_targets import matches_hf_lora_target
from miles.utils.hf_utils.weight_mapping import HfWeightMapping

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
    checkpoint_parameters: frozenset[str]


def _matches_megatron_target(module, target):
    return fnmatchcase(module if "." in target else module.rsplit(".", 1)[-1], target)


def _canonical_adapter_module(module, checkpoint_parameter):
    leaf = checkpoint_parameter.removesuffix(".weight").rsplit(".", 1)[-1]
    assert leaf in _CANONICAL_PROJECTIONS, f"CanonicalLoRA has no split adapter for {checkpoint_parameter!r}"
    return f"{module.rsplit('.', 1)[0]}.{_CANONICAL_PROJECTIONS[leaf]}"


def _checkpoint_parameters(mapping):
    return {mapping.hf_param} if isinstance(mapping.hf_param, str) else set(mapping.hf_param.values())


def resolve_megatron_lora_targets(targets, mappings, *, parameter_names, hf_mapping, canonical, exclude_modules=()):
    candidates = {}
    covered_sources = set()
    visited = set()
    for mapping in mappings:
        module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        regex = re.compile(re.escape(mapping.megatron_param).replace(r"\*", "(.*)"))
        selected_by_parameter = {}
        for name in sorted(parameter_names - visited):
            match = regex.fullmatch(name)
            if match is None:
                continue
            visited.add(name)
            resolved = mapping.resolve(match.groups())
            sources = _checkpoint_parameters(resolved)
            hf_parameters = {source: hf_mapping.model_parameter(source) for source in sources}
            # Bridge can expose auxiliary layers, such as MTP, absent from the HF model.
            selected = {
                source
                for source, hf_parameter in hf_parameters.items()
                if (not hf_mapping.parameter_shapes or hf_parameter in hf_mapping.parameter_shapes)
                and any(matches_hf_lora_target(hf_parameter.removesuffix(".weight"), target) for target in targets)
            }
            selected -= {
                source
                for source in selected
                if any(
                    matches_hf_lora_target(hf_parameters[source].removesuffix(".weight"), target)
                    or _matches_megatron_target(module, target)
                    for target in exclude_modules
                )
            }
            selected_by_parameter[name] = (sources, selected)
        if not any(selected for _, selected in selected_by_parameter.values()):
            continue
        # One adapter wraps the whole grouped module; template injection cannot select individual layers/experts.
        assert all(
            selected for _, selected in selected_by_parameter.values()
        ), f"LoRA cannot select a subset of parameters in {mapping.megatron_param!r}"
        selected_adapters = []
        for sources, selected in selected_by_parameter.values():
            covered_sources.update(selected)
            split = canonical and len(sources) > 1 and ".experts." not in module
            if split:
                assert module.rsplit(".", 1)[-1] in (
                    "linear_qkv",
                    "linear_fc1",
                ), f"CanonicalLoRA does not define split adapters for {module!r}"
                adapter_sources = {_canonical_adapter_module(module, source): {source} for source in selected}
            else:
                assert selected == sources, (
                    f"LoRA on fused module {module!r} requires all HF targets; "
                    "use canonical_lora to select individual projections"
                )
                adapter_sources = {module: selected}
            selected_adapters.append(frozenset(adapter_sources))
            for adapter, parameters in adapter_sources.items():
                previous = candidates.get(adapter)
                if previous is not None:
                    parameters = parameters | previous.checkpoint_parameters
                candidates[adapter] = _TargetModule(module, frozenset(parameters))
        assert (
            len(set(selected_adapters)) == 1
        ), f"LoRA cannot select different projections across parameters in {mapping.megatron_param!r}"
    assert candidates, "LoRA targets have no Megatron modules"
    hf_mapping.validate_coverage(covered_sources, targets)
    return candidates


def validate_lora_target_adapters(model_chunks, candidates):
    missing = set()
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if any(fnmatchcase(name, mapping.megatron_module) for mapping in candidates.values()):
                if not any(param.requires_grad for param in module.parameters()):
                    missing.add(name)
    if dist.is_initialized():
        names_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(names_by_rank, missing)
        missing = set().union(*names_by_rank)
    assert not missing, f"LoRA injection skipped selected Megatron modules: {sorted(missing)}"


def normalize_lora_targets_to_hf(hf_checkpoint, target_modules, *, canonical, exclude_modules):
    # Only legacy Megatron selectors need Bridge before trainer creation.
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    hf_mapping = HfWeightMapping.from_config(bridge.hf_pretrained.config)
    model_bridge = bridge._model_bridge
    model_bridge.hf_pretrained = bridge.hf_pretrained
    selected, covered = set(), set()
    for mapping in model_bridge.mapping_registry().get_all_mappings():
        module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        for source in _checkpoint_parameters(mapping):
            hf_module = hf_mapping.model_parameter(source).removesuffix(".weight")
            if hf_mapping.parameter_shapes and not any(
                matches_hf_lora_target(name.removesuffix(".weight"), hf_module) for name in hf_mapping.parameter_shapes
            ):
                continue
            for target in target_modules:
                matches = matches_hf_lora_target(hf_module, target) or _matches_megatron_target(module, target)
                if (
                    canonical
                    and module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1")
                    and ".experts." not in module
                ):
                    matches |= _matches_megatron_target(_canonical_adapter_module(module, source), target)
                if matches:
                    covered.add(target)
                    if not any(
                        matches_hf_lora_target(hf_module, exclude) or _matches_megatron_target(module, exclude)
                        for exclude in exclude_modules
                    ):
                        selected.add(hf_module)
    assert (
        set(target_modules) <= covered
    ), f"LoRA targets have no Bridge mapping: {sorted(set(target_modules) - covered)}"
    assert selected, "No LoRA targets remain after applying --exclude-modules"
    return sorted(selected)
