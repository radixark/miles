import re

from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.utils import matches_lora_target

_CANONICAL_PROJECTIONS = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
}


def _canonical_adapter_module(module, checkpoint_parameter):
    leaf = checkpoint_parameter.removesuffix(".weight").rsplit(".", 1)[-1]
    assert leaf in _CANONICAL_PROJECTIONS, f"CanonicalLoRA has no split adapter for {checkpoint_parameter!r}"
    return f"{module.rsplit('.', 1)[0]}.{_CANONICAL_PROJECTIONS[leaf]}"


def _checkpoint_parameters(mapping):
    return {mapping.hf_param} if isinstance(mapping.hf_param, str) else set(mapping.hf_param.values())


def _select_checkpoint_parameters(checkpoint_parameters, targets, *, hf_mapping, megatron_module, exclude_modules):
    hf_parameters = {name: hf_mapping.model_parameter(name) for name in checkpoint_parameters}
    # Bridge can expose auxiliary layers, such as MTP, absent from the HF model.
    selected = {
        checkpoint_parameter
        for checkpoint_parameter, hf_parameter in hf_parameters.items()
        if (not hf_mapping.parameter_names or hf_parameter in hf_mapping.parameter_names)
        and any(matches_lora_target(hf_parameter.removesuffix(".weight"), target) for target in targets)
    }
    return {
        checkpoint_parameter
        for checkpoint_parameter in selected
        if not any(
            matches_lora_target(hf_parameters[checkpoint_parameter].removesuffix(".weight"), target)
            or matches_lora_target(megatron_module, target)
            for target in exclude_modules
        )
    }


def _resolve_adapter_targets(megatron_module, checkpoint_parameters, selected, *, canonical):
    if (
        canonical
        and len(checkpoint_parameters) > 1
        and ".experts." not in megatron_module
        and megatron_module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1")
    ):
        return list(dict.fromkeys(_canonical_adapter_module(megatron_module, name) for name in selected))
    assert (
        selected == checkpoint_parameters
    ), f"LoRA on fused module {megatron_module!r} requires all HF targets with this adapter"
    if canonical and ".experts." in megatron_module and megatron_module.endswith(".linear_fc1"):
        # CanonicalLoRA requires split aliases even for fused expert adapters.
        return [f"{megatron_module}_gate", f"{megatron_module}_up"]
    return [megatron_module]


def resolve_megatron_lora_targets(targets, mappings, *, parameter_names, hf_mapping, canonical, exclude_modules=()):
    adapter_targets = {}
    matched_megatron_parameters = set()
    for mapping in mappings:
        megatron_module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        regex = re.compile(re.escape(mapping.megatron_param).replace(r"\*", "(.*)"))
        selections = []
        for megatron_parameter in sorted(parameter_names - matched_megatron_parameters):
            match = regex.fullmatch(megatron_parameter)
            if match is None:
                continue
            matched_megatron_parameters.add(megatron_parameter)
            checkpoint_parameters = _checkpoint_parameters(mapping.resolve(match.groups()))
            selected = _select_checkpoint_parameters(
                checkpoint_parameters,
                targets,
                hf_mapping=hf_mapping,
                megatron_module=megatron_module,
                exclude_modules=exclude_modules,
            )
            selections.append((checkpoint_parameters, selected))
        if not any(selected for _, selected in selections):
            continue
        # Template injection must select the same projections for every layer/expert it matches.
        assert all(
            selected for _, selected in selections
        ), f"LoRA cannot select a subset of parameters in {mapping.megatron_param!r}"
        adapter_selections = set()
        for checkpoint_parameters, selected in selections:
            selected_adapters = _resolve_adapter_targets(
                megatron_module, checkpoint_parameters, selected, canonical=canonical
            )
            adapter_selections.add(frozenset(selected_adapters))
            adapter_targets.update(dict.fromkeys(selected_adapters))
        assert (
            len(adapter_selections) == 1
        ), f"LoRA cannot select different projections across parameters in {mapping.megatron_param!r}"
    assert adapter_targets, "LoRA targets have no Megatron modules"
    return list(adapter_targets)


def normalize_lora_targets_to_hf(hf_checkpoint, target_modules, *, canonical, exclude_modules, explicit_targets=()):
    # Only legacy Megatron selectors need Bridge before trainer creation.
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    hf_mapping = HfWeightMapping.from_config(bridge.hf_pretrained.config)
    model_bridge = bridge._model_bridge
    model_bridge.hf_pretrained = bridge.hf_pretrained
    selected_hf_modules, matched_targets = set(), set()
    for mapping in model_bridge.mapping_registry().get_all_mappings():
        megatron_module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        for checkpoint_parameter in _checkpoint_parameters(mapping):
            hf_module = hf_mapping.model_parameter(checkpoint_parameter).removesuffix(".weight")
            if hf_mapping.parameter_names and not any(
                matches_lora_target(name.removesuffix(".weight"), hf_module) for name in hf_mapping.parameter_names
            ):
                continue
            for target in target_modules:
                matches = matches_lora_target(hf_module, target) or matches_lora_target(megatron_module, target)
                if (
                    canonical
                    and megatron_module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1")
                    and ".experts." not in megatron_module
                ):
                    matches |= matches_lora_target(
                        _canonical_adapter_module(megatron_module, checkpoint_parameter), target
                    )
                if matches:
                    matched_targets.add(target)
                    excluded = any(
                        matches_lora_target(hf_module, exclude) or matches_lora_target(megatron_module, exclude)
                        for exclude in exclude_modules
                    )
                    assert not (
                        excluded and target in explicit_targets
                    ), f"Explicit LoRA target {target!r} overlaps --exclude-modules at {hf_module!r}"
                    if not excluded:
                        selected_hf_modules.add(hf_module)
    assert (
        set(target_modules) <= matched_targets
    ), f"LoRA targets have no Bridge mapping: {sorted(set(target_modules) - matched_targets)}"
    assert selected_hf_modules, "No LoRA targets remain after applying --exclude-modules"
    return sorted(selected_hf_modules)
