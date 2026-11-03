"""PEFT, sampler and external (official Tinker) exports, written from whole HF-keyed tensors on one process."""

from collections.abc import Sequence
from pathlib import Path

import pyrage
import torch

from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    SAMPLER_EXPERT_NAMES_BY_HF,
    fuse_lora_b,
    fused_leaves,
    map_hf_keys_to_peft_keys,
    map_hf_keys_to_tinker_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets
from miles.utils.lora.utils import get_adapter_target_modules, save_adapter_to_disk


def _expand_shared_lora_a(
    weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> dict:
    """Each module gets its own A, except that an HF-fused module whose components share A goes back to its name."""
    expanded = dict(weights)
    for group in parametrization.shared_lora_a_groups:
        shared_a = expanded.pop(f"{group[0]}.lora_A")
        parent = group[0].rpartition(".")[0]
        unfused = list(group)
        for fused_leaf, leaves in fused_leaves(parent, targets).items():
            members = [f"{parent}.{leaf}" for leaf in leaves]
            if all(m in unfused for m in members):
                fused = f"{parent}.{fused_leaf}"
                expanded[f"{fused}.lora_A"] = shared_a.clone()
                expanded[f"{fused}.lora_B"] = fuse_lora_b(
                    fused, [expanded.pop(f"{m}.lora_B") for m in members], targets
                )
                unfused = [m for m in unfused if m not in members]
        for member in unfused:
            # safetensors refuses tensors that share storage
            expanded[f"{member}.lora_A"] = shared_a.clone()
    return expanded


def _packed_experts(expanded: dict[str, torch.Tensor]) -> set[str]:
    """The ``.experts`` modules whose gate / up went back to one packed ``gate_up_proj``; they stay stacked 3D."""
    return {
        key.removesuffix(".gate_up_proj.lora_A") for key in expanded if key.endswith(".experts.gate_up_proj.lora_A")
    }


def _stays_stacked(key: str, packed: set[str], targets: HfLoraTargets) -> bool:
    # a packed family's HF down_proj is one 3D tensor whether or not gate_up_proj is adapted
    is_packed_down = targets.packed_expert_gate_up_layout is not None and ".experts.down_proj." in f"{key}."
    return _experts_module(key) in packed or is_packed_down


def _experts_module(key: str) -> str:
    return key.rsplit(".", 2)[0]


def _adapter_config(parametrization: AdapterParametrization, tensor_names) -> dict:
    assert parametrization.lora_alpha is not None, "lora_alpha must be resolved before an export is written"
    return {
        "peft_type": "LORA",
        "r": parametrization.lora_rank,
        "lora_alpha": parametrization.lora_alpha,
        "target_modules": get_adapter_target_modules(tensor_names),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def write_peft_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> None:
    """bf16 PEFT with per-expert 2D tensors (experts re-packed into ``gate_up_proj`` stay stacked); a shared-outer
    side stays ``[1, ...]`` under the unindexed name."""
    tensors = {}
    expanded = _expand_shared_lora_a(weights, parametrization, targets)
    packed = _packed_experts(expanded)
    for key, value in expanded.items():
        value = value.to(torch.bfloat16).contiguous()
        if ".experts." in key and value.dim() == 3 and value.shape[0] > 1 and not _stays_stacked(key, packed, targets):
            for expert_index in range(value.shape[0]):
                tensors[map_hf_keys_to_peft_keys(key, expert_index)] = value[expert_index].clone()
        else:
            tensors[map_hf_keys_to_peft_keys(key)] = value
    save_adapter_to_disk(directory, _adapter_config(parametrization, tensors), tensors)


def write_sampler_adapter(
    directory: Path, weights: dict[str, torch.Tensor], parametrization: AdapterParametrization, targets: HfLoraTargets
) -> None:
    """bf16 adapter sglang loads from disk; experts stay stacked 3D. Carries its parametrization, so it reads back."""
    tensors = {}
    expanded = _expand_shared_lora_a(weights, parametrization, targets)
    packed = _packed_experts(expanded)
    for key, value in expanded.items():
        name = map_hf_keys_to_peft_keys(key)
        # packed experts keep their HF names, as main's export serves them
        if not _stays_stacked(key, packed, targets):
            for hf_fragment, sglang_fragment in SAMPLER_EXPERT_NAMES_BY_HF.items():
                name = name.replace(hf_fragment, sglang_fragment)
        tensors[name] = value.to(torch.bfloat16).contiguous()
    # target_modules name the HF modules the adapter applies to, not sglang's tensor names for them
    hf_names = [map_hf_keys_to_peft_keys(key) for key in expanded]
    save_adapter_to_disk(directory, _adapter_config(parametrization, hf_names), tensors)
    write_parametrization(Path(directory), parametrization)


def write_external_adapter(
    directory: Path,
    weights: dict[str, torch.Tensor],
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    age_recipients: Sequence[str] = (),
) -> None:
    """The official Tinker export: F32 under Tinker names, experts stacked 3D as the parametrization has them
    (``[1, ...]`` on a shared-outer side), and every module its own copy of a shared A (no re-fusing: Tinker trains
    the components apart). With ``age_recipients`` both files are age-encrypted to them and named ``<file>.age``."""
    tensors = {}
    shared_copies = {member: group[0] for group in parametrization.shared_lora_a_groups for member in group[1:]}
    for module in parametrization.target_modules:
        for factor in ("lora_A", "lora_B"):
            source = f"{shared_copies.get(module, module)}.lora_A" if factor == "lora_A" else f"{module}.lora_B"
            tensors[map_hf_keys_to_tinker_keys(f"{module}.{factor}", targets)] = weights[source].to(
                torch.float32, copy=True
            )
    hf_names = [map_hf_keys_to_peft_keys(f"{module}.lora_A") for module in parametrization.target_modules]
    # as the official export writes it: the training run, not the file, names the base
    config = _adapter_config(parametrization, hf_names) | {"base_model_name_or_path": None}
    save_adapter_to_disk(directory, config, tensors)
    if age_recipients:
        _age_encrypt_in_place(Path(directory), age_recipients)


def parse_age_recipient(recipient: str) -> pyrage.x25519.Recipient | pyrage.ssh.Recipient:
    """The SDK documents both native age (age1...) and SSH (ssh-...) public keys."""
    if recipient.startswith("ssh-"):
        return pyrage.ssh.Recipient.from_str(recipient)
    return pyrage.x25519.Recipient.from_str(recipient)


def _age_encrypt_in_place(directory: Path, recipients: Sequence[str]) -> None:
    parsed = [parse_age_recipient(r) for r in recipients]
    # not the whole directory: it also holds the gateway's tenant.json, which must stay readable to resolve the owner
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = directory / name
        path.with_name(path.name + ".age").write_bytes(pyrage.encrypt(path.read_bytes(), parsed))
        path.unlink()
