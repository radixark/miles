"""Check a merged --save-hf export against base + (alpha/r) * B @ A computed from its adapter/ at the HF level.

usage: python verify_merged_export.py <export_dir> <base_hf_dir>
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def _tensors(directory: Path):
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(directory / "model.safetensors"), framework="pt") as handle:
            index = dict.fromkeys(handle.keys(), "model.safetensors")
    handles = {}
    for name, shard in index.items():
        if shard not in handles:
            handles[shard] = safe_open(str(directory / shard), framework="pt")
        yield name, handles[shard]


_EXPERT_LEAF_ALIASES = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def _interleave(delta):
    """Inkling's HF w13 layout alternates gate and up rows."""
    gate, up = delta.chunk(2, dim=0)
    return torch.stack((gate, up), dim=1).reshape(delta.shape)


def _find_key(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _pair(adapter, module):
    return adapter.get(f"{module}.lora_A.weight"), adapter.get(f"{module}.lora_B.weight")


def _adapter_delta(name, adapter, scale, reference_shape):
    module = name.removesuffix(".weight")
    a, b = _pair(adapter, module)
    if a is not None:
        return scale * (b.float() @ a.float())
    if module.endswith(".unembed"):
        heads = [key.removesuffix(".lora_A.weight") for key in adapter if key.endswith("lm_head.lora_A.weight")]
        if heads:
            a, b = _pair(adapter, heads[0])
            return scale * (b.float() @ a.float())
    if module.endswith(".mlp.w13_dn"):
        a, b = _pair(adapter, module.removesuffix("w13_dn") + "gate_up_proj")
        return None if a is None else scale * _interleave(b.float() @ a.float())
    if module.endswith(".mlp.w2_md"):
        a, b = _pair(adapter, module.removesuffix("w2_md") + "down_proj")
        return None if a is None else scale * (b.float() @ a.float())
    if module.endswith(".shared_experts.shared_w13_weight") or module.endswith(".shared_experts.shared_w2_weight"):
        prefix = module.rsplit(".", 1)[0]
        a1, b1 = _pair(adapter, f"{prefix}.w1")
        if a1 is None:
            return None
        a3, b3 = _pair(adapter, f"{prefix}.w3")
        a2, b2 = _pair(adapter, f"{prefix}.w2")
        return scale * _shared_expert_deltas(reference_shape[0], module.endswith("w13_weight"), a1, b1, a3, b3, a2, b2)
    match = re.fullmatch(r"(.*\.experts)\.(\d+)\.(\w+)", module)
    if match is None:
        return None
    for leaf in (match[3], _EXPERT_LEAF_ALIASES.get(match[3])):
        a, b = _pair(adapter, f"{match[1]}.{leaf}")
        if a is not None:
            expert = int(match[2])
            return scale * (b[0 if b.shape[0] == 1 else expert].float() @ a[0 if a.shape[0] == 1 else expert].float())
    return None


def _shared_expert_deltas(num_shared, fc1, a1, b1, a3, b3, a2, b2):
    """Sub-expert slices of the concatenated shared-expert adapter, stacked like Inkling's HF tensors."""
    inter = b1.shape[0] // num_shared
    deltas = []
    for index in range(num_shared):
        rows = slice(index * inter, (index + 1) * inter)
        if fc1:
            deltas.append(_interleave(torch.cat([b1[rows].float() @ a1.float(), b3[rows].float() @ a3.float()])))
        else:
            deltas.append(b2.float() @ a2[:, rows].float())
    return torch.stack(deltas)


def _base_reference(name, base, merged):
    """The base tensor for ``name``; per-expert exports slice the base's packed expert tensors."""
    if name in base:
        return base[name].get_tensor(name).float()
    match = re.fullmatch(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
    assert match is not None, f"{name} is neither in the base nor a per-expert slice of it"
    expert = int(match[2])
    if match[3] == "down_proj":
        packed = f"{match[1]}.w2_weight"
        return base[packed].get_slice(packed)[expert].float()
    packed = f"{match[1]}.w13_weight"
    w13 = base[packed].get_slice(packed)[expert].float()
    half = w13.shape[0] // 2
    candidates = {
        "halves": w13[:half] if match[3] == "gate_proj" else w13[half:],
        "interleaved": w13[0::2] if match[3] == "gate_proj" else w13[1::2],
    }
    return min(candidates.values(), key=lambda candidate: (candidate - merged).abs().max().item())


def main(export_dir: str, base_dir: str) -> int:
    export_dir, base_dir = Path(export_dir), Path(base_dir)
    config = json.loads((export_dir / "adapter" / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    mup = _find_key(json.loads((export_dir / "config.json").read_text()), "logits_mup_width_multiplier")
    with safe_open(str(export_dir / "adapter" / "adapter_model.safetensors"), framework="pt") as handle:
        adapter = {name.removeprefix("base_model.model."): handle.get_tensor(name) for name in handle.keys()}
    base = dict(_tensors(base_dir))

    adapted = plain = 0
    changed = expected_changed = total = 0
    reshaped = []
    worst_adapted = worst_plain = 0.0
    smallest_delta = float("inf")
    for name, handle in _tensors(export_dir):
        merged = handle.get_tensor(name).float()
        reference = _base_reference(name, base, merged)
        delta = _adapter_delta(name, adapter, scale, reference.shape)
        if delta is None:
            plain += 1
            if merged.shape != reference.shape:
                reshaped.append(f"{name} {tuple(merged.shape)} vs base {tuple(reference.shape)}")
                continue
            worst_plain = max(worst_plain, (merged - reference).abs().max().item())
            continue
        adapted += 1
        if name.endswith(("lm_head.weight", "unembed.weight")):
            delta = delta / (mup or 1.0)
            delta = torch.nn.functional.pad(delta, (0, 0, 0, reference.shape[0] - delta.shape[0]))
        expected = (reference + delta).to(torch.bfloat16).float()
        changed += (merged != reference).sum().item()
        expected_changed += (expected != reference).sum().item()
        total += merged.numel()
        error = (merged - expected).abs().max().item() / max(expected.abs().max().item(), 1e-6)
        worst_adapted = max(worst_adapted, error)
        smallest_delta = min(smallest_delta, delta.abs().max().item())
    print(
        f"adapted={adapted} plain={plain} max_rel_err(adapted)={worst_adapted:.3e} "
        f"max_abs_err(plain)={worst_plain:.3e} min_max|delta|={smallest_delta:.3e} scale={scale}"
    )
    print(f"adapted elements changed vs base: {changed}/{total} (expected {expected_changed})")
    print(f"plain tensors whose exported shape differs from the base ({len(reshaped)}): {reshaped[:6]}")
    ok = adapted > 0 and worst_adapted < 2e-2 and worst_plain == 0.0 and changed > 0
    print("MERGED EXPORT OK" if ok else "MERGED EXPORT MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
