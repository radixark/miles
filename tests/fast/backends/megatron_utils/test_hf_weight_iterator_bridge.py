from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.update_weight.common import get_atomic_update_groups
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import (
    HfWeightIteratorBridge,
    _stream_atomic_units,
)

Q_DOWN = "decoder.layers.0.self_attention.linear_q_down_proj.weight"
KV_DOWN = "decoder.layers.0.self_attention.linear_kv_down_proj.weight"


def _units(megatron_names):
    """Run names through the real export pipeline: normalise, then group."""
    stub = SimpleNamespace(
        args=Namespace(vocab_size=None, q_lora_rank=1536),
        quantization_config=None,
    )
    items = [(f"hf.{name}", torch.zeros(1), name) for name in megatron_names]
    processed = HfWeightIteratorBridge._postprocess_and_quantize(stub, items, "base")
    groups = get_atomic_update_groups(Namespace(q_lora_rank=1536), "kimi_k25")
    return [[hf for hf, _ in unit] for unit in _stream_atomic_units(processed, groups)]


def test_mla_down_projections_group_atomically():
    assert len(_units([Q_DOWN, KV_DOWN])) == 1


def test_lora_wrapped_names_still_group_atomically():
    """megatron-bridge PEFT renames a wrapped module's weight to `<module>.to_wrap.weight`.

    The atomic group suffixes are unwrapped names, so a wrapper segment left in the
    megatron name makes `endswith` miss and splits q_a_proj/kv_a_proj into separate
    units. SGLang fuses those two into `fused_qkv_a_proj_with_mqa`, so shipping them
    in different chunks leaves half the fused tensor stale. Nothing asserts on this
    path -- neither param ever enters `pending` -- so it corrupts weights silently.
    """
    wrapped = [name.replace(".weight", ".to_wrap.weight") for name in (Q_DOWN, KV_DOWN)]
    assert len(_units(wrapped)) == 1


def _units_from(entries):
    """entries: list of (hf_name, megatron_name) in emission order."""
    items = [(hf, torch.zeros(1), megatron) for hf, megatron in entries]
    groups = get_atomic_update_groups(Namespace(q_lora_rank=1536), "kimi_k25")
    return [[hf for hf, _ in unit] for unit in _stream_atomic_units(iter(items), groups)]


def test_quantized_param_keeps_its_tensors_in_one_unit():
    """A quantized param leaves the exporter as several tensors sharing one megatron name.

    Per-tensor slot assignment let the later ones overwrite the earlier, so the group
    both dropped tensors and tripped the end-of-stream assert once `.to_wrap.` stripping
    made the suffixes match at all.
    """
    entries = [
        (f"{proj}.{suffix}", name)
        for name, proj in ((Q_DOWN, "q_a"), (KV_DOWN, "kv_a"))
        for suffix in ("weight_packed", "weight_scale", "weight_shape")
    ]
    units = _units_from(entries)
    assert len(units) == 1
    assert len(units[0]) == 6


def test_params_outside_any_group_pass_through():
    units = _units_from([("fc2", "decoder.layers.0.mlp.linear_fc2.weight")])
    assert units == [["fc2"]]
