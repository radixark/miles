"""A slot checkpoint's storage keys are slot-agnostic, so any free slot can reload it."""

from types import SimpleNamespace

from miles.backends.megatron_utils.lora.checkpoint import _canonicalize_slot_keys


def test_saved_and_loading_slots_meet_at_the_canonical_key():
    saved = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.2.linear_in.weight")
    loading = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.0.linear_in.weight")
    _canonicalize_slot_keys({"weights": {"w": saved}}, 2)
    _canonicalize_slot_keys({"weights": {"w": loading}}, 0)
    assert saved.key == loading.key == "decoder.layers.3.linear_qkv.adapter.linear_in.weight"
