from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
    _use_dsv4_top_pow2_scales,
)
from miles.backends.sglang_utils import dsv4_top_patches
from miles.backends.sglang_utils.sglang_engine import (
    _dsv4_top_runtime_patches_enabled,
)
from miles.backends.sglang_utils.dsv4_top_moe import (
    dsv4_top_moe_context,
    get_dsv4_top_moe_context,
)

_CONTRACT_PATH = Path(dsv4_top_patches.__file__).with_name("dsv4_top_sglang_reference.yaml")


def test_dsv4_top_source_contract_is_reviewable_and_diagnostics_free():
    text = _CONTRACT_PATH.read_text()
    for forbidden in (
        "_codex",
        "dumper.dump",
        "diagnostic",
        "probe",
    ):
        assert forbidden not in text.lower()

    contract = yaml.safe_load(text)
    patches = contract["patches"]
    assert len(patches) == 12
    assert len({patch["target"] for patch in patches}) == len(patches)
    assert sum(len(patch.get("edits", ())) for patch in patches) == 20

    for patch in patches:
        assert patch["target"].startswith("sglang.")
        for edit in patch.get("edits", ()):
            assert edit["match"].strip()
            assert edit["replacement"].strip()


@pytest.mark.parametrize(
    ("source_contract", "activates_source_contract"),
    (("0", False), ("1", True)),
)
def test_dsv4_top_source_contract_switch(
    monkeypatch,
    source_contract,
    activates_source_contract,
):
    calls = []
    monkeypatch.setenv("MILES_DSV4_TOP_SOURCE_CONTRACT", source_contract)
    monkeypatch.setattr(
        dsv4_top_patches,
        "_activate_reference_source_patches",
        lambda: calls.append("source"),
    )
    monkeypatch.setattr(
        dsv4_top_patches,
        "_patch_hash_topk_fp32_input",
        lambda: calls.append("hash_topk"),
    )

    dsv4_top_patches.apply_dsv4_top_sglang_patches()

    assert calls == (["source", "hash_topk"] if activates_source_contract else ["hash_topk"])


def test_dsv4_top_source_contract_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("MILES_DSV4_TOP_SOURCE_CONTRACT", "true")

    with pytest.raises(
        RuntimeError,
        match="MILES_DSV4_TOP_SOURCE_CONTRACT must be 0 or 1",
    ):
        dsv4_top_patches.apply_dsv4_top_sglang_patches()


@pytest.mark.parametrize(
    ("value", "enabled"),
    (("0", False), ("1", True)),
)
def test_dsv4_top_runtime_patch_switch(monkeypatch, value, enabled):
    monkeypatch.setenv("MILES_DSV4_TOP_RUNTIME_PATCHES", value)
    assert _dsv4_top_runtime_patches_enabled() is enabled


def test_dsv4_top_runtime_patch_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("MILES_DSV4_TOP_RUNTIME_PATCHES", "true")
    with pytest.raises(
        ValueError,
        match="MILES_DSV4_TOP_RUNTIME_PATCHES must be 0 or 1",
    ):
        _dsv4_top_runtime_patches_enabled()


@pytest.mark.parametrize(
    ("true_on_policy", "attention_variant", "expected"),
    (
        (True, "dsv4", True),
        (False, "dsv4", False),
        (True, None, False),
        (True, "mla", False),
    ),
)
def test_pow2_weight_scales_are_scoped_to_dsv4_top(
    true_on_policy,
    attention_variant,
    expected,
):
    args = SimpleNamespace(
        true_on_policy_mode=true_on_policy,
        experimental_attention_variant=attention_variant,
    )
    assert _use_dsv4_top_pow2_scales(args) is expected


def test_dsv4_top_moe_context_is_request_local_and_restored():
    hidden_states = torch.empty((96, 4096))
    topk_ids = torch.zeros((96, 6), dtype=torch.int64)
    assert get_dsv4_top_moe_context() is None

    with dsv4_top_moe_context(
        layer_id=0,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
    ) as context:
        assert context.active is True
        assert get_dsv4_top_moe_context() is context

    assert get_dsv4_top_moe_context() is None
