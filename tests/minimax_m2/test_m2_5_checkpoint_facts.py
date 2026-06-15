"""Pin the structural facts of the released MiniMax-M2.5 HF checkpoint.

The bridge + spec rely on three load-bearing assumptions about the
checkpoint stored at /home/yangchengyi/data/models/MiniMax-M2.5:

1. **No MTP weights ship in the checkpoint.** Despite ``use_mtp: true,
   num_mtp_modules: 3`` in config.json, every key in
   ``model.safetensors.index.json`` falls under ``model.layers.0..61`` /
   ``model.embed_tokens`` / ``model.norm`` / ``lm_head``. The slime-aligned
   launcher therefore must not enable MTP training by default.

2. **MoE experts use Mixtral-style ``block_sparse_moe.experts.{i}.{w1,w2,w3}``.**
   :class:`miles_plugins.mbridge.minimax_m2.MiniMaxM2Bridge` hard-codes this
   layout in ``_MLP_MAPPING``.

3. **Per-layer Q/K-Norm and routing bias are present in the checkpoint.**
   Training uses ``--spec ... get_minimax_m2_layer_spec`` so Megatron swaps in
   ``MiniMaxM2SelfAttention``; mbridge maps ``q_norm`` / ``k_norm`` and
   ``e_score_correction_bias`` for HF round-trip.

If any of these assertions ever fails, MiniMax has changed the checkpoint
shape and the bridge / spec must be updated to match -- treat a failure
as an actionable bug, not a flake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

M2_5_INDEX = Path(
    "/home/yangchengyi/data/models/MiniMax-M2.5/model.safetensors.index.json"
)


@pytest.fixture
def weight_map():
    if not M2_5_INDEX.exists():
        pytest.skip(f"{M2_5_INDEX} not present; checkpoint-fact test skipped")
    return json.loads(M2_5_INDEX.read_text())["weight_map"]


def test_m2_5_checkpoint_has_no_mtp_weights(weight_map):
    """Pin the surprising fact: config.json claims MTP, the ckpt does not.

    If this fails, MiniMax has shipped MTP weights and we need to:
    1. Inspect the new key names (likely model.layers.{62..64}.* or mtp.*).
    2. Revisit the slime-aligned launcher/model spec before enabling MTP.
    """
    mtp_keys = sorted(k for k in weight_map if "mtp" in k.lower())
    assert mtp_keys == [], (
        f"Released M2.5 checkpoint now contains {len(mtp_keys)} MTP weight(s) "
        f"-- revisit the slime-aligned launcher/model spec. Sample: {mtp_keys[:5]}"
    )


def test_m2_5_checkpoint_uses_mixtral_block_sparse_moe(weight_map):
    """The mapping registry assumes block_sparse_moe.experts.{i}.{w1,w2,w3}."""
    assert any(
        ".block_sparse_moe.experts." in k and k.endswith(".w1.weight")
        for k in weight_map
    ), "Expected Mixtral-style experts.*.w1.weight in M2.5 checkpoint"
    assert any(
        ".block_sparse_moe.experts." in k and k.endswith(".w2.weight")
        for k in weight_map
    ), "Expected experts.*.w2.weight in M2.5 checkpoint"
    assert any(
        ".block_sparse_moe.experts." in k and k.endswith(".w3.weight")
        for k in weight_map
    ), "Expected experts.*.w3.weight in M2.5 checkpoint"


def test_m2_5_has_aux_loss_free_routing_bias(weight_map):
    """provider_bridge sets moe_router_enable_expert_bias=True; verify HF ships the buffer."""
    assert any(
        ".block_sparse_moe.e_score_correction_bias" in k for k in weight_map
    ), (
        "Expected e_score_correction_bias buffer in M2.5 (aux-loss-free routing). "
        "If absent, set provider.moe_router_enable_expert_bias=False."
    )


def test_m2_5_has_per_layer_qk_norm(weight_map):
    """MiniMaxM2SelfAttention stores full-dimension q_norm / k_norm weights."""
    assert any(".self_attn.q_norm.weight" in k for k in weight_map), (
        "Expected per-layer q_norm.weight in M2.5"
    )
    assert any(".self_attn.k_norm.weight" in k for k in weight_map), (
        "Expected per-layer k_norm.weight in M2.5"
    )


def test_m2_5_top_level_prefixes_are_only_model_and_lm_head(weight_map):
    """Bridge assumes only model.* and lm_head.* exist; nothing else."""
    expected = {"lm_head", "model"}
    actual = {k.split(".", 1)[0] for k in weight_map}
    unexpected = actual - expected
    assert not unexpected, f"Unexpected top-level prefixes: {unexpected}"


def test_m2_5_has_exactly_62_decoder_layers(weight_map):
    """Verify the 62-layer trunk count we wire into provider_bridge."""
    import re

    pat = re.compile(r"^model\.layers\.(\d+)\.")
    ids = {int(m.group(1)) for m in (pat.match(k) for k in weight_map) if m}
    assert ids == set(range(62)), (
        f"Expected exactly layers 0..61. Got {sorted(ids)[:5]}...{sorted(ids)[-5:]}, "
        f"total={len(ids)}"
    )
