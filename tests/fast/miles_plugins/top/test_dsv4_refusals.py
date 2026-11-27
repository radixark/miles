"""DeepSeek-V4 TOP refuses trainer configurations its router, experts and checkpoint install do not cover."""

import json
from types import SimpleNamespace

import pytest
import torch

dsv4_spec = pytest.importorskip("miles_plugins.top.dsv4_spec")
dsv4_checkpoint = pytest.importorskip("miles_plugins.top.dsv4_checkpoint")
fp8_loader = pytest.importorskip("miles_plugins.top.fp8_loader")

_HF = dict(
    hidden_size=4096,
    num_attention_heads=64,
    n_routed_experts=256,
    moe_intermediate_size=2048,
    num_experts_per_tok=6,
    routed_scaling_factor=1.5,
    num_hash_layers=3,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    n_shared_experts=1,
    tie_word_embeddings=False,
)


def _config(**overrides):
    values = dict(
        hidden_size=4096,
        num_attention_heads=64,
        num_moe_experts=256,
        moe_ffn_hidden_size=2048,
        moe_router_topk=6,
        moe_router_topk_scaling_factor=1.5,
        moe_n_hash_layers=3,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        moe_shared_expert_intermediate_size=2048,
        moe_permute_fusion=False,
        moe_apply_probs_on_input=False,
        moe_shared_expert_overlap=False,
        moe_shared_expert_gate=False,
        moe_expert_capacity_factor=None,
        moe_input_jitter_eps=None,
        moe_latent_size=None,
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        moe_router_force_load_balancing=False,
        moe_router_dtype=None,
    )
    return SimpleNamespace(**(values | overrides))


def test_the_gate_moe_configuration_is_admitted():
    dsv4_spec.validate_dsv4_moe_config(_config())


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"moe_aux_loss_coeff": 1e-3}, "aux loss"),
        ({"moe_aux_loss_coeff": [0.0, 1e-3]}, "aux loss"),
        ({"moe_z_loss_coeff": 1e-3}, "z-loss"),
        ({"moe_input_jitter_eps": 1e-2}, "moe_input_jitter_eps"),
        ({"moe_router_force_load_balancing": True}, "force-load-balancing"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor"),
        ({"moe_router_dtype": "fp32"}, "router-dtype"),
        ({"moe_shared_expert_overlap": True}, "moe_shared_expert_overlap"),
        ({"moe_permute_fusion": True}, "moe_permute_fusion"),
    ],
    ids=[
        "aux-loss",
        "aux-loss-list",
        "z-loss",
        "jitter",
        "force-balance",
        "capacity",
        "router-dtype",
        "shared-overlap",
        "permute-fusion",
    ],
)
def test_moe_options_the_top_router_drops_are_refused(overrides, refusal):
    with pytest.raises(NotImplementedError, match=rf"\[top\].*{refusal}"):
        dsv4_spec.validate_dsv4_moe_config(_config(**overrides))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(dsv4_spec, "checkpoint_is_block_fp8", lambda _: True)

    def write(**overrides):
        (tmp_path / "config.json").write_text(json.dumps(_HF | overrides))
        return SimpleNamespace(
            hf_checkpoint=str(tmp_path), backend=SimpleNamespace(untie_embeddings_and_output_weights=True)
        )

    return write


def test_the_gate_checkpoint_is_admitted(checkpoint):
    dsv4_spec.validate_dsv4_checkpoint(checkpoint(), _config())


@pytest.mark.parametrize(
    "hf_overrides",
    [
        {"num_experts_per_tok": 8},
        {"routed_scaling_factor": 2.5},
        {"hc_sinkhorn_iters": 10},
        {"num_hash_layers": 2},
        {"hc_mult": 2},
        {"n_shared_experts": 2},
    ],
    ids=["topk", "routed-scaling", "sinkhorn-iterations", "hash-layers", "residual-streams", "shared-width"],
)
def test_trainer_constants_must_match_the_checkpoint(checkpoint, hf_overrides):
    with pytest.raises(ValueError, match=r"\[top\].*(disagree|differs)"):
        dsv4_spec.validate_dsv4_checkpoint(checkpoint(**hf_overrides), _config())


def test_another_mhc_eps_is_refused(checkpoint):
    with pytest.raises(NotImplementedError, match=r"\[top\].*hc_eps"):
        dsv4_spec.validate_dsv4_checkpoint(checkpoint(hc_eps=1e-5), _config())


@pytest.mark.parametrize("tied", ["checkpoint", "trainer"])
def test_tied_embeddings_are_refused(checkpoint, tied):
    args = checkpoint(tie_word_embeddings=tied == "checkpoint")
    args.backend.untie_embeddings_and_output_weights = tied != "trainer"
    with pytest.raises(NotImplementedError, match=r"\[top\].*untied"):
        dsv4_spec.validate_dsv4_checkpoint(args, _config())


def test_an_unquantized_checkpoint_is_refused_before_install(checkpoint, monkeypatch):
    """A BF16 DS4 checkpoint would install nothing and fail only at the first forward."""
    args = checkpoint()
    monkeypatch.setattr(dsv4_spec, "checkpoint_is_block_fp8", lambda _: False)
    with pytest.raises(NotImplementedError, match=r"\[top\].*block-FP8"):
        dsv4_spec.validate_dsv4_checkpoint(args, _config())


def test_checkpoint_tensors_must_fit_their_parameters():
    ck = dsv4_checkpoint.DSV4Checkpoint.__new__(dsv4_checkpoint.DSV4Checkpoint)
    param = torch.nn.Parameter(torch.zeros(4, 8))
    ck._set(param, torch.ones(4, 8))
    assert torch.equal(param.data, torch.ones(4, 8))
    with pytest.raises(ValueError, match=r"\[top\].*does not fit"):
        ck._set(param, torch.ones(8, 4))


def test_save_hf_is_refused_for_the_ds4_install():
    """DS4 installs through the FP8 masters loader, whose export refusal covers it (mixed-namespace checkpoint)."""
    assert "deepseek_v4" in fp8_loader._FAMILIES
    args = SimpleNamespace(
        backend=SimpleNamespace(overlap_param_gather=False),
        lora_rank=0,
        lora_adapter_path=None,
        save_hf="/out/{rollout_id}",
        eval_uses_snapshots=False,
        eval_hf_dir=None,
    )
    with pytest.raises(NotImplementedError, match=r"\[top\].*--save-hf"):
        fp8_loader.install(args, model=None)
