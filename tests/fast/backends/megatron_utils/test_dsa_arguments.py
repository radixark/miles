from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

from miles_plugins.models.glm5.arguments import (
    MEGATRON_DSA_SPEC,
    MILES_DSA_SPEC,
    add_dsa_arguments,
    normalize_dsa_args,
)


def _args(**overrides):
    values = dict(
        dsa_impl="megatron",
        spec=list(MILES_DSA_SPEC),
        megatron_to_hf_mode="raw",
        context_parallel_size=1,
        allgather_cp=False,
        dsa_kernel_backend="cudnn",
        dsa_indexer_loss_coeff=None,
        dsa_indexer_weights_proj_output_dtype="bf16",
        freeze_indexer=False,
    )
    return Namespace(**(values | overrides))


def _hf_config(**overrides):
    values = dict(model_type="deepseek_v32", index_n_heads=64, index_head_dim=128, index_topk=2048)
    return SimpleNamespace(**(values | overrides))


def test_default_preserves_the_existing_miles_path():
    args = add_dsa_arguments(ArgumentParser()).parse_args([])
    assert args.dsa_impl == "miles"
    assert args.freeze_indexer is False
    before = vars(args).copy()
    normalize_dsa_args(args, None)
    assert vars(args) == before


@pytest.mark.parametrize(
    ("hf_overrides", "interleaved", "frequency", "offset"),
    [
        ({"model_type": "deepseek_v32"}, False, 1, 0),
        ({"model_type": "glm_moe_dsa", "indexer_rope_interleave": True}, True, 1, 0),
        (
            {
                "model_type": "glm_moe_dsa",
                "indexer_rope_interleave": True,
                "index_topk_freq": 4,
                "index_skip_topk_offset": 2,
            },
            True,
            4,
            2,
        ),
    ],
    ids=["deepseek-v32", "glm5", "glm52-shared-indices"],
)
def test_native_dsa_preserves_checkpoint_indexer_conventions(hf_overrides, interleaved, frequency, offset):
    args = _args()
    hf_config = _hf_config(**hf_overrides)
    normalize_dsa_args(args, hf_config)

    assert args.spec == list(MEGATRON_DSA_SPEC)
    assert args.experimental_attention_variant == "dsa"
    assert args.enable_experimental is True
    assert (args.dsa_indexer_n_heads, args.dsa_indexer_head_dim, args.dsa_indexer_topk) == (64, 128, 2048)
    assert args.dsa_indexer_rope_interleaved is interleaved
    assert args.indexer_rope_interleave is interleaved
    assert (args.dsa_indexer_topk_freq, args.dsa_indexer_skip_topk_offset) == (frequency, offset)
    assert args.dsa_indexer_rotate_activation is False
    assert args.dsa_indexer_k_norm_epsilon == 1e-6
    assert args.dsa_indexer_k_norm_fp32 is True
    assert args.dsa_kernel_backend == "cudnn"
    assert args.dsa_indexer_weights_proj_output_dtype == "bf16"
    assert args.dsa_indexer_loss_coeff == 0.0

    # Checkpoint conversion and training can both normalize an already native spec.
    before = vars(args).copy()
    normalize_dsa_args(args, hf_config)
    assert vars(args) == before


@pytest.mark.parametrize("loss_coeff", [None, 0.0])
def test_frozen_indexer_has_no_auxiliary_objective(loss_coeff):
    args = _args(freeze_indexer=True, dsa_indexer_loss_coeff=loss_coeff)
    normalize_dsa_args(args, _hf_config())
    assert args.freeze_indexer
    assert args.dsa_indexer_loss_coeff == 0.0


def test_explicit_indexer_training_objective_is_preserved():
    args = _args(dsa_indexer_loss_coeff=0.001)
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_loss_coeff == 0.001


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"freeze_indexer": True, "dsa_indexer_loss_coeff": 0.001}, "--freeze-indexer requires"),
        ({"use_indexer_replay": True}, "does not support indexer replay"),
        ({"use_rollout_indexer_replay": True}, "does not support indexer replay"),
        ({"megatron_to_hf_mode": "bridge"}, "requires --megatron-to-hf-mode raw"),
        ({"spec": None}, "requires the shared DeepSeek-V3.2/GLM DSA spec"),
        ({"spec": ["miles_plugins.models.deepseek_v4", "get_dsv4_spec"]}, "requires the shared"),
        ({"context_parallel_size": 2, "allgather_cp": True}, "uses zigzag CP token partitioning"),
    ],
    ids=["frozen-loss", "indexer-replay", "rollout-indexer-replay", "bridge", "no-spec", "v4-spec", "contiguous-cp"],
)
def test_incompatible_native_configuration_fails_early(overrides, message):
    with pytest.raises(ValueError, match=message):
        normalize_dsa_args(_args(**overrides), _hf_config())


def test_unsupported_checkpoint_cannot_select_native_dsa():
    with pytest.raises(ValueError, match="does not support model_type='deepseek_v4'"):
        normalize_dsa_args(_args(), _hf_config(model_type="deepseek_v4"))


def test_native_cp_uses_allgather_communication_with_zigzag_partitioning():
    args = _args(context_parallel_size=4, cp_comm_type=["p2p"])
    normalize_dsa_args(args, _hf_config())
    assert args.cp_comm_type == ["allgather"]
    assert args.allgather_cp is False


@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
def test_native_dsa_keeps_the_requested_topk_backend(backend):
    args = _args(miles_dsa_topk_backend=backend)
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_topk_backend == backend


def test_native_dsa_defaults_to_the_existing_torch_topk_backend():
    args = _args()
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_topk_backend == "torch"
