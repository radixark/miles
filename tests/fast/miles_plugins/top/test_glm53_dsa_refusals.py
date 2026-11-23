"""GLM-5.3-Flash DSA under TOP refuses checkpoints, configs and options its program would not run as serving does."""

from types import SimpleNamespace

import pytest
import torch

glm53_dsa = pytest.importorskip("miles_plugins.top.glm53_dsa")


def _hf(**overrides):
    values = dict(
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        v_head_dim=256,
        kv_lora_rank=512,
        num_attention_heads=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        # The checkpoint's own (unscaled) rope entry must be admitted.
        rope_scaling={"rope_theta": 800000.0, "partial_rotary_factor": 1.0, "rope_type": "default"},
        q_lora_rank=1536,
        rms_norm_eps=1e-5,
    )
    return SimpleNamespace(**(values | overrides))


def _config(**overrides):
    values = dict(
        qk_pos_emb_head_dim=0,
        qk_head_dim=256,
        v_head_dim=256,
        kv_lora_rank=512,
        num_attention_heads=64,
        q_lora_rank=1536,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
    )
    return SimpleNamespace(**(values | overrides))


def _args(**overrides):
    return SimpleNamespace(**(dict(use_indexer_replay=False) | overrides))


def test_the_checkpoint_recipe_is_admitted():
    glm53_dsa.validate_dsa_config(_args(), _config(), _hf())


@pytest.mark.parametrize(
    ("args", "config", "hf", "refusal"),
    [
        ({}, {}, {"qk_rope_head_dim": 64}, "qk_rope_head_dim=0"),
        ({}, {}, {"index_kpool": 1}, "index_kpool=4"),
        ({}, {}, {"index_kpool_compress": False}, "index_kpool_compress"),
        ({}, {}, {"index_topk": 1024}, "index_topk=2048"),
        ({}, {}, {"rope_scaling": {"type": "yarn"}}, "rope_scaling"),
        ({}, {"layernorm_epsilon": 1e-6}, {}, "layernorm_epsilon"),
        ({}, {"layernorm_zero_centered_gamma": True}, {}, "zero_centered_gamma"),
        ({}, {"q_lora_rank": 1024}, {}, "q_lora_rank"),
        ({"use_indexer_replay": True}, {}, {}, "indexer replay"),
    ],
    ids=["rope", "kpool", "compress", "topk", "rope-scaling", "norm-eps", "zero-centered", "q-lora", "replay"],
)
def test_what_the_program_does_not_run_is_refused(args, config, hf, refusal):
    with pytest.raises(NotImplementedError, match=refusal):
        glm53_dsa.validate_dsa_config(_args(**args), _config(**config), _hf(**hf))


def _packed(cu, padded=None):
    return SimpleNamespace(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        cu_seqlens_q_padded=None if padded is None else torch.tensor(padded, dtype=torch.int32),
    )


def test_sample_bounds_cover_every_row():
    """Selection is per sample: padded bounds win, and trailing pad rows form their own sample, never a real one's."""
    assert glm53_dsa.sample_bounds(_packed([0, 3, 7]), 7, "cpu").tolist() == [0, 3, 7]
    assert glm53_dsa.sample_bounds(_packed([0, 3, 7], [0, 4, 8]), 10, "cpu").tolist() == [0, 4, 8, 10]
    with pytest.raises(ValueError, match="beyond"):
        glm53_dsa.sample_bounds(_packed([0, 3, 9]), 7, "cpu")
