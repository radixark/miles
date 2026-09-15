import json
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.torchtitan._common import CaseConfig, build_train_args

from miles.backends.torchtitan_utils import parallel as tp
from miles.backends.torchtitan_utils.arguments import validate_torchtitan_args

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def _args(**overrides) -> Namespace:
    base = dict(
        titan_model_name="qwen3",
        titan_seq_len=8192,
        titan_pipeline_parallel_degree=1,
        titan_context_parallel_degree=1,
        titan_expert_parallel_degree=1,
        rollout_max_context_len=8192,
        rollout_max_response_len=4096,
        ref_update_interval=None,
        save_debug_train_data=None,
    )
    return Namespace(**{**base, **overrides})


def test_the_backend_needs_torch_213(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.12.0")
    with pytest.raises(ValueError, match="torch>=2.13"):
        validate_torchtitan_args(_args())
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    validate_torchtitan_args(_args())


def test_the_sequence_must_cover_the_rotary_tables(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="titan-seq-len"):
        validate_torchtitan_args(_args(titan_seq_len=8192, rollout_max_context_len=16384))
    with pytest.raises(ValueError, match="no room for a prompt"):
        validate_torchtitan_args(
            _args(titan_seq_len=8192, rollout_max_context_len=None, rollout_max_response_len=8192)
        )
    validate_torchtitan_args(_args(titan_seq_len=16384, rollout_max_context_len=None, rollout_max_response_len=8192))


def test_context_parallelism_is_rejected_for_qwen3_5(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="context parallelism"):
        validate_torchtitan_args(_args(titan_model_name="qwen3_5", titan_context_parallel_degree=2))


def test_unsupported_flags_are_rejected_rather_than_ignored(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def test_a_tied_checkpoint_is_refused_under_pipeline_parallelism(tmp_path):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "tie_word_embeddings": True}))
    args = Namespace(optimizer="adam", titan_pipeline_parallel_degree=2)
    with pytest.raises(ValueError, match="pipeline"):
        build_trainer_config(args, hf_assets_path=str(tmp_path), lr_total_steps=1, dump_subdir="x")


class _Mesh:
    def __init__(self, size: int):
        self._group = f"pg(size={size})"

    def get_group(self):
        return self._group


class _ParallelDims:
    def __init__(self, meshes: dict[str, int]):
        self._meshes = {name: _Mesh(size) for name, size in meshes.items()}

    def get_optional_mesh(self, name):
        return self._meshes.get(name)

    def get_mesh(self, name):
        return self._meshes[name]


@pytest.fixture
def dist_stub():
    sizes: dict = {}

    def get_world_size(group=None):
        return sizes.get(group, sizes.get("__world__", 1))

    with (
        patch.object(tp.dist, "get_rank", lambda group=None: 0),
        patch.object(tp.dist, "get_world_size", get_world_size),
        patch.object(
            tp.dist,
            "new_group",
            lambda ranks, backend=None: "self_group" if backend is None else f"gloo_sub{tuple(ranks)}",
        ),
        patch.object(tp.dist, "get_process_group_ranks", lambda group: list(range(sizes.get(group, 1)))),
        patch.object(
            tp.dist, "all_gather_object", lambda out, obj, group=None: out.__setitem__(slice(None), [obj] * len(out))
        ),
        patch.object(tp, "get_gloo_group", lambda: "gloo"),
    ):
        yield sizes


def _state(dist_stub, meshes: dict[str, int], world: int | None = None, **kwargs):
    for size in meshes.values():
        dist_stub[f"pg(size={size})"] = size
    world = world if world is not None else meshes.get("loss", 1)
    dist_stub["__world__"] = world
    dist_stub["gloo"] = world
    dist_stub["self_group"] = 1
    dp = meshes.get("batch", 1)
    dist_stub[f"gloo_sub{tuple(range(dp))}"] = dp
    return tp.create_titan_parallel_state(_ParallelDims(meshes), **kwargs)


def test_context_parallelism_stays_inside_the_trainer_and_absent_axes_are_trivial(dist_stub):
    state = _state(dist_stub, {"batch": 4, "loss": 8, "cp": 2})
    assert (state.intra_dp.size, state.intra_dp_cp.size, state.cp.size) == (4, 4, 1)
    for axis in (state.tp, state.pp, state.ep, state.etp, state.indep_dp):
        assert (axis.size, axis.rank) == (1, 0)


def test_a_dp_cp_group_narrower_than_the_world_gets_its_own_gloo_subgroup(dist_stub):
    state = _state(dist_stub, {"batch": 2, "loss": 2, "tp": 4}, world=8)
    assert state.intra_dp_cp.gloo_group == "gloo_sub(0, 1)"


def test_a_degree_one_dp_cp_still_gets_a_singleton_gloo_group(dist_stub):
    dist_stub["__world__"] = 2
    dist_stub["gloo"] = 2
    dist_stub["self_group"] = 1
    dist_stub["gloo_sub(0,)"] = 1
    state = tp.create_titan_parallel_state(_ParallelDims({"tp": 2}))
    assert state.intra_dp_cp.size == 1
    assert state.intra_dp_cp.gloo_group == "gloo_sub(0,)"


def test_pp_last_stage_comes_from_the_trainer_not_the_mesh(dist_stub):
    state = _state(dist_stub, {"batch": 2, "loss": 2, "pp": 2}, is_pp_last_stage=False)
    assert state.is_pp_last_stage is False


def _case(**overrides) -> CaseConfig:
    base = dict(
        model_repo="Qwen/Qwen3-0.6B",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        num_gpus=4,
        seq_len=4096,
        max_response_len=2048,
    )
    return CaseConfig(**{**base, **overrides})


def test_the_parallelism_degrees_all_reach_the_command_line():
    args = build_train_args(_case(num_gpus=8, tp_size=2, pp_size=2, cp_size=2, ep_size=2), wandb_file=__file__)
    for flag in ("tensor", "pipeline", "context", "expert"):
        assert f"--titan-{flag}-parallel-degree 2 " in args
    with pytest.raises(ValueError, match="divisible"):
        _case(num_gpus=4, tp_size=2, pp_size=2, cp_size=2)


def test_disaggregated_cases_size_the_engine_off_the_rollout_pool():
    with pytest.raises(ValueError, match="cannot colocate"):
        _case(fully_async=True, colocate=True)
    args = build_train_args(_case(colocate=False, rollout_num_gpus=2, fully_async=True), wandb_file=__file__)
    assert "--rollout-num-gpus 2 " in args
    assert "--rollout-num-gpus-per-engine 2 " in args
    assert "--colocate " not in args
    assert "--fully-async --pause-generation-mode in_place " in args
    assert "--rollout-num-gpus-per-engine 4 " in build_train_args(_case(), wandb_file=__file__)


def test_the_transfer_mode_reaches_the_command_line_with_its_directories():
    args = build_train_args(_case(colocate=False, rollout_num_gpus=2, transfer_mode="disk-delta"), wandb_file=__file__)
    assert "--update-weight-transfer-mode disk-delta " in args
    assert "--update-weight-disk-dir " in args
    assert "--update-weight-local-checkpoint-dir " in args
    assert "--update-weight-transfer-mode" not in build_train_args(_case(), wandb_file=__file__)


_GLM47_FLASH_HF_CONFIG = dict(
    hidden_size=2048,
    num_hidden_layers=47,
    num_attention_heads=20,
    q_lora_rank=768,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    intermediate_size=10240,
    moe_intermediate_size=1536,
    n_routed_experts=64,
    n_shared_experts=1,
    num_experts_per_tok=4,
    first_k_dense_replace=1,
    routed_scaling_factor=1.8,
    norm_topk_prob=True,
    rope_theta=1000000,
    vocab_size=154880,
)

_GLM47_FLASH_CHECKPOINT_KEYS = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.self_attn.q_a_proj.weight",
    "model.layers.{}.self_attn.q_a_layernorm.weight",
    "model.layers.{}.self_attn.q_b_proj.weight",
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight",
    "model.layers.{}.self_attn.kv_b_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight",
    "model.layers.{}.mlp.up_proj.weight",
    "model.layers.{}.mlp.down_proj.weight",
    "model.layers.{}.mlp.gate.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias",
    "model.layers.{}.mlp.experts.{}.gate_proj.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight",
]


def test_miles_model_packages_resolve_ahead_of_torchtitan_and_unknown_names_name_both_roots():
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    spec = resolve_model_spec(_args(titan_model_name="glm4_moe_lite", titan_model_flavor="30B-A3B"))
    assert spec.name == "glm4_moe_lite"
    with pytest.raises(ValueError, match="miles.backends.torchtitan_utils.models.nope.*torchtitan.models.nope"):
        resolve_model_spec(_args(titan_model_name="nope", titan_model_flavor="x"))


def test_glm4_7_flash_flavor_matches_its_hf_config_and_its_checkpoint_keys():
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    spec = resolve_model_spec(_args(titan_model_name="glm4_moe_lite", titan_model_flavor="30B-A3B"))
    hf = _GLM47_FLASH_HF_CONFIG
    model = spec.model
    assert (model.dim, model.vocab_size, len(model.layers)) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["num_hidden_layers"],
    )
    attention = model.layers[0].attention
    assert attention.n_heads == hf["num_attention_heads"]
    assert (
        attention.q_lora_rank,
        attention.kv_lora_rank,
        attention.qk_nope_head_dim,
        attention.qk_rope_head_dim,
        attention.v_head_dim,
    ) == (hf["q_lora_rank"], hf["kv_lora_rank"], hf["qk_nope_head_dim"], hf["qk_rope_head_dim"], hf["v_head_dim"])
    assert attention.rope.theta == hf["rope_theta"] and attention.rope.scaling == "none"
    assert [layer.moe is None for layer in model.layers] == [i < hf["first_k_dense_replace"] for i in range(47)]
    assert model.layers[0].feed_forward.w1.out_features == hf["intermediate_size"]
    moe = model.layers[1].moe
    assert moe.num_experts == hf["n_routed_experts"]
    assert moe.routed_experts.inner_experts.hidden_dim == hf["moe_intermediate_size"]
    assert moe.shared_experts.w1.out_features == hf["moe_intermediate_size"] * hf["n_shared_experts"]
    assert (moe.router.top_k, moe.router.score_func, moe.router.route_scale, moe.router.route_norm) == (
        hf["num_experts_per_tok"],
        "sigmoid",
        hf["routed_scaling_factor"],
        hf["norm_topk_prob"],
    )
    assert model.mtp_layers == []

    adapter = spec.state_dict_adapter(model, None)
    assert set(_GLM47_FLASH_CHECKPOINT_KEYS) <= set(adapter.from_hf_map)
