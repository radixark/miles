import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.torchtitan_utils import parallel as tp
from miles.backends.torchtitan_utils.arguments import validate_torchtitan_args

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def _args(**overrides) -> Namespace:
    base = dict(
        titan_model_name="qwen3",
        seq_length=8192,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        rollout_max_context_len=8192,
        rollout_max_response_len=4096,
        ref_update_interval=None,
        save_debug_train_data=None,
        fp16=False,
        lr_decay_style="constant",
        lr_warmup_fraction=None,
        lr_wsd_decay_iters=None,
        lr_decay_iters=None,
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
    with pytest.raises(ValueError, match="seq-length"):
        validate_torchtitan_args(_args(seq_length=8192, rollout_max_context_len=16384))
    with pytest.raises(ValueError, match="no room for a prompt"):
        validate_torchtitan_args(_args(seq_length=8192, rollout_max_context_len=None, rollout_max_response_len=8192))
    validate_torchtitan_args(_args(seq_length=16384, rollout_max_context_len=None, rollout_max_response_len=8192))


def test_context_parallelism_is_rejected_for_qwen3_5(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="context parallelism"):
        validate_torchtitan_args(_args(titan_model_name="qwen3_5", context_parallel_size=2))


def test_unsupported_flags_are_rejected_rather_than_ignored(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def _config_args(**overrides) -> Namespace:
    base = dict(
        optimizer="adam",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        seq_length=4096,
        dp_replicate_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        global_batch_size=8,
        micro_batch_size=1,
        clip_grad=1.0,
        lr=1e-6,
        min_lr=0.0,
        lr_warmup_iters=0,
        lr_decay_style="constant",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        weight_decay=0.1,
        seed=1,
        gradient_checkpointing=False,
        save=None,
        load=None,
    )
    return Namespace(**{**base, **overrides})


def _checkpoint_dir(tmp_path, **config) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", **config}))
    return str(tmp_path)


@pytest.fixture
def single_gpu_dims(monkeypatch):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import config as titan_config

    monkeypatch.setattr(
        titan_config,
        "parallel_dims_from_config",
        lambda parallelism: SimpleNamespace(dp_replicate=1, dp_shard=1, pp_enabled=False),
    )


def test_a_tied_checkpoint_is_refused_under_pipeline_parallelism(tmp_path):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="pipeline"):
        build_trainer_config(
            _config_args(pipeline_model_parallel_size=2), hf_assets_path=hf, lr_total_steps=1, dump_subdir="x"
        )


def test_the_lr_schedule_follows_miles_flags_not_torchtitan_defaults(tmp_path, single_gpu_dims):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    constant = build_trainer_config(
        _config_args(lr_warmup_iters=3), hf_assets_path=hf, lr_total_steps=10, dump_subdir="x"
    )
    assert (constant.lr_scheduler.warmup_steps, constant.lr_scheduler.min_lr_factor) == (3, 1.0)
    cosine = build_trainer_config(
        _config_args(lr_decay_style="cosine", lr=1e-6, min_lr=1e-7),
        hf_assets_path=hf,
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert cosine.lr_scheduler.decay_type == "cosine"
    assert cosine.lr_scheduler.min_lr_factor == pytest.approx(0.1)


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


def test_the_loss_adapter_undoes_the_summed_dp_and_cp_gradients():
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_gradient_scale(1.0 / 8)
    batch = {"tokens": torch.zeros(1)}
    adapter.arm([batch], lambda pred, b: (pred.sum(), {"seen": b is batch}), is_training=True)
    loss, _ = adapter(torch.full((2,), 4.0), torch.zeros(2, dtype=torch.long))
    assert loss.item() == pytest.approx(1.0)
    assert adapter.collect() == [{"seen": True}]


def test_resume_reads_from_load_and_writes_to_save(tmp_path, single_gpu_dims):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path / "hf", tie_word_embeddings=False)
    load_root = tmp_path / "load"
    for step in (3, 12):
        (load_root / "torchtitan" / "actor" / "checkpoint" / f"step-{step}").mkdir(parents=True)
    save_root = tmp_path / "save"

    config = build_trainer_config(
        _config_args(load=str(load_root), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert config.dump_folder == str(save_root / "torchtitan" / "actor")
    assert config.checkpoint.initial_load_path == str(load_root / "torchtitan" / "actor" / "checkpoint" / "step-12")
    assert (config.checkpoint.initial_load_model_only, config.checkpoint.initial_load_in_hf) == (False, False)
    assert config.checkpoint.last_save_model_only is False

    fresh = build_trainer_config(
        _config_args(load=str(tmp_path / "empty"), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert fresh.checkpoint.initial_load_path is None
    assert (fresh.checkpoint.initial_load_model_only, fresh.checkpoint.initial_load_in_hf) == (True, True)


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
