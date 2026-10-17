import json
import os
import threading
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


def test_the_trainer_keeps_dtensor_only_on_the_model_parallel_axes(tmp_path, single_gpu_dims):
    """torchtitan v0.3.0 defaults to the spmd_types backend, whose FSDP wrapping wants every
    parameter distributed over the full mesh first and rejects a plain-DP model."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    config = build_trainer_config(
        _config_args(),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert config.parallelism.spmd_backend == "partial_dtensor"


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


def test_the_replayed_router_forward_mirrors_the_pinned_torchtitan_router():
    """routing_replay._token_router_forward is a copy of TokenChoiceTopKRouter.forward with the
    top-k swapped for the replayed one; a torchtitan bump that changes the upstream forward must
    re-sync the copy before this fingerprint is updated."""
    import ast
    import hashlib
    import inspect
    import textwrap

    moe = pytest.importorskip("torchtitan.models.common.moe")

    source = textwrap.dedent(inspect.getsource(moe.TokenChoiceTopKRouter.forward))
    fingerprint = hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()
    assert fingerprint == "e3c6e10d721ebf5bdc8d968524cc204fbc3c74e775157b30248d6ec2f1aecef9"


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


def test_unknown_model_names_name_both_package_roots():
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

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


# Recorded from Qwen/Qwen3.8-Flash-Next config.json (text_config).
_QWEN38_NEXT_HF_CONFIG = dict(
    hidden_size=2560,
    vocab_size=248320,
    num_hidden_layers=48,
    hc_count=4,
    hc_lowrank=320,
    num_attention_heads=24,
    num_key_value_heads=2,
    head_dim=256,
    partial_rotary_factor=0.25,
    rope_theta=10000000,
    full_attention_interval=4,
    num_experts=512,
    num_experts_per_tok=10,
    moe_intermediate_size=640,
    shared_expert_intermediate_size=640,
    ple_layer_ids=[2],
    ngram_size=3,
    heads_per_ngram=8,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_n_heads=4,
    indexer_head_dim=128,
    eos_token_id=248044,
)

# Recorded from the safetensors headers of CharyZeng/Qwen3.8-Flash-Next-4layer, per layer kind;
# mtp.*, model.visual.* and the frozen ple_embedding.* are deliberately not loaded.
_QWEN38_NEXT_HC_SHAPES = {
    "block_inject_weight.weight": [4, 10240],
    "hc_norm.weight": [10240],
    "input_mix_weight_down.weight": [320, 10240],
    "input_mix_weight_up.weight": [10240, 320],
}
_QWEN38_NEXT_LAYER_SHAPES = {
    **{
        f"{site}.{k}": v
        for site in ("attn_hyper_connection", "mlp_hyper_connection")
        for k, v in _QWEN38_NEXT_HC_SHAPES.items()
    },
    "mlp.experts.down_proj": [512, 2560, 640],
    "mlp.experts.gate_up_proj": [512, 1280, 2560],
    "mlp.gate.weight": [512, 2560],
    "mlp.shared_expert.down_proj.weight": [2560, 640],
    "mlp.shared_expert.gate_proj.weight": [640, 2560],
    "mlp.shared_expert.up_proj.weight": [640, 2560],
    "mlp.shared_expert_gate.weight": [1, 2560],
}
_QWEN38_NEXT_GDN_SHAPES = {
    "linear_attn.A_log": [48],
    "linear_attn.conv1d.weight": [10240, 1, 4],
    "linear_attn.dt_bias": [48],
    "linear_attn.in_proj_a.weight": [48, 2560],
    "linear_attn.in_proj_b.weight": [48, 2560],
    "linear_attn.in_proj_qkv.weight": [10240, 2560],
    "linear_attn.in_proj_z.weight": [6144, 2560],
    "linear_attn.norm.weight": [128],
    "linear_attn.out_proj.weight": [2560, 6144],
}
_QWEN38_NEXT_QSA_SHAPES = {
    "self_attn.indexer.index_qk_proj.weight": [640, 2560],
    "self_attn.indexer.k_layernorm.weight": [128],
    "self_attn.indexer.q_layernorm.weight": [128],
    "self_attn.k_norm.weight": [256],
    "self_attn.k_proj.weight": [512, 2560],
    "self_attn.o_proj.weight": [2560, 6144],
    "self_attn.q_norm.weight": [256],
    "self_attn.q_proj.weight": [12288, 2560],
    "self_attn.v_proj.weight": [512, 2560],
}
_QWEN38_NEXT_PLE_SHAPES = {
    "ple.conv1d.weight": [10240, 1, 4],
    "ple.key_proj.weight": [10240, 2560],
    "ple.norm_conv.weight": [10240],
    "ple.norm_key.weight": [10240],
    "ple.norm_query.weight": [10240],
    "ple.value_proj.weight": [2560, 2560],
}
_QWEN38_NEXT_GLOBAL_SHAPES = {
    "lm_head.weight": [248320, 2560],
    "model.language_model.embed_tokens.weight": [248320, 2560],
    "model.language_model.hyper_connection_mixer.hc_norm.weight": [10240],
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": [320, 10240],
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": [10240, 320],
}


def _qwen38_next_checkpoint_shapes(n_layers: int) -> dict[str, list[int]]:
    hf = _QWEN38_NEXT_HF_CONFIG
    ple_layers = {i - 1 for i in hf["ple_layer_ids"]}
    shapes = dict(_QWEN38_NEXT_GLOBAL_SHAPES)
    for layer in range(n_layers):
        full = (layer + 1) % hf["full_attention_interval"] == 0
        suffixes = {
            **_QWEN38_NEXT_LAYER_SHAPES,
            **(_QWEN38_NEXT_QSA_SHAPES if full else _QWEN38_NEXT_GDN_SHAPES),
            **(_QWEN38_NEXT_PLE_SHAPES if layer in ple_layers else {}),
        }
        shapes.update({f"model.language_model.layers.{layer}.{k}": v for k, v in suffixes.items()})
    return shapes


@pytest.mark.parametrize("flavor, n_layers", [("Flash-Next-4layer", 4), ("Flash-Next", 48)])
def test_qwen3_8_flash_next_flavor_exports_exactly_the_checkpoint_shapes(flavor, n_layers):
    pytest.importorskip("torchtitan")
    pytest.importorskip("triton")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    spec = resolve_model_spec(_args(titan_model_name="qwen3_8_next", titan_model_flavor=flavor))
    hf = _QWEN38_NEXT_HF_CONFIG
    attention = spec.model.layers[hf["full_attention_interval"] - 1].attention
    assert (attention.n_heads, attention.n_kv_heads, attention.head_dim) == (
        hf["num_attention_heads"],
        hf["num_key_value_heads"],
        hf["head_dim"],
    )
    assert attention.rotary_dim == hf["head_dim"] * hf["partial_rotary_factor"]
    assert attention.rope.theta == hf["rope_theta"] == attention.indexer_rope_theta
    assert (attention.indexer_budget, attention.indexer_compress_ratio) == (
        hf["indexer_budget"],
        hf["indexer_compress_ratio"],
    )
    moe = spec.model.layers[0].moe
    assert (moe.router.top_k, moe.router.score_func, moe.router.route_norm) == (
        hf["num_experts_per_tok"],
        "softmax",
        True,
    )
    # sglang routes without an expert bias; torchtitan's default would learn one
    assert moe.load_balance_coeff is None
    assert [layer.ple is not None for layer in spec.model.layers] == [
        i + 1 in hf["ple_layer_ids"] for i in range(n_layers)
    ]

    with torch.device("meta"):
        model = spec.model.build()
    exported = spec.state_dict_adapter(spec.model, None).to_hf(model.state_dict())
    assert {k: list(v.shape) for k, v in exported.items()} == _qwen38_next_checkpoint_shapes(n_layers)

    reloaded = spec.state_dict_adapter(spec.model, None).from_hf(exported)
    assert set(reloaded) == {name for name, _ in model.named_parameters()}


def _fake_ple_checkpoint(path, *, rows_per_shard=3, dim=4, shards=2):
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    tensors = {
        f"{prefix}{i}.weight": torch.arange(rows_per_shard * dim, dtype=torch.bfloat16).reshape(rows_per_shard, dim)
        + i
        for i in range(shards)
    }
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    return tensors


@pytest.fixture
def ple_module(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")
    pytest.importorskip("triton")
    from miles.backends.torchtitan_utils.models.qwen3_8_next import ple

    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(ple, "_SHM_DIR", str(shm))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    return ple, ckpt, shm


def test_qwen3_8_next_ple_table_is_written_once_and_superseded_when_the_checkpoint_changes(ple_module):
    ple, ckpt, shm = ple_module
    tensors = _fake_ple_checkpoint(ckpt)
    shards = ple.ple_table_shards(str(ckpt), 1)
    path = ple._shm_path(str(ckpt), 1, shards)
    ple._materialize_once(path, shards, rows=6, dim=4)
    table = torch.from_file(path, shared=False, size=24, dtype=torch.bfloat16).view(6, 4)
    assert torch.equal(table, torch.cat(list(tensors.values())))

    os.utime(ckpt / "model.safetensors", ns=(1, 1))  # the checkpoint's contents changed in place
    new_path = ple._shm_path(str(ckpt), 1, ple.ple_table_shards(str(ckpt), 1))
    assert new_path != path
    ple._materialize_once(new_path, shards, rows=6, dim=4)
    assert sorted(os.listdir(shm)) == [os.path.basename(new_path)]


def test_qwen3_8_next_ple_waiters_fail_fast_when_the_writer_gives_up(ple_module):
    ple, _, shm = ple_module
    path = str(shm / "table.bf16")
    open(f"{path}.lock", "w").close()
    # the writer drops its lock without renaming the table into place
    threading.Timer(0.5, os.remove, args=(f"{path}.lock",)).start()
    with pytest.raises(RuntimeError, match="gave up"):
        ple._wait_for_writer(path, f"{path}.lock")


def test_qwen3_8_next_ple_truncated_shard_raises_and_leaves_no_temp_file(ple_module):
    ple, ckpt, shm = ple_module
    _fake_ple_checkpoint(ckpt)
    shards = ple.ple_table_shards(str(ckpt), 1)
    with open(ckpt / "model.safetensors", "r+b") as f:
        f.truncate(shards[-1].end - 4)  # an interrupted download
    path = str(shm / "table.bf16")
    with pytest.raises(OSError, match="download complete"):
        ple._materialize_once(path, shards, rows=6, dim=4)
    assert os.listdir(shm) == []


def test_qwen3_8_next_rejects_pipeline_parallelism():
    pytest.importorskip("torchtitan")
    pytest.importorskip("triton")
    from torchtitan.config import ParallelismConfig

    from miles.backends.torchtitan_utils.config import resolve_model_spec

    spec = resolve_model_spec(_args(titan_model_name="qwen3_8_next", titan_model_flavor="Flash-Next-4layer"))
    config = SimpleNamespace(parallelism=ParallelismConfig(pipeline_parallel_degree=2), hf_assets_path=None)
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        spec.model.update_from_config(config=config)
