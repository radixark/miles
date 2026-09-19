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
        titan_seq_len=8192,
        titan_pipeline_parallel_degree=1,
        titan_context_parallel_degree=1,
        titan_expert_parallel_degree=1,
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


def _config_args(**overrides) -> Namespace:
    base = dict(
        optimizer="adam",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        titan_seq_len=4096,
        titan_data_parallel_replicate_degree=1,
        titan_tensor_parallel_degree=1,
        titan_pipeline_parallel_degree=1,
        titan_context_parallel_degree=1,
        titan_expert_parallel_degree=1,
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
            _config_args(titan_pipeline_parallel_degree=2), hf_assets_path=hf, lr_total_steps=1, dump_subdir="x"
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
