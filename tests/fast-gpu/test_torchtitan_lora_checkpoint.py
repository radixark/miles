from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import os
import socket
import weakref
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from tests.fast.dist_utils import find_free_port, init_gloo, run_multiprocess

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3 import build_model_config

from miles.backends.torchtitan_utils import actor as titan_actor
from miles.backends.torchtitan_utils import lora_checkpoint
from miles.backends.torchtitan_utils.config import _checkpointer_config
from miles.backends.torchtitan_utils.lora_checkpoint import (
    TorchtitanAdapterSlots,
    TorchtitanCheckpointLoader,
    TorchtitanLoraCheckpoint,
    TorchtitanOptimizerMembers,
    load_lr_schedulers,
    probe_torchtitan_row_mapping,
)
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import load_adapter_source, save_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.adapter_state import RanksFailedError
from miles.backends.training_utils.checkpoint.adapter.commit import commit_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    map_named_tensors_to_hf_keys,
    map_peft_keys_to_hf_keys,
)
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets
from miles.utils.timer import Timer

_TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 28})
_QKV, _WO, _W13 = "layers.3.attention.qkv_linear.wqkv", "layers.3.attention.wo", "layers.3.feed_forward.w13"


_QWEN3_CONFIG = SimpleNamespace(to_dict=lambda: {"model_type": "qwen3", "hidden_size": 8, "vocab_size": 16})


def _adapter(rank: int = 8) -> tuple[object, dict[str, torch.Tensor]]:
    model = build_model_config("0.6B", parallelism=ParallelismConfig(), seq_len=4096, attn_backend="flex")
    sd_adapter = model._owner.state_dict_adapter_cls(model, None)
    qkv = model.layers[3].attention.qkv_linear.wqkv
    g = torch.Generator().manual_seed(0)
    params = {
        f"{_QKV}.lora_a.weight": torch.randn(rank, qkv.in_features, generator=g),
        f"{_QKV}.lora_b.weight": torch.randn(qkv.out_features, rank, generator=g),
        f"{_WO}.lora_a.weight": torch.randn(rank, 2048, generator=g),
        f"{_WO}.lora_b.weight": torch.randn(1024, rank, generator=g),
        f"{_W13}.lora_a.weight": torch.randn(rank, 1024, generator=g),
        f"{_W13}.lora_b.weight": torch.randn(2, 3072, rank, generator=g),
    }
    return sd_adapter, params


def _expected(sd_adapter, params) -> dict[str, torch.Tensor]:
    weights, _ = map_named_tensors_to_hf_keys(
        iter_hf_adapter_tensors(sd_adapter, params), map_peft_keys_to_hf_keys, _TARGETS
    )
    return weights


def _check(mapping, params_local: list[torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    flat = torch.cat([p.reshape(-1) for p in params_local])
    for piece in mapping.pieces:
        box = tuple(slice(o, o + n) for o, n in zip(piece.offsets, piece.local_elements.shape, strict=True))
        assert torch.equal(flat[piece.local_elements], expected[piece.key][box]), piece.key


def _whole(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    sd_adapter, params = _adapter()
    mapping = probe_torchtitan_row_mapping(params, sd_adapter)
    attn, mlp = "model.layers.3.self_attn", "model.layers.3.mlp"
    assert mapping.shared_lora_a_groups == (
        (f"{mlp}.gate_proj", f"{mlp}.up_proj"),
        (f"{attn}.k_proj", f"{attn}.q_proj", f"{attn}.v_proj"),
    )
    # each shared lora_A is stored once, under its sorted group's first member
    assert sorted(p.key for p in mapping.pieces) == sorted(
        [f"{attn}.k_proj.lora_A", f"{attn}.o_proj.lora_A", f"{mlp}.gate_proj.lora_A"]
        + [f"{attn}.{m}_proj.lora_B" for m in "qkvo"]
        + [f"{mlp}.{m}_proj.lora_B" for m in ("gate", "up")]
    )
    _check(mapping, list(params.values()), _expected(sd_adapter, params))
    # one index per local element stays resident for the run
    assert {p.local_elements.dtype for p in mapping.pieces} == {torch.int32}
    dist.destroy_process_group()


def test_whole_tensors_map_through_the_models_own_split():
    run_multiprocess(_whole, world_size=1)


_LORA_A, _LORA_B = "layers.0.attention.wq.lora_a.weight", "layers.0.attention.wq.lora_b.weight"


@pytest.fixture
def single_rank_gloo():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    init_gloo_group()
    try:
        yield
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class _Run:
    base_model: str
    lora_alpha: int | None


def _load_dcp(tmp_path, saved: dict) -> TorchtitanCheckpointLoader:
    path = tmp_path / "step-10"
    dcp.save(saved, checkpoint_id=path)
    params = {key: torch.nn.Parameter(torch.zeros(2, 2)) for key in (_LORA_A, _LORA_B)}
    slots = SimpleNamespace(parameters=params)
    members = SimpleNamespace(adam_states=lambda: [{"step": torch.tensor(0.0)} for _ in params])
    sd_adapter = SimpleNamespace(to_hf=lambda tensors: {})
    loader = TorchtitanCheckpointLoader(slots, members, [], sd_adapter, _Run(str(tmp_path), 16), load_optimizer=False)
    with (
        patch.object(lora_checkpoint, "resolve_hf_dir", return_value=tmp_path),
        patch.object(lora_checkpoint, "read_base_check_tensors", return_value={}),
        patch.object(lora_checkpoint, "assert_same_base_tensors"),
    ):
        assert loader.parametrization(path)[1] is False
        # the checks read the source only; nothing is loaded until load_adapter_source accepted it
        assert all(torch.equal(param, torch.zeros(2, 2)) for param in params.values())
        loader.apply(loader.prepare(path))
    for key, param in params.items():
        assert torch.equal(param, saved[key])
    return loader


def test_a_model_only_dcp_loads_the_adapters_without_training_state(tmp_path, single_rank_gloo):
    """torchtitan's default final checkpoint (last_save_model_only) holds no train_state or lr_scheduler."""
    loader = _load_dcp(tmp_path, {_LORA_A: torch.ones(2, 2), _LORA_B: torch.full((2, 2), 2.0)})
    assert loader.training_state() is None


def test_a_full_dcp_loads_its_training_state(tmp_path, single_rank_gloo):
    saved = {_LORA_A: torch.ones(2, 2), _LORA_B: torch.full((2, 2), 2.0)}
    saved |= {"train_state.step": 8, "train_state.ntokens_seen": 640, "lr_scheduler.last_epoch": 8}
    loader = _load_dcp(tmp_path, saved)
    assert loader.training_state() == ({"step": 8, "ntokens_seen": 640}, {"last_epoch": 8})


@pytest.mark.parametrize("with_optimizer", [True, False])
def test_a_dcp_reports_whether_it_holds_optimizer_state(tmp_path, single_rank_gloo, with_optimizer):
    """load_adapter_source refuses a source without it unless --no-load-optim."""
    saved = {_LORA_A: torch.ones(2, 2), _LORA_B: torch.ones(2, 2)}
    if with_optimizer:
        for key in (_LORA_A, _LORA_B):
            saved |= {f"optimizer.state.{key}.{name}": torch.zeros(2, 2) for name in ("exp_avg", "exp_avg_sq")}
            saved[f"optimizer.state.{key}.step"] = torch.tensor(1.0)
    path = tmp_path / "step-10"
    dcp.save(saved, checkpoint_id=path, no_dist=True)
    params = {key: torch.nn.Parameter(torch.zeros(2, 2)) for key in (_LORA_A, _LORA_B)}
    loader = TorchtitanCheckpointLoader(
        SimpleNamespace(parameters=params), None, [], None, _Run(str(tmp_path), 16), load_optimizer=True
    )
    with (
        patch.object(lora_checkpoint, "resolve_hf_dir", return_value=tmp_path),
        patch.object(lora_checkpoint, "read_base_check_tensors", return_value={}),
        patch.object(lora_checkpoint, "assert_same_base_tensors"),
        patch.object(loader, "_base_check_tensors", return_value={}),
    ):
        assert loader.parametrization(path) == (replace(_Run(str(tmp_path), 16), lora_alpha=None), with_optimizer)


def _base_check_worker(rank: int, world_size: int, port: int, path) -> None:
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = {key: torch.nn.Parameter(torch.zeros(2, 2)) for key in (_LORA_A, _LORA_B)}
    members = SimpleNamespace(adam_states=lambda: [{"step": torch.tensor(0.0)} for _ in params])
    sd_adapter = SimpleNamespace(to_hf=lambda tensors: {})
    loader = TorchtitanCheckpointLoader(
        SimpleNamespace(parameters=params), members, [], sd_adapter, _Run("hub/model", 16), load_optimizer=False
    )
    with (
        patch.object(lora_checkpoint, "resolve_hf_dir", return_value=path if rank == 0 else None),
        patch.object(lora_checkpoint, "read_base_check_tensors", return_value={}),
        patch.object(lora_checkpoint, "assert_same_base_tensors"),
        pytest.raises(RuntimeError, match="rank 1: .*has no local HF weights"),
    ):
        loader.parametrization(path)
    dist.destroy_process_group()


def test_a_base_check_failing_on_one_rank_fails_every_rank(tmp_path):
    """Otherwise the other ranks wait in the collective dcp.load for the rank that failed."""
    path = tmp_path / "step-10"
    dcp.save({_LORA_A: torch.ones(2, 2), _LORA_B: torch.ones(2, 2)}, checkpoint_id=path, no_dist=True)
    mp.spawn(_base_check_worker, args=(2, find_free_port(), path), nprocs=2, join=True)


class _Norms(torch.nn.Module):
    """The titan parameters the base check reads, under torchtitan's names: ``norm`` and layer 0's attention norm."""

    def __init__(self, norm: torch.Tensor):
        super().__init__()
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(norm.clone())
        self.layers = torch.nn.ModuleDict({"0": torch.nn.Module()})
        self.layers["0"].attention_norm = torch.nn.Module()
        self.layers["0"].attention_norm.weight = torch.nn.Parameter(norm.clone() + 1)


@pytest.mark.parametrize("same_base", [True, False])
def test_the_base_check_maps_the_dcps_norms_through_the_real_qwen3_adapter(tmp_path, single_rank_gloo, same_base):
    from safetensors.torch import save_file

    sd_adapter, _ = _adapter()
    part = _Norms(torch.arange(4.0))
    base = {fqn: param.detach().clone() for fqn, param in part.named_parameters()}
    path = tmp_path / "step-10"
    dcp.save(base | {_LORA_A: torch.ones(2, 2), _LORA_B: torch.ones(2, 2)}, checkpoint_id=path, no_dist=True)
    hf = sd_adapter.to_hf(base)
    assert set(hf) == {"model.norm.weight", "model.layers.0.input_layernorm.weight"}
    if not same_base:
        hf["model.norm.weight"] = hf["model.norm.weight"] + 1
    (tmp_path / "hf").mkdir()
    save_file(hf, str(tmp_path / "hf" / "model.safetensors"))
    params = {key: torch.nn.Parameter(torch.zeros(2, 2)) for key in (_LORA_A, _LORA_B)}
    members = SimpleNamespace(adam_states=lambda: [{"step": torch.tensor(0.0)} for _ in params])
    loader = TorchtitanCheckpointLoader(
        SimpleNamespace(parameters=params), members, [part], sd_adapter, _Run(str(tmp_path / "hf"), 16), False
    )
    if same_base:
        loader.parametrization(path)
    else:
        with pytest.raises(RuntimeError, match=r"another base model .*model\.norm\.weight"):
            loader.parametrization(path)


_STEPS_PER_ROLLOUT = 4


class _Engine:
    def __init__(self):
        self.num_completed_steps, self.ntokens_seen = 0, 0
        self.optim = SimpleNamespace(lr_schedulers=MagicMock())
        self.state_dict_adapter = None

    def load_state_dict(self, state: dict) -> None:
        self.num_completed_steps, self.ntokens_seen = state["step"], state["ntokens_seen"]


class _Trainer:
    model_parts = []

    def __init__(self):
        self.engine = _Engine()
        self.optimizers = SimpleNamespace(optimizers=[])

    @property
    def step(self) -> int:
        return self.engine.num_completed_steps


def _restore(tmp_path, *, dcp_step=None, training_state=None, **flags):
    """``dcp_step``: the source is a torchtitan DCP with this train_state step; ``training_state``: an adapter dir's."""

    def load_adapter_source(path, members, slot, run, targets, *, load_optimizer, in_model_loaders_by_kind):
        if dcp_step is not None:
            (loader,) = in_model_loaders_by_kind.values()
            loader.training_state = lambda: ({"step": dcp_step, "ntokens_seen": 7}, {"last_epoch": dcp_step})
        return run

    backend = SimpleNamespace(no_load_optim=False, no_save_optim=False, save=None)
    args = SimpleNamespace(
        lora_adapter_path=str(tmp_path / "source"),
        lora_adapter_convert_save=None,
        num_rollout=10,
        start_rollout_id=None,
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        hf_checkpoint=str(tmp_path),
        lora_rank=8,
        lora_alpha=16,
        lora_type="lora",
        backend=backend,
    )
    for name, value in flags.items():
        setattr(backend if hasattr(backend, name) else args, name, value)
    trainer = _Trainer()
    slots = MagicMock()
    with (
        patch.object(lora_checkpoint, "load_hf_config", return_value=_QWEN3_CONFIG),
        patch.object(lora_checkpoint, "_torchtitan_adapter_slots", return_value=slots),
        patch.object(lora_checkpoint, "load_adapter_source", side_effect=load_adapter_source),
        patch.object(lora_checkpoint, "read_training_state", return_value=training_state),
        patch.object(lora_checkpoint, "load_lr_schedulers") as load_lr_schedulers,
        patch.object(lora_checkpoint, "save_and_commit_adapter_dir") as save,
    ):
        checkpoint = TorchtitanLoraCheckpoint(trainer, args, _STEPS_PER_ROLLOUT)
        iteration = checkpoint.restore()
    lr_states = [call.args[1] for call in load_lr_schedulers.call_args_list]
    return SimpleNamespace(
        iteration=iteration,
        step=trainer.step,
        lr_states=lr_states,
        save=save,
        base_model=slots.parametrization.call_args.kwargs["base_model"],
        checkpoint=checkpoint,
    )


@pytest.mark.parametrize("step", [6, 11 * _STEPS_PER_ROLLOUT + 1])
def test_a_dcp_off_this_runs_rollout_grid_is_refused(tmp_path, step):
    """A standalone torchtitan run's step count says nothing about this run's rollouts."""
    with pytest.raises(AssertionError, match=f"at step {step}, not a whole rollout"):
        _restore(tmp_path, dcp_step=step)


@pytest.mark.parametrize("step", [6, 11 * _STEPS_PER_ROLLOUT + 1])
def test_a_dcp_off_this_runs_rollout_grid_loads_its_weights_and_starts_fresh(tmp_path, step):
    restored = _restore(tmp_path, dcp_step=step, no_load_optim=True, lora_adapter_convert_save=str(tmp_path / "out"))
    assert (restored.iteration, restored.step, restored.lr_states) == (None, 0, [])
    assert restored.save.call_args.kwargs["training_state"] is None


@pytest.mark.parametrize("save_moments", [True, False])
def test_the_adapter_state_carries_moments_only_when_they_are_saved(save_moments):
    """--no-save-optim writes the fp32 masters alone, as Megatron does."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    slots = SimpleNamespace(
        parameters={_LORA_A: param},
        slot_parameters=lambda slot: [param],
        get_local_shard_specs=lambda slot: [SimpleNamespace(key="q_proj.lora_A", local_shape=(2, 2))],
        local_pieces=lambda slot, tensors: tensors,
    )
    optimizer = torch.optim.AdamW([param])
    state = TorchtitanOptimizerMembers(slots, [optimizer], save_moments=save_moments).export_adapter_state(0)
    assert state.has_optimizer_state is save_moments
    assert torch.equal(state.weights[0], param.detach())
    assert len(state.exp_avgs) == (1 if save_moments else 0)


def test_a_dcp_resumes_after_its_last_whole_rollout(tmp_path):
    restored = _restore(tmp_path, dcp_step=2 * _STEPS_PER_ROLLOUT)
    assert (restored.iteration, restored.step) == (1, 8)
    assert restored.lr_states == [{"last_epoch": 8}]


def test_a_dcp_at_step_zero_starts_the_run(tmp_path):
    restored = _restore(tmp_path, dcp_step=0)
    assert (restored.iteration, restored.step) == (None, 0)


def test_a_model_only_dcp_starts_a_fresh_schedule(tmp_path):
    restored = _restore(tmp_path)
    assert (restored.iteration, restored.step, restored.lr_states) == (None, 0, [])


def test_a_torchtitan_adapter_dir_restores_its_own_schedule(tmp_path):
    schedule = {"last_epoch": 12, "_step_count": 13}
    restored = _restore(
        tmp_path,
        training_state={
            "iteration": 2,
            "step": 12,
            "first_rollout": 0,
            "steps_per_rollout": _STEPS_PER_ROLLOUT,
            "lr_scheduler": schedule,
        },
    )
    assert (restored.iteration, restored.step, restored.lr_states) == (2, 12, [schedule])


def test_a_torchtitan_adapter_dir_off_this_runs_rollout_grid_is_refused(tmp_path):
    """Saved at 2 steps per rollout: its iteration-9 schedule would sit at rollout 5 of this run's 4."""
    with pytest.raises(AssertionError, match="at step 20 after iteration 9, but this run takes 4 steps per rollout"):
        _restore(
            tmp_path,
            training_state={
                "iteration": 9,
                "step": 20,
                "first_rollout": 0,
                "steps_per_rollout": 2,
                "lr_scheduler": {"last_epoch": 20},
            },
        )


def test_a_torchtitan_adapter_dir_of_this_runs_steps_per_rollout_resumes_any_step(tmp_path):
    """Compacted rollouts train a data-dependent number of steps each: the run's own dir sits off the grid, but its
    recorded step and schedule are exact for a run that configures the same steps per rollout."""
    schedule = {"last_epoch": 21}
    restored = _restore(
        tmp_path,
        training_state={
            "iteration": 4,
            "step": 21,
            "first_rollout": 0,
            "steps_per_rollout": _STEPS_PER_ROLLOUT,
            "lr_scheduler": schedule,
        },
    )
    assert (restored.iteration, restored.step, restored.lr_states) == (4, 21, [schedule])


def test_a_torchtitan_adapter_dir_off_this_runs_rollout_grid_loads_its_weights_and_starts_fresh(tmp_path):
    restored = _restore(
        tmp_path,
        training_state={
            "iteration": 9,
            "step": 20,
            "first_rollout": 0,
            "steps_per_rollout": 2,
            "lr_scheduler": {"last_epoch": 20},
        },
        no_load_optim=True,
        lora_adapter_convert_save=str(tmp_path / "out"),
    )
    assert (restored.iteration, restored.step, restored.lr_states) == (9, 0, [])
    converted = restored.save.call_args.kwargs["training_state"]
    assert (converted["iteration"], converted["step"], converted["first_rollout"]) == (9, 0, 10)


def test_a_dir_saved_by_a_run_from_start_rollout_id_resumes(tmp_path):
    """--start-rollout-id 5 starts the steps at rollout 5: its rollout-6 save is at step 8, on its own grid."""
    trainer = _Trainer()
    trainer.engine.num_completed_steps = 2 * _STEPS_PER_ROLLOUT
    args = SimpleNamespace(
        start_rollout_id=5,
        hf_checkpoint="hub/model",
        lora_rank=8,
        lora_alpha=16,
        lora_type="lora",
        backend=SimpleNamespace(no_save_optim=False, save=str(tmp_path)),
    )
    with (
        patch.object(lora_checkpoint, "load_hf_config", return_value=_QWEN3_CONFIG),
        patch.object(lora_checkpoint, "_torchtitan_adapter_slots", return_value=MagicMock()),
        patch.object(lora_checkpoint, "save_and_commit_adapter_dir") as save,
    ):
        TorchtitanLoraCheckpoint(trainer, args, _STEPS_PER_ROLLOUT).save(6)
    training_state = save.call_args.kwargs["training_state"]
    restored = _restore(tmp_path, training_state=training_state)
    assert (restored.iteration, restored.step) == (6, 8)
    # the resumed run's own saves count their steps from the same rollout
    assert restored.checkpoint._training_state(7)["first_rollout"] == 5


@pytest.mark.parametrize(
    "source",
    [
        {"dcp_step": 40},
        {
            "training_state": {
                "iteration": 9,
                "step": 40,
                "first_rollout": 0,
                "steps_per_rollout": _STEPS_PER_ROLLOUT,
                "lr_scheduler": {"last_epoch": 40},
            }
        },
        {"training_state": {"iteration": 9, "first_rollout": 0, "opt_param_scheduler": {"num_steps": 320}}},
    ],
)
def test_a_resume_from_another_start_rollout_id_writes_dirs_that_resume(tmp_path, source):
    """Restored at step 40 and started at --start-rollout-id 20: its rollout-29 save is 10 rollouts later, at step 80."""
    resumed = _restore(tmp_path, start_rollout_id=20, **source).checkpoint
    resumed._trainer.engine.num_completed_steps = 80
    restored = _restore(tmp_path, training_state=resumed._training_state(29))
    assert (restored.iteration, restored.step) == (29, 80)


def test_a_megatron_adapter_dir_resumes_the_schedule_after_its_rollout(tmp_path):
    """Megatron's scheduler state does not apply here; rollouts 0..2 are done."""
    restored = _restore(
        tmp_path, training_state={"iteration": 2, "first_rollout": 0, "opt_param_scheduler": {"num_steps": 96}}
    )
    assert (restored.iteration, restored.step) == (2, 3 * _STEPS_PER_ROLLOUT)
    assert restored.lr_states == [{"last_epoch": 3 * _STEPS_PER_ROLLOUT}]


def test_a_megatron_adapter_dir_resumes_from_the_rollout_its_schedule_counts_from(tmp_path):
    """A Megatron run whose schedule counts from rollout 5 trained 2 rollouts by iteration 6."""
    restored = _restore(
        tmp_path, training_state={"iteration": 6, "first_rollout": 5, "opt_param_scheduler": {"num_steps": 64}}
    )
    assert (restored.iteration, restored.step) == (6, 2 * _STEPS_PER_ROLLOUT)
    assert restored.lr_states == [{"last_epoch": 2 * _STEPS_PER_ROLLOUT}]
    # the resumed run's own saves count their steps from the same rollout
    assert restored.checkpoint._training_state(7)["first_rollout"] == 5


@pytest.mark.parametrize(
    "num_steps, flags",
    [
        # one 8-sample step of three rollouts skipped for an inf/nan grad: Megatron's scheduler did not count it
        (88, {}),
        (96, {"use_dynamic_global_batch_size": True}),
    ],
)
def test_a_megatron_adapter_dir_whose_samples_do_not_map_to_rollouts_resumes_with_its_optimizer(
    tmp_path, num_steps, flags
):
    """The rollouts it trained, not the samples Megatron's scheduler counted, place the schedule."""
    training_state = {"iteration": 2, "first_rollout": 0, "opt_param_scheduler": {"num_steps": num_steps}}
    restored = _restore(tmp_path, training_state=training_state, **flags)
    assert (restored.iteration, restored.step) == (2, 3 * _STEPS_PER_ROLLOUT)
    assert restored.lr_states == [{"last_epoch": 3 * _STEPS_PER_ROLLOUT}]


def test_a_convert_save_of_a_model_only_source_writes_no_training_state(tmp_path):
    """As Megatron's convert-save: a step and schedule the source never had would resume as if real."""
    restored = _restore(tmp_path, lora_adapter_convert_save=str(tmp_path / "out"))
    assert restored.save.call_args.kwargs["training_state"] is None


def test_a_convert_save_at_num_rollout_zero_keeps_a_dcps_training_state(tmp_path):
    """The documented convert run passes --num-rollout 0; a DCP past it is still on the rollout grid."""
    restored = _restore(
        tmp_path, dcp_step=2 * _STEPS_PER_ROLLOUT, num_rollout=0, lora_adapter_convert_save=str(tmp_path / "out")
    )
    training_state = restored.save.call_args.kwargs["training_state"]
    assert (training_state["iteration"], training_state["step"]) == (1, 8)
    assert restored.save.call_args.args[2]._save_moments


def test_restored_lr_schedule_reaches_the_optimizers():
    """The first resumed step must train at the restored schedule's lr, not the fresh run's."""
    optimizers = [torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1.0) for _ in range(2)]
    schedulers = LRSchedulersContainer(optimizers, lambda step: 1 / (step + 1))
    load_lr_schedulers(schedulers, {"last_epoch": 3})
    assert [group["lr"] for optimizer in optimizers for group in optimizer.param_groups] == [0.25, 0.25]


def test_lora_resumes_only_through_the_adapter_path(tmp_path):
    """A step dir under --load or --save would resume the adapters through torchtitan's checkpointer."""

    def config(load=None, save=None, lora_adapter_path=None):
        args = SimpleNamespace(backend=SimpleNamespace(load=load, save=save), lora_adapter_path=lora_adapter_path)
        return _checkpointer_config(args, dump_subdir="actor", with_lora=True)

    old = tmp_path / "old"
    step = old / "torchtitan" / "actor" / "checkpoint" / "step-4"
    step.mkdir(parents=True)
    for flag in ("load", "save"):
        with pytest.raises(ValueError, match=f"--{flag} holds the torchtitan checkpoint {step}"):
            config(**{flag: str(old)})
    with pytest.raises(ValueError, match=f"move {step} out of --save and pass its new path as --lora-adapter-path"):
        config(save=str(old), lora_adapter_path=str(step))
    # migrating the old run: its rollout state from --load, its adapters from --lora-adapter-path
    assert config(load=str(old), lora_adapter_path=str(step)).initial_load_path is None
    adapters = tmp_path / "adapters"
    adapter = adapters / "iter_0000003" / "adapter"
    adapter.mkdir(parents=True)
    for flag in ("load", "save"):
        with pytest.raises(ValueError, match=f"--{flag} {adapters} holds LoRA adapter checkpoints"):
            config(**{flag: str(adapters)})
    # --load still restores the rollout state; the adapter comes from --lora-adapter-path
    resumed = config(load=str(adapters), save=str(adapters), lora_adapter_path=str(adapter))
    assert resumed.initial_load_path is None
    base = config(load=str(tmp_path / "empty"), save=str(tmp_path / "save"))
    assert (base.initial_load_path, base.initial_load_model_only, base.initial_load_in_hf) == (None, True, True)


@pytest.mark.parametrize(
    "save, lora_adapter_path, start_rollout_id, restored, expected",
    [
        (None, None, None, None, 0),
        ("/save", None, None, None, 0),
        (None, "/source", None, 2, 3),
        ("/save", "/source", 5, 2, 5),
    ],
)
def test_the_actor_starts_after_the_restored_rollout_and_saves_lora_as_adapter_dirs(
    monkeypatch, save, lora_adapter_path, start_rollout_id, restored, expected
):
    """``init`` builds the LoRA checkpoint only for --save or --lora-adapter-path, starts after the rollout the
    restore returns, and LoRA saves go to iter_N/adapter instead of torchtitan's checkpointer."""
    # init starts the process-wide train_wait timer
    monkeypatch.setattr(Timer(), "start_time", {})
    built = []

    class _LoraCheckpoint:
        def __init__(self, trainer, args, steps_per_rollout):
            built.append(steps_per_rollout)
            self.saved = []

        def restore(self):
            return restored

        def save(self, rollout_id):
            self.saved.append(rollout_id)

    trainer = SimpleNamespace(
        step=0,
        model_parts=[],
        optimizers=SimpleNamespace(optimizers=[]),
        align_token_side_channel=None,
        configure_loss_reduction=lambda: None,
        has_last_stage=lambda: True,
        parallelism=SimpleNamespace(get_optional_mesh=lambda name: None),
        engine=SimpleNamespace(load_checkpoint=lambda: None, save_checkpoint=MagicMock()),
    )
    for name, value in {
        "compat": SimpleNamespace(install=lambda: None),
        "build_trainer_config": lambda *a, **k: SimpleNamespace(),
        "enable_routing_replay": lambda args: None,
        "TrainProfiler": lambda args: SimpleNamespace(on_init_end=lambda: None),
        "TitanTrainer": lambda config: trainer,
        "set_parallel_state": lambda state: None,
        "create_titan_parallel_state": lambda *a, **k: None,
        "install_routing_replay": lambda parts: None,
        "get_parallel_state": lambda: SimpleNamespace(is_metrics_rank=False),
        "TorchtitanLoraCheckpoint": _LoraCheckpoint,
        "clear_memory": lambda: None,
    }.items():
        monkeypatch.setattr(titan_actor, name, value)
    monkeypatch.setattr(titan_actor.TorchNativeTrainRayActor, "_init_common", lambda *a, **k: None)
    monkeypatch.setattr(titan_actor.TorchtitanTrainRayActor, "load_hf_assets", lambda self: None)
    monkeypatch.setattr(titan_actor.TorchtitanTrainRayActor, "_build_weight_updater", lambda *a: None)
    monkeypatch.setattr(titan_actor, "TorchtitanArgsNamespace", SimpleNamespace)
    args = SimpleNamespace(
        backend=SimpleNamespace(save=save),
        hf_checkpoint="/hf",
        num_rollout=10,
        rollout_batch_size=4,
        n_samples_per_prompt=8,
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        debug_rollout_only=False,
        lora_rank=8,
        lora_adapter_path=lora_adapter_path,
        start_rollout_id=start_rollout_id,
        stream_optimizer_state_to_disk=False,
        offload_train=False,
    )
    actor = titan_actor.TorchtitanTrainRayActor.__new__(titan_actor.TorchtitanTrainRayActor)

    start = actor.init(args, "actor", indep_dp_info=SimpleNamespace(quorum_id=0), indep_dp_store_addr=None)

    assert start == expected
    assert built == ([4] if save or lora_adapter_path else [])
    if built:
        actor.args = args
        actor._save_checkpoint(7)
        assert actor._lora_checkpoint.saved == [7]
        trainer.engine.save_checkpoint.assert_not_called()


def _one_rank_fails_decoding(rank, world_size, port):
    # a short timeout turns a rank left in a collective into a failure instead of a hung test
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
    init_gloo_group()
    mesh = init_device_mesh("cpu", (world_size,))
    sd_adapter, params = _adapter()
    sharded = {key: distribute_tensor(value, mesh, [Shard(0)]) for key, value in params.items()}
    real_rectangles = lora_checkpoint.rectangles

    def rectangles(name, mine, *args):
        if rank == 1:
            raise AssertionError(f"rank-local layout failure on {name}")
        return real_rectangles(name, mine, *args)

    lora_checkpoint.rectangles = rectangles
    with pytest.raises(RanksFailedError, match=r"rank 1: AssertionError: rank-local layout failure") as failed:
        probe_torchtitan_row_mapping(sharded, sd_adapter)
    # rank 0 must not fail on its own, e.g. timed out in a collective rank 1 never joined
    assert "rank 0" not in str(failed.value), str(failed.value)
    dist.destroy_process_group()


def test_a_rank_local_probe_failure_fails_every_rank():
    run_multiprocess(_one_rank_fails_decoding)


def _one_projection_at_a_time(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    sd_adapter, params = _adapter()
    real_local_codes, placed, most_alive = lora_checkpoint._local_codes, [], 0

    def local_codes(param, start):
        nonlocal most_alive
        most_alive = max(most_alive, sum(ref() is not None for ref in placed))
        codes = real_local_codes(param, start)
        placed.append(weakref.ref(codes))
        return codes

    lora_checkpoint._local_codes = local_codes
    mapping = probe_torchtitan_row_mapping(params, sd_adapter)
    _check(mapping, list(params.values()), _expected(sd_adapter, params))
    # a projection's lora_b is placed beside its own lora_a, never beside an earlier projection's codes
    assert most_alive == 1, most_alive
    dist.destroy_process_group()


def test_the_probe_holds_one_projections_codes_at_a_time():
    run_multiprocess(_one_projection_at_a_time, world_size=1)


def _uneven_rows(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    mesh = init_device_mesh("cpu", (world_size,))
    sd_adapter, params = _adapter()
    expected = _expected(sd_adapter, params)
    # wqkv's 4096 lora_B rows over 3 ranks: 1366 / 1366 / 1364, so rank 1's shard crosses q_proj into k_proj
    sharded = {key: distribute_tensor(value, mesh, [Shard(0)]) for key, value in params.items()}
    mapping = probe_torchtitan_row_mapping(sharded, sd_adapter)
    _check(mapping, [p.to_local() for p in sharded.values()], expected)
    attn = "model.layers.3.self_attn"
    keys = {p.key for p in mapping.pieces}
    if rank == 1:
        assert {f"{attn}.q_proj.lora_B", f"{attn}.k_proj.lora_B"} <= keys, "a shard splitting HF modules maps to both"
    dist.destroy_process_group()


def test_an_uneven_shard_across_hf_modules_maps_to_each_module():
    run_multiprocess(_uneven_rows, world_size=3)


def _two_dim_mesh(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    mesh = init_device_mesh("cpu", (2, 2))
    sd_adapter, params = _adapter()
    expected = _expected(sd_adapter, params)
    # lora_A (rank, in) is split along its input columns as well as its rows
    sharded = {key: distribute_tensor(value, mesh, [Shard(0), Shard(1)]) for key, value in params.items()}
    mapping = probe_torchtitan_row_mapping(sharded, sd_adapter)
    _check(mapping, [p.to_local() for p in sharded.values()], expected)
    dist.destroy_process_group()


def test_a_two_dim_mesh_with_column_shards_maps_to_rectangles():
    run_multiprocess(_two_dim_mesh, world_size=4)


def _model_part(params: dict[str, torch.Tensor]) -> torch.nn.Module:
    """A model part whose named_parameters() are ``params`` under their titan keys."""
    part = torch.nn.Module()
    for key, value in params.items():
        *path, name = key.split(".")
        module = part
        for step in path:
            if not hasattr(module, step):
                module.add_module(step, torch.nn.Module())
            module = getattr(module, step)
        module.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    return part


def _slots_on_strided_dtensors(rank, world_size, port, placements):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    mesh = init_device_mesh("cpu", (2, 2))
    sd_adapter, params = _adapter()
    expected = _expected(sd_adapter, params)
    replicated = [Replicate(), Replicate()]
    sharded = {
        key: distribute_tensor(value, mesh, replicated).redistribute(mesh, placements) for key, value in params.items()
    }

    slots = TorchtitanAdapterSlots([_model_part(sharded)], None, sd_adapter, _TARGETS)

    specs = slots.get_local_shard_specs(0)
    pieces = slots.local_pieces(0, slots.slot_parameters(0))
    assert specs and len(specs) == len(pieces)
    for spec, piece in zip(specs, pieces, strict=True):
        box = tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))
        assert torch.equal(piece.reshape(spec.local_shape), expected[spec.key][box]), spec.key
    dist.destroy_process_group()


def test_the_slots_own_pieces_of_strided_dtensors_are_their_hf_slices():
    # FSDP2 over TP on the same dim
    placements = [_StridedShard(0, split_factor=2), Shard(0)]
    run_multiprocess(partial(_slots_on_strided_dtensors, placements=placements), world_size=4)


def _members_on(mesh, placements, sd_adapter, weights, moments=None):
    """Members of a model part holding ``weights`` so placed; ``moments``: their AdamW (exp_avg, exp_avg_sq) at step 5."""

    def place(tensor):
        return distribute_tensor(tensor, mesh, [Replicate(), Replicate()]).redistribute(mesh, placements)

    part = _model_part({key: place(value) for key, value in weights.items()}).requires_grad_(True)
    slots = TorchtitanAdapterSlots([part], None, sd_adapter, _TARGETS)
    members = TorchtitanOptimizerMembers(slots, [torch.optim.AdamW(part.parameters())], save_moments=True)
    if moments is not None:
        for key, state in zip(slots.parameters, members.adam_states(), strict=True):
            for name, value in zip(("exp_avg", "exp_avg_sq"), moments[key], strict=True):
                state[name].to_local().copy_(place(value).to_local())
            state["step"].fill_(5)
    return slots, members


def _cross_layout_round_trip(rank, world_size, port, path, source, target):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    mesh = init_device_mesh("cpu", (2, 2))
    sd_adapter, params = _adapter()
    g = torch.Generator().manual_seed(1)
    moments = {key: (torch.randn(v.shape, generator=g), torch.rand(v.shape, generator=g)) for key, v in params.items()}
    slots, saved = _members_on(mesh, source, sd_adapter, params, moments)
    run = slots.parametrization(
        base_model="hub/model",
        lora_rank=8,
        lora_alpha=16,
        lora_type="lora",
        experts_shared_outer_loras=False,
        base_model_fingerprint=None,
    )
    written = save_adapter_dir(path, "adapter", saved, 0, run, tenant_digest=None)
    if rank == 0:
        commit_adapter_dir(path, "adapter", written, overwrite=True)
    dist.barrier()

    slots, loaded = _members_on(mesh, target, sd_adapter, {key: torch.zeros_like(v) for key, v in params.items()})
    load_adapter_source(written.path, loaded, 0, run, slots.targets, load_optimizer=True)
    for (key, param), state in zip(slots.parameters.items(), loaded.adam_states(), strict=True):
        assert torch.equal(param.full_tensor(), params[key]), key
        assert torch.equal(state["exp_avg"].full_tensor(), moments[key][0]), key
        assert torch.equal(state["exp_avg_sq"].full_tensor(), moments[key][1]), key
        assert int(state["step"]) == 5, key
    dist.destroy_process_group()


_STRIDED_ROWS = [Shard(0), _StridedShard(0, split_factor=2)]
_COLUMNS_THEN_ROWS = [Shard(1), Shard(0)]


def test_an_adapter_dir_moves_weights_and_moments_across_parallel_layouts(tmp_path):
    args = (4, find_free_port(), tmp_path, _STRIDED_ROWS, _COLUMNS_THEN_ROWS)
    mp.spawn(_cross_layout_round_trip, args=args, nprocs=4, join=True)


def _pipeline_stages(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    sd_adapter, params = _adapter()
    # stage 0 holds the attention layer, stage 1 the MLP: each sees only its own modules
    stage = {key: value for key, value in params.items() if (_W13 in key) == (rank == 1)}
    mapping = probe_torchtitan_row_mapping(stage, sd_adapter)
    attn, mlp = "model.layers.3.self_attn", "model.layers.3.mlp"
    assert mapping.shared_lora_a_groups == (
        (f"{mlp}.gate_proj", f"{mlp}.up_proj"),
        (f"{attn}.k_proj", f"{attn}.q_proj", f"{attn}.v_proj"),
    ), "every stage holds the whole run's shared-A groups"
    assert set(mapping.target_modules) == {f"{attn}.{m}_proj" for m in "qkvo"} | {
        f"{mlp}.{m}_proj" for m in ("gate", "up")
    }, "every stage holds the whole run's target modules"
    _check(mapping, list(stage.values()), _expected(sd_adapter, params))
    dist.destroy_process_group()


def test_pipeline_stages_share_the_runs_groups_and_modules():
    run_multiprocess(_pipeline_stages)
