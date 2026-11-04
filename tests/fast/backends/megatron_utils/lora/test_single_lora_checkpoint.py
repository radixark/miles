"""Bridge single LoRA's --save / --lora-adapter-path glue around the adapter library."""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
import torch
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from tests.fast.dist_utils import run_multiprocess

import miles.backends.megatron_utils.lora.checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora.checkpoint import (
    DistributedOptimizerMembers,
    LegacyRankShardLoader,
    _distributed_optimizer_step,
    _set_distributed_optimizer_step,
    _world_slice,
    restore_single_lora,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    assert_parametrization_matches_run,
)

_MODULE = "miles.backends.megatron_utils.lora.checkpoint"


def _restore(path, *, convert_save=None, source_alpha=16.0):
    """``restore_single_lora`` with the model-facing pieces mocked; returns (iteration, scheduler, mocks)."""
    args = SimpleNamespace(
        hf_checkpoint="/hf",
        lora_adapter_path=str(path),
        lora_adapter_convert_save=convert_save,
        lora_rank=8,
        lora_alpha=16.0,
        lora_type="lora",
        experts_shared_outer_loras=False,
        megatron_to_hf_mode="bridge",
        multi_lora=False,
        backend=SimpleNamespace(no_load_optim=True, no_save_optim=False, use_precision_aware_optimizer=False),
    )
    scheduler = MagicMock()
    slots = MagicMock()
    slots.parametrization.return_value = SimpleNamespace(lora_alpha=16.0)
    with patch.multiple(
        _MODULE,
        DistributedOptimizerMembers=DEFAULT,
        load_adapter_dir=DEFAULT,
        save_and_commit_adapter_dir=DEFAULT,
        load_hf_config=lambda path: SimpleNamespace(to_dict=lambda: {"model_type": "qwen3"}),
    ) as mocks:
        mocks["load_adapter_dir"].return_value = SimpleNamespace(lora_alpha=source_alpha)
        iteration = restore_single_lora(
            [], args, adapter_slots=slots, optimizer=MagicMock(), opt_param_scheduler=scheduler
        )
    return iteration, scheduler, mocks


def test_an_old_format_save_resumes_its_iteration_whichever_reader_loads_the_weights(tmp_path):
    # --no-load-optim reads an old-format save's PEFT pair, not its rank shards
    torch.save(
        {"iteration": 7, "optimizer": None, "parameter_state": None, "opt_param_scheduler": {"num_steps": 56}},
        tmp_path / "training_state_rank0.pt",
    )
    iteration, scheduler, _ = _restore(tmp_path)
    assert iteration == 7
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 56})


def test_a_source_without_training_state_restarts_at_rollout_0(tmp_path):
    iteration, scheduler, _ = _restore(tmp_path)
    assert iteration is None
    scheduler.load_state_dict.assert_not_called()


def test_a_weights_only_convert_save_writes_neither_moments_nor_a_placeholder_training_state(tmp_path):
    _, _, mocks = _restore(tmp_path / "source", convert_save=str(tmp_path / "converted"))
    assert mocks["DistributedOptimizerMembers"].call_args.kwargs["save_moments"] is False
    assert mocks["save_and_commit_adapter_dir"].call_args.kwargs["training_state"] is None


def test_a_source_trained_with_another_alpha_is_refused_on_the_trainer(tmp_path):
    # the driver may have resolved --lora-alpha without seeing the path the trainers load
    with pytest.raises(AssertionError, match="lora_alpha=32.0; this run uses 16.0"):
        _restore(tmp_path, source_alpha=32.0)


def test_an_old_format_save_reports_the_alpha_its_peft_pair_records(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32.0}))
    run = AdapterParametrization.__new__(AdapterParametrization)
    with patch.multiple(
        _MODULE, _native_adapter_shard=DEFAULT, _read_native_adapter_shard=DEFAULT, replace=DEFAULT
    ) as m:
        LegacyRankShardLoader([], MagicMock(), run, load_optimizer=False, precision_aware=False).parametrization(
            tmp_path
        )
    assert m["replace"].call_args.kwargs["lora_alpha"] == 32.0
    # reading the parametrization leaves the model alone: a refused source loads nothing
    m["_read_native_adapter_shard"].assert_not_called()


def _prepare_with_one_rank_missing_its_shard(rank, world_size, port, root):
    import torch.distributed as dist
    from tests.fast.dist_utils import init_gloo

    from miles.backends.training_utils.checkpoint.adapter.adapter_state import raise_if_any_rank_failed
    from miles.utils.distributed_utils import init_gloo_group

    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    lora_checkpoint._read_native_adapter_shard = lambda model, shard: []
    applied = []
    # applying runs a collective, which a rank that failed alone would never join
    optimizer = SimpleNamespace(reload_model_params=lambda: (applied.append(rank), dist.barrier()))
    loader = LegacyRankShardLoader([], optimizer, None, load_optimizer=False, precision_aware=False)
    error = None
    try:
        loader.prepare(Path(root))
    except Exception as exc:
        error = exc
    with pytest.raises(RuntimeError, match="none for global rank 1"):
        raise_if_any_rank_failed(error)
    assert applied == []
    dist.destroy_process_group()


def test_a_rank_missing_its_old_format_shard_fails_every_rank_before_the_collectives(tmp_path):
    # saved by one rank, loaded by two: rank 1 has no shard
    torch.save({}, tmp_path / "adapter_megatron_rank0.pt")
    run_multiprocess(partial(_prepare_with_one_rank_missing_its_shard, root=str(tmp_path)))


def test_an_optimizer_other_than_megatrons_distributed_one_is_refused():
    optimizer = SimpleNamespace(chained_optimizers=[MagicMock()])
    members = DistributedOptimizerMembers(MagicMock(), optimizer, [], precision_aware=False)
    with pytest.raises(AssertionError, match="--use-distributed-optimizer"):
        members._distributed_optimizers()


@pytest.mark.parametrize("leaves_out_elements", [True, False])
def test_a_restore_keeps_the_state_of_elements_no_checkpoint_holds(leaves_out_elements):
    """Native head_B's vocab-padding rows are left out of every checkpoint; a restore must not zero them. Without
    such elements every element is overwritten, so the world state is not gathered."""
    param = torch.zeros(4)
    current = {key: torch.full((4,), value) for key, value in (("param", 1.0), ("exp_avg", 2.0), ("exp_avg_sq", 3.0))}
    buffer = SimpleNamespace(
        param_index_map={param: (0, 4, 0)}, bucket_indices=[(0, 4)], param_dtype="p", grad_dtype="g"
    )
    loaded, gathered_data = {}, []

    def get_parameter_state_dp_zero(use_gloo_comm, empty_data=False):
        gathered_data.append(not empty_data)
        # Megatron's world tensors start from zeros and are filled only without empty_data
        world = {key: torch.zeros(4) if empty_data else tensor.clone() for key, tensor in current.items()}
        return {0: {("p", "g"): world}, "buckets_coalesced": True}

    sub = DistributedOptimizer.__new__(DistributedOptimizer)
    sub.init_state_fn = None
    sub.optimizer = SimpleNamespace(param_groups=[])
    sub.buffers = [buffer]
    sub.get_parameter_state_dp_zero = get_parameter_state_dp_zero
    sub.load_parameter_state_from_dp_zero = loaded.update
    slots = MagicMock(load_overwrites_every_element=not leaves_out_elements)
    slots.slot_parameters.return_value = [param]
    # element 3 is vocab padding: no piece holds it
    held = torch.tensor([0, 1, 2]) if leaves_out_elements else torch.arange(4)
    slots.piece_parameter.return_value = [(0, held)]
    members = DistributedOptimizerMembers(slots, SimpleNamespace(chained_optimizers=[sub]), [], precision_aware=False)
    rows = len(held)
    spec = LocalShardSpec("model.lm_head.lora_B", (4,), (0,), (rows,))
    state = AdapterState(
        [spec], [torch.full((rows,), 5.0)], [torch.full((rows,), 6.0)], [torch.full((rows,), 7.0)], step=9
    )
    parallel = SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0), edp=SimpleNamespace(rank=0))

    with patch.multiple(_MODULE, get_parallel_state=lambda: parallel, _build_cast_main_to_params_fn=MagicMock()):
        members.import_adapter_state(0, state)

    assert gathered_data == [leaves_out_elements]
    world = loaded[0][("p", "g")]
    kept = [1.0, 2.0, 3.0] if leaves_out_elements else [5.0, 6.0, 7.0]
    assert world["param"].tolist() == [5.0, 5.0, 5.0, kept[0]]
    assert world["exp_avg"].tolist() == [6.0, 6.0, 6.0, kept[1]]
    assert world["exp_avg_sq"].tolist() == [7.0, 7.0, 7.0, kept[2]]


def _offloaded_optimizer(*, offloaded: bool) -> tuple[HybridDeviceOptimizer, torch.nn.Parameter, torch.nn.Parameter]:
    """An HDO with one GPU-resident parameter and, if ``offloaded``, one host one; built without CUDA."""
    host, device = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    hdo = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    hdo.cpu_optimizers = [torch.optim.AdamW([host])] if offloaded else []
    # stands in for TE FusedAdam, which keeps its step in the param group
    hdo.gpu_optimizer = torch.optim.SGD([device], lr=0.1)
    hdo.defaults = {"cpu_optimizer_cls": torch.optim.AdamW}
    hdo.param_update_in_fp32 = False
    hdo.inner_param_to_orig_param = {host: host, device: device}
    hdo.state = {host: {"exp_avg": torch.zeros(2)}, device: {}} if offloaded else {device: {}}
    return hdo, host, device


def test_a_rank_whose_offloaded_shard_is_all_on_the_gpu_exports_the_gpu_optimizers_step():
    hdo, _, _ = _offloaded_optimizer(offloaded=False)
    hdo.gpu_optimizer.param_groups[0]["step"] = 3
    assert _distributed_optimizer_step(SimpleNamespace(optimizer=hdo)) == 3


def test_restoring_the_step_of_an_offloaded_optimizer_reaches_its_gpu_sub_optimizer():
    hdo, host, _ = _offloaded_optimizer(offloaded=True)
    sub = SimpleNamespace(optimizer=hdo)
    _set_distributed_optimizer_step(sub, 5)
    assert hdo.gpu_optimizer.param_groups[0]["step"] == 5
    assert hdo.cpu_optimizers[0].state[host]["step"] == hdo.state[host]["step"] == 5
    assert _distributed_optimizer_step(sub) == 5


def test_a_parameters_world_slice_skips_the_earlier_buckets_end_padding():
    """dp-zero world tensors drop each bucket's end padding; a later bucket's parameter sits after the real elements."""
    a, b, c = (torch.nn.Parameter(torch.empty(n)) for n in (4, 2, 5))
    # bucket 0 spans [0, 8) with 6 real elements; bucket 1 spans [8, 16) with 5
    buffer = SimpleNamespace(
        param_index_map={a: (0, 4, 0), b: (4, 6, 0), c: (8, 13, 1)},
        bucket_indices=[(0, 8), (8, 16)],
        buckets=[SimpleNamespace(numel=8, numel_unpadded=6), SimpleNamespace(numel=8, numel_unpadded=5)],
        param_dtype=torch.bfloat16,
        grad_dtype=torch.float32,
    )
    # hand-written: a, b, then c, with no padding between them
    world = torch.tensor([10.0, 11, 12, 13, 20, 21, 30, 31, 32, 33, 34])
    gathered = {0: {(torch.bfloat16, torch.float32): {"param": world}}}
    for param, expected in ((a, [10.0, 11, 12, 13]), (b, [20.0, 21]), (c, [30.0, 31, 32, 33, 34])):
        tensors, start, end = _world_slice(gathered, 0, buffer, param)
        assert torch.equal(tensors["param"][start:end], torch.tensor(expected))


def _recorded_parametrization(monkeypatch, hf_checkpoint, hidden_size=2560):
    config = {"model_type": "qwen3", "hidden_size": hidden_size, "num_hidden_layers": 36, "vocab_size": 151936}
    monkeypatch.setattr(lora_checkpoint, "load_hf_config", lambda path: SimpleNamespace(to_dict=lambda: config))
    slots = SimpleNamespace(
        slot_spec=lambda slot: None,
        parametrization=lambda **fields: AdapterParametrization(target_modules=("q",), **fields),
    )
    args = SimpleNamespace(
        hf_checkpoint=hf_checkpoint, lora_rank=8, lora_alpha=16.0, lora_type="lora", experts_shared_outer_loras=False
    )
    return lora_checkpoint.single_lora_parametrization(slots, args)


def test_an_adapter_saved_under_one_mount_loads_under_another(monkeypatch):
    saved = _recorded_parametrization(monkeypatch, "/root/models/Qwen3-4B")
    run = _recorded_parametrization(monkeypatch, "/lustre/models/Qwen3-4B")
    assert saved.base_model == "/root/models/Qwen3-4B", "the path stays as given, for information"
    assert_parametrization_matches_run(saved, run, [], load_optimizer=False, has_optimizer_state=False)
    other = _recorded_parametrization(monkeypatch, "/root/models/Qwen3-4B", hidden_size=4096)
    with pytest.raises(AssertionError, match="is another model than run base"):
        assert_parametrization_matches_run(saved, other, [], load_optimizer=False, has_optimizer_state=False)


_GDN = "model.layers.0.linear_attn"


class _CanonicalGdnMembers:
    """One canonical GDN component's own A and B, as native's default canonical_lora trains them."""

    bf16_adapter_key_style = "sampler"
    head_rows = None

    def get_local_shard_specs(self, slot):
        return [
            LocalShardSpec(f"{_GDN}.in_proj_q.lora_A", (2, 6), (0, 0), (2, 6)),
            LocalShardSpec(f"{_GDN}.in_proj_q.lora_B", (2, 2), (0, 0), (2, 2)),
        ]

    def export_adapter_state(self, slot):
        specs = self.get_local_shard_specs(slot)
        pieces = [torch.ones(spec.local_shape) for spec in specs]
        return AdapterState(specs, pieces, pieces, pieces, step=3)

    def export_bf16_adapter(self, slot):
        raise AssertionError("no PEFT view is written for this parametrization")


def test_a_native_canonical_gdn_save_commits_without_a_peft_view(tmp_path):
    from miles.backends.training_utils.checkpoint.adapter.commit import resolve_adapter_dir
    from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text", "hidden_size": 6}))
    targets = get_hf_lora_targets(
        {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 3,
        }
    )
    slots = SimpleNamespace(
        slot_spec=lambda slot: None,
        targets=targets,
        parametrization=lambda **fields: AdapterParametrization(target_modules=(f"{_GDN}.in_proj_q",), **fields),
    )
    args = SimpleNamespace(
        hf_checkpoint=str(hf),
        lora_rank=2,
        lora_alpha=4.0,
        lora_type="canonical_lora",
        experts_shared_outer_loras=False,
        backend=SimpleNamespace(save=str(tmp_path / "save"), use_precision_aware_optimizer=False, no_save_optim=False),
    )
    with patch(f"{_MODULE}.DistributedOptimizerMembers", return_value=_CanonicalGdnMembers()):
        lora_checkpoint.save_single_lora([], MagicMock(), None, slots, args, iteration=4)

    committed = resolve_adapter_dir(tmp_path / "save" / "iter_0000004", "adapter", tenant=None)
    assert (committed.path / "adapter_parametrization.json").is_file()
    assert not (committed.path / "adapter_model.safetensors").exists()
