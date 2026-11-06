"""Bridge single LoRA's --save / --lora-adapter-path glue around the adapter library."""

import json
import multiprocessing
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from tests.fast.dist_utils import init_gloo, run_multiprocess

import miles.backends.megatron_utils.lora.checkpoint as lora_checkpoint
import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.checkpoint import (
    LegacyRankShardLoader,
    _distributed_optimizer_step,
    _set_distributed_optimizer_step,
    _world_slice,
    restore_single_lora,
)
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from miles.backends.megatron_utils.lora.utils import _adapter_shard_params
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    assert_parametrization_matches_run,
)
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError

_MODULE = "miles.backends.megatron_utils.lora.checkpoint"


def _restore(path, *, convert_save=None):
    """``restore_single_lora`` with the model-facing pieces mocked; returns (its result, scheduler, mocks)."""
    args = SimpleNamespace(
        hf_checkpoint="/hf",
        lora_adapter_path=str(path),
        lora_adapter_convert_save=convert_save,
        lora_rank=8,
        lora_alpha=16.0,
        lora_type="lora",
        lora_dropout=0.05,
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
        MegatronBridgeCheckpointLoader=DEFAULT,
        load_adapter_source=DEFAULT,
        save_and_commit_adapter_dir=DEFAULT,
        load_hf_config=lambda path: SimpleNamespace(to_dict=lambda: {"model_type": "qwen3"}),
    ) as mocks:
        mocks["load_adapter_source"].return_value = SimpleNamespace(lora_alpha=16.0)
        restored = restore_single_lora(
            [], args, adapter_slots=slots, optimizer=MagicMock(), opt_param_scheduler=scheduler
        )
    return restored, scheduler, mocks


def test_an_old_format_save_resumes_its_iteration_whichever_reader_loads_the_weights(tmp_path):
    # --no-load-optim reads an old-format save's PEFT pair, not its rank shards
    torch.save(
        {"iteration": 7, "optimizer": None, "parameter_state": None, "opt_param_scheduler": {"num_steps": 56}},
        tmp_path / "training_state_rank0.pt",
    )
    restored, scheduler, mocks = _restore(tmp_path, convert_save=str(tmp_path / "converted"))
    # it predates first_rollout: its schedule is taken to count from rollout 0, and a convert-save records that
    assert restored == (7, True, 0)
    assert mocks["save_and_commit_adapter_dir"].call_args.kwargs["training_state"]["first_rollout"] == 0
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 56})


def test_a_torchtitan_dir_reports_the_rollout_its_schedule_started_at(tmp_path):
    """Saved by a run from --start-rollout-id 5: Megatron's scheduler catch-up counts only rollouts 5..6."""
    training_state = {"iteration": 6, "step": 8, "first_rollout": 5, "lr_scheduler": {"last_epoch": 8}}
    with patch(f"{_MODULE}.read_training_state", return_value=training_state):
        restored, scheduler, _ = _restore(tmp_path)
    assert restored == (6, False, 5)
    scheduler.load_state_dict.assert_not_called()


def test_a_source_without_training_state_restarts_at_rollout_0(tmp_path):
    (iteration, _, _), scheduler, _ = _restore(tmp_path)
    assert iteration is None
    scheduler.load_state_dict.assert_not_called()


def test_a_weights_only_convert_save_writes_neither_moments_nor_a_placeholder_training_state(tmp_path):
    _, _, mocks = _restore(tmp_path / "source", convert_save=str(tmp_path / "converted"))
    assert mocks["DistributedOptimizerMembers"].call_args.kwargs["save_moments"] is False
    assert mocks["save_and_commit_adapter_dir"].call_args.kwargs["training_state"] is None
    assert mocks["save_and_commit_adapter_dir"].call_args.kwargs["lora_dropout"] == 0.05


def test_an_old_format_save_reports_the_alpha_its_peft_pair_records(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32.0}))
    run = AdapterParametrization(
        base_model="/hf",
        lora_rank=2,
        lora_alpha=16.0,
        lora_type="lora",
        target_modules=("model.layers.0.self_attn.q_proj",),
        experts_shared_outer_loras=False,
    )
    with patch.multiple(_MODULE, _native_adapter_shard=DEFAULT, _read_native_adapter_shard=DEFAULT) as m:
        loaded, has_optimizer_state = LegacyRankShardLoader(
            [], MagicMock(), run, load_optimizer=False, precision_aware=False
        ).parametrization(tmp_path)
    assert loaded.lora_alpha == 32.0 and not has_optimizer_state
    # reading the parametrization leaves the model alone: a refused source loads nothing
    m["_read_native_adapter_shard"].assert_not_called()


@pytest.mark.parametrize(
    "training_state, has_optimizer_state",
    [
        ({"optimizer": {"step": 7}, "parameter_state": []}, True),
        ({"optimizer": None, "parameter_state": None}, False),
        ({"optimizer": {"step": 7}}, False),
    ],
    ids=["full", "saved-with-no-save-optim", "pre-parameter-state-format"],
)
def test_an_old_format_save_reports_whether_it_holds_optimizer_state(tmp_path, training_state, has_optimizer_state):
    """load_adapter_source refuses a source without it unless --no-load-optim."""
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32.0}))
    state = {"iteration": 5, "opt_param_scheduler": None} | training_state
    torch.save(state, tmp_path / "training_state_rank0.pt")
    run = AdapterParametrization("/hf", 2, 16.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
    loader = LegacyRankShardLoader([], MagicMock(), run, load_optimizer=True, precision_aware=False)
    assert loader.parametrization(tmp_path)[1] == has_optimizer_state


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
    with pytest.raises(AdapterSourceError, match="is another model than run base"):
        assert_parametrization_matches_run(saved, other, [], load_optimizer=False, has_optimizer_state=False)


RUN = AdapterParametrization("m", 1, 2.0, "lora", ("model.layers.0.self_attn.q_proj",), False)


class _Adapter(NativeLoRAAdapter):
    def __init__(self):
        super().__init__("model.layers.0.self_attn.q_proj.", (), tp_rank=0)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))

    def export_plan(self, gather):
        return []


class _Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.adapter = _Adapter()
        self.param_syncs = 0

    def start_param_sync(self, force_sync):
        self.param_syncs += 1


def _save_old_format(path, chunks, **training_state):
    """The old-format files a rank wrote: its adapter shard and, optionally, its training state."""
    torch.save(
        {name: param.detach().cpu() for name, param in _adapter_shard_params(chunks).items()},
        path / "adapter_megatron_rank0.pt",
    )
    if training_state:
        state = {"iteration": 5, "optimizer": {"step": 7}, "opt_param_scheduler": {"lr": 0.5}} | training_state
        torch.save(state, path / "training_state_rank0.pt")


def _random_chunks(n):
    chunks = [_Chunk() for _ in range(n)]
    for chunk in chunks:
        for param in chunk.adapter.parameters():
            param.data.copy_(torch.randn_like(param))
    return chunks


def _optimizer(calls):
    return SimpleNamespace(
        load_state_dict=lambda state: calls.append(("load_state_dict", state)),
        reload_model_params=lambda: calls.append(("reload_model_params", None)),
    )


def _load(path, chunks, calls, *, load_optimizer):
    loader = LegacyRankShardLoader(chunks, _optimizer(calls), RUN, load_optimizer, precision_aware=False)
    parametrization, has_optimizer_state = loader.parametrization(path)
    # load_adapter_source's check between reading the parametrization and preparing
    assert_parametrization_matches_run(
        parametrization, RUN, [], load_optimizer=load_optimizer, has_optimizer_state=has_optimizer_state
    )
    loader.apply(loader.prepare(path))
    return parametrization


class TestLegacyRankShardLoader:
    """LegacyRankShardLoader reads a single-LoRA ``--save`` written before the adapter format (adapter_megatron_rank*.pt) into the model."""

    @pytest.fixture(autouse=True)
    def _single_rank(self, monkeypatch):
        monkeypatch.setattr(
            lora_utils,
            "get_parallel_state",
            lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
        )

    def test_virtual_pipeline_chunks_restore_their_own_adapters_bitwise(self, tmp_path):
        saved = _random_chunks(2)
        _save_old_format(tmp_path, saved)
        restored, calls = [_Chunk(), _Chunk()], []

        # the old save records no alpha; load_adapter_source falls back to the run's
        assert _load(tmp_path, restored, calls, load_optimizer=False) == replace(RUN, lora_alpha=None)

        for before, after in zip(saved, restored, strict=True):
            for a, b in zip(before.adapter.parameters(), after.adapter.parameters(), strict=True):
                assert torch.equal(a, b)
        # the fp32 masters were built before the adapter load; without the refresh the first step reverts it
        assert calls == [("reload_model_params", None)]

    def test_a_full_checkpoint_hands_the_saved_optimizer_and_parameter_state_over_unchanged(
        self, tmp_path, monkeypatch
    ):
        """The old path loaded optimizer.state_dict() and each DistributedOptimizer's reshardable parameter state."""
        calls, casts = [], []
        parameter_state = {"per_bucket_numel": {}, 0: {torch.float32: [[{"param": torch.randn(2), "padding": False}]]}}
        distributed = SimpleNamespace(
            optimizer=object(),
            load_parameter_state_from_dp_reshardable=lambda state: calls.append(("parameter", state)),
        )
        monkeypatch.setattr(lora_checkpoint, "_distributed_optimizers", lambda _optimizer: [distributed])
        monkeypatch.setattr(
            lora_checkpoint,
            "_build_cast_main_to_params_fn",
            lambda _optimizer, precision_aware: lambda: casts.append(1),
        )
        saved = _random_chunks(1)
        _save_old_format(tmp_path, saved, parameter_state=[parameter_state])
        restored = [_Chunk()]

        _load(tmp_path, restored, calls, load_optimizer=True)

        assert [name for name, _ in calls] == ["load_state_dict", "parameter"]
        assert calls[0][1] == {"step": 7}
        assert torch.equal(
            calls[1][1][0][torch.float32][0][0]["param"], parameter_state[0][torch.float32][0][0]["param"]
        )
        # the loaded masters, not the shard's bf16 weights, become the model's parameters on every rank
        assert casts == [1] and restored[0].param_syncs == 1
        for a, b in zip(saved[0].adapter.parameters(), restored[0].adapter.parameters(), strict=True):
            assert torch.equal(a, b)

    @pytest.mark.parametrize(
        "training_state",
        [{}, {"optimizer": None, "parameter_state": None}, {"optimizer": {"step": 7}}],
        ids=["no-training-state", "saved-with-no-save-optim", "pre-parameter-state-format"],
    )
    def test_an_unrestorable_optimizer_fails_unless_no_load_optim(self, tmp_path, monkeypatch, training_state):
        monkeypatch.setattr(lora_checkpoint, "_distributed_optimizers", lambda _optimizer: [])
        _save_old_format(tmp_path, _random_chunks(1), **training_state)

        with pytest.raises(AdapterSourceError, match=r"no optimizer state; load weights only \(--no-load-optim\)"):
            _load(tmp_path, [_Chunk()], [], load_optimizer=True)
        _load(tmp_path, [_Chunk()], [], load_optimizer=False)

    def test_a_shard_that_does_not_match_the_adapter_fails(self, tmp_path):
        """A silently partial load resumes training on a half-initialized adapter."""
        torch.save(
            {"adapter.lora_A": torch.ones(1, 2), "stale_lora_C": torch.ones(1)}, tmp_path / "adapter_megatron_rank0.pt"
        )

        with pytest.raises(RuntimeError, match=r"missing=\['adapter.lora_B'\], unexpected=\['stale_lora_C'\]"):
            _load(tmp_path, [_Chunk()], [], load_optimizer=False)

    def test_a_shard_tensor_of_another_shape_fails(self, tmp_path):
        """copy_ would broadcast a shared-outer [1, ...] into a per-expert [E, ...] parameter without a word."""
        torch.save(
            {"adapter.lora_A": torch.ones(1, 2), "adapter.lora_B": torch.ones(1, 1)},
            tmp_path / "adapter_megatron_rank0.pt",
        )

        with pytest.raises(RuntimeError, match=r"adapter.lora_B: saved \(1, 1\) != parameter \(2, 1\)"):
            _load(tmp_path, [_Chunk()], [], load_optimizer=False)

    def test_shards_saved_under_another_layout_fail(self, tmp_path):
        """Falling through to fresh adapter weights would hide a resharding mistake."""
        torch.save({"adapter.lora_A": torch.ones(1, 2)}, tmp_path / "adapter_megatron_rank1.pt")

        with pytest.raises(AssertionError, match="none for global rank 0"):
            _load(tmp_path, [_Chunk()], [], load_optimizer=False)

    def test_a_directory_without_a_shard_for_this_rank_fails(self, tmp_path):
        with pytest.raises(AssertionError, match="no adapter shard for this rank"):
            _load(tmp_path, [_Chunk()], [], load_optimizer=False)


class _DpZeroOptimizer(DistributedOptimizer):
    """A DistributedOptimizer's dp-zero gather and scatter over one parameter, on the gloo group Megatron gives it."""

    def __init__(self, param, master, group):
        self.group, self.init_state_fn, self.optimizer = group, None, torch.optim.Adam([torch.zeros(1)])
        self.world = {
            "param": master.flatten().clone(),
            "exp_avg": master.flatten() + 1,
            "exp_avg_sq": master.flatten() + 2,
        }
        n = param.numel()
        self.buffers = [
            SimpleNamespace(
                param_index_map={param: (0, n, 0)},
                bucket_indices=[(0, n)],
                buckets=[SimpleNamespace(numel_unpadded=n)],
                param_dtype=torch.bfloat16,
                grad_dtype=torch.float32,
            )
        ]

    def get_parameter_state_dp_zero(self, use_gloo_comm, empty_data=False):
        assert use_gloo_comm
        if self.group.rank() != 0:
            return None
        world = {k: torch.empty_like(v) if empty_data else v.clone() for k, v in self.world.items()}
        return {0: {(torch.bfloat16, torch.float32): world}, "buckets_coalesced": None}

    def load_parameter_state_from_dp_zero(self, gathered):
        leader = dist.get_global_rank(self.group, 0)
        for key, tensor in self.world.items():
            if gathered is not None:
                tensor.copy_(gathered[0][(torch.bfloat16, torch.float32)][key])
            dist.broadcast(tensor, src=leader, group=self.group)


def _expert_dp_round_trip(rank, world_size, port):
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    from miles.backends.megatron_utils import parallel as megatron_parallel
    from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state

    init_gloo(rank, world_size, port=port)
    # dense DP is 4 (one leader), expert DP is 2 (a leader per expert-parallel rank)
    parallel_state.initialize_model_parallel(expert_model_parallel_size=2)
    megatron_parallel.get_args = lambda: SimpleNamespace(cp_comm_type=None)
    set_parallel_state(megatron_parallel.create_megatron_parallel_state(indep_dp=None))

    # dense values are the same on every rank, expert values the same within an expert DP group
    masters = [
        torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16().float()
        for shape, seed in (((2, 3), 0), ((4, 3), 1 + get_parallel_state().ep.rank))
    ]
    params = [torch.nn.Parameter(m.bfloat16()) for m in masters]
    params[1].allreduce = False
    # the gloo groups Megatron's optimizer setup hands the dense and the expert DistributedOptimizer
    optimizer_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(None, [SimpleNamespace()])
    groups = [optimizer_groups["intra_dp_cp_group_gloo"], optimizer_groups["intra_expt_dp_group_gloo"]]
    subs = [_DpZeroOptimizer(p, m, g) for p, m, g in zip(params, masters, groups, strict=True)]
    slots = SimpleNamespace(
        bf16_adapter_key_style="peft",
        slot_parameters=lambda slot: params,
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=params[i].shape) for i in sorted(holds)],
        piece_parameter=lambda slot, holds: [(i, torch.arange(params[i].numel())) for i in sorted(holds)],
    )

    def cast_masters_to_params():
        for param, sub in zip(params, subs, strict=True):
            param.data.copy_(sub.world["param"].view(param.shape))

    lora_checkpoint._build_cast_main_to_params_fn = lambda optimizer, precision_aware: cast_masters_to_params
    members = lora_checkpoint.DistributedOptimizerMembers(
        slots, SimpleNamespace(chained_optimizers=subs), [], precision_aware=False
    )
    state = members.export_adapter_state(0)
    expected = [{k: v.clone() for k, v in sub.world.items()} for sub in subs]
    for sub in subs:
        for tensor in sub.world.values():
            tensor.zero_()
    members.import_adapter_state(0, state)
    members.finish_import_adapter_state(0)
    for sub, saved in zip(subs, expected, strict=True):
        assert all(torch.equal(sub.world[k], saved[k]) for k in saved)
    assert all(torch.equal(p.float(), m) for p, m in zip(params, masters, strict=True))
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_each_distributed_optimizers_gloo_leader_exports_and_imports_its_parameters():
    """Dense and expert parameters have different DP groups, so different leaders."""
    process = multiprocessing.get_context("spawn").Process(target=run_multiprocess, args=(_expert_dp_round_trip, 4))
    process.start()
    process.join(timeout=120)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


class _LedElsewhere(torch.Tensor):
    """A parameter another rank leads: exporting this rank's pieces must never read or copy it."""

    def detach(self):
        raise AssertionError("a parameter this rank does not lead was read")

    def float(self):
        raise AssertionError("a parameter this rank does not lead was copied")


def test_an_export_reads_only_the_parameters_this_rank_leads(monkeypatch):
    master = torch.tensor([[1.0, 2.0]])
    led, elsewhere = torch.nn.Parameter(master.bfloat16()), torch.zeros(3, 4).as_subclass(_LedElsewhere)
    elsewhere.allreduce = False
    # this rank leads its dense DP group but not its expert DP group
    monkeypatch.setattr(
        lora_checkpoint,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0), edp=SimpleNamespace(rank=1)),
    )
    dense = _DpZeroOptimizer(led, master, SimpleNamespace(rank=lambda: 0))
    expert = _DpZeroOptimizer(elsewhere, torch.zeros(3, 4), SimpleNamespace(rank=lambda: 1))
    params = [led, elsewhere]
    slots = SimpleNamespace(
        bf16_adapter_key_style="peft",
        slot_parameters=lambda slot: params,
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=led.shape)],
        piece_parameter=lambda slot, holds: [(0, torch.arange(2))],
    )
    members = lora_checkpoint.DistributedOptimizerMembers(
        slots, SimpleNamespace(chained_optimizers=[dense, expert]), [], precision_aware=False
    )
    state = members.export_adapter_state(0)
    assert torch.equal(state.weights[0], master) and torch.equal(state.exp_avgs[0], master + 1)


def test_an_export_of_masters_holding_nan_saves_them(monkeypatch):
    master = torch.tensor([[1.0, float("nan")]])
    param = torch.nn.Parameter(master.bfloat16())
    monkeypatch.setattr(
        lora_checkpoint,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0), edp=SimpleNamespace(rank=0)),
    )
    dense = _DpZeroOptimizer(param, master, SimpleNamespace(rank=lambda: 0))
    slots = SimpleNamespace(
        bf16_adapter_key_style="peft",
        slot_parameters=lambda slot: [param],
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=param.shape)],
        piece_parameter=lambda slot, holds: [(0, torch.arange(2))],
    )
    members = lora_checkpoint.DistributedOptimizerMembers(
        slots, SimpleNamespace(chained_optimizers=[dense]), [], precision_aware=False
    )
    state = members.export_adapter_state(0)
    assert torch.equal(state.weights[0].isnan(), master.isnan())


def test_without_an_optimizer_an_adapter_path_loads_the_weights_and_starts_over(tmp_path, monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    slots = SimpleNamespace(
        bf16_adapter_key_style="peft",
        targets=None,
        slot_parameters=lambda slot: [param],
        piece_parameter=lambda slot, holds: [(0, torch.tensor([3, 0]))],
    )

    loaded_with = {}

    def load_adapter_source(path, members, slot, run, targets, *, in_model_loaders_by_kind, **kwargs):
        loaded_with.update(kwargs)
        # a Bridge checkpoint's load refreshes the optimizer's masters; without one it is refused as unloadable
        assert AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT not in in_model_loaders_by_kind
        members.import_adapter_state(
            0,
            AdapterState(
                [SimpleNamespace(local_shape=(2,))],
                [torch.tensor([5.0, 7.0])],
                [],
                [],
                step=None,
                has_optimizer_state=False,
            ),
        )
        members.finish_import_adapter_state(0)
        return SimpleNamespace(lora_alpha=16.0)

    monkeypatch.setattr(lora_checkpoint, "single_lora_parametrization", lambda *_: SimpleNamespace(lora_alpha=16.0))
    monkeypatch.setattr(lora_checkpoint, "load_adapter_source", load_adapter_source)
    args = SimpleNamespace(
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save=None,
        megatron_to_hf_mode="bridge",
        backend=SimpleNamespace(no_load_optim=False, no_save_optim=True, use_precision_aware_optimizer=False),
    )
    restored = restore_single_lora([], args, adapter_slots=slots, optimizer=None, opt_param_scheduler=None)
    assert restored == (None, False, 0)
    assert loaded_with["load_optimizer"] is False
    assert torch.equal(param, torch.tensor([[7.0, 0.0], [0.0, 5.0]], dtype=torch.bfloat16))


def test_without_an_optimizer_an_old_format_save_loads_the_weights_only():
    param = torch.nn.Parameter(torch.zeros(2))
    loader = LegacyRankShardLoader([], None, None, load_optimizer=False, precision_aware=False)
    loader.apply(([(param, torch.ones(2))], None))
    assert torch.equal(param, torch.ones(2))
