"""Bridge single LoRA's --save / --lora-adapter-path glue around the adapter library."""

import json
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
import torch
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

import miles.backends.megatron_utils.lora.checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora.checkpoint import (
    DistributedOptimizerMembers,
    LegacyRankShardLoader,
    _distributed_optimizer_step,
    _set_distributed_optimizer_step,
    _world_slice,
    restore_single_lora,
)
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
        _MODULE, _native_adapter_shard=DEFAULT, _load_native_adapter_shard=DEFAULT, replace=DEFAULT
    ) as m:
        LegacyRankShardLoader([], MagicMock(), run, load_optimizer=False, precision_aware=False).parametrization(
            tmp_path
        )
    assert m["replace"].call_args.kwargs["lora_alpha"] == 32.0
    # reading the parametrization leaves the model alone: a refused source loads nothing
    m["_load_native_adapter_shard"].assert_not_called()


def test_an_optimizer_other_than_megatrons_distributed_one_is_refused():
    optimizer = SimpleNamespace(chained_optimizers=[MagicMock()])
    members = DistributedOptimizerMembers(MagicMock(), optimizer, [], precision_aware=False)
    with pytest.raises(AssertionError, match="--use-distributed-optimizer"):
        members._distributed_optimizers()


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
