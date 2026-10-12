"""Unit tests for the PP gather of a LoRA adapter in the megatron HF weight iterator:
every PP rank receives every stage's tensors, one owner stage at a time, and the
export is validated before anything is streamed."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["lora", "weight-update"])


from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight import hf_weight_iterator as hwi
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


def _stage_tensors(stage: int) -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(stage)
    return [
        (f"model.layers.{2 * stage + 1}.mlp.down_proj.lora_B.weight", torch.randn(6, 2, generator=g)),
        (f"model.layers.{2 * stage}.self_attn.o_proj.lora_A.weight", torch.randn(2, 6, generator=g)),
        (f"model.layers.{2 * stage}.self_attn.o_proj.lora_B.weight", torch.randn(6, 2, generator=g).bfloat16()),
    ]


class _FakePP:
    """A PP group of `size` stages seen from `rank`: collectives return every stage's data."""

    def __init__(self, size: int, rank: int):
        self.size, self.rank, self.group = size, rank, object()
        self.broadcasts: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stages = {s: _stage_tensors(s) for s in range(self.size)}

        def all_gather_object(out, local, group):
            for s in range(self.size):
                out[s] = local if s == self.rank else [(n, tuple(t.shape), t.dtype) for n, t in stages[s]]

        def broadcast(flat, src, group):
            self.broadcasts.append(src)
            if src == self.rank:
                return
            parts = [t.reshape(-1) for _, t in sorted(stages[src]) if t.dtype == flat.dtype]
            # entries are flattened in the owner's metadata order for that dtype
            by_name = {n: t for n, t in stages[src]}
            ordered = [by_name[n].reshape(-1) for n, _, d in self.layout[src] if d == flat.dtype]
            assert sum(p.numel() for p in parts) == flat.numel()
            flat.copy_(torch.cat(ordered))

        monkeypatch.setattr(hwi, "get_parallel_state", lambda: SimpleNamespace(pp=self))
        monkeypatch.setattr(hwi.dist, "all_gather_object", all_gather_object)
        monkeypatch.setattr(hwi.dist, "broadcast", broadcast)
        monkeypatch.setattr(hwi.dist, "get_process_group_ranks", lambda group: list(range(self.size)))
        monkeypatch.setattr(hwi.torch.cuda, "current_device", lambda: "cpu")


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_every_rank_receives_every_stage_in_owner_order(monkeypatch, rank):
    pp = _FakePP(size=3, rank=rank)
    pp.install(monkeypatch)
    local = _stage_tensors(rank)
    pp.layout = hwi._gather_pp_adapter_layout(local)

    seen_stages = []
    received = {}
    for stage in hwi._iter_pp_adapter_stages(local, pp.layout):
        names = [n for n, _ in stage]
        assert names == sorted(names), "a stage streams sorted by name"
        seen_stages.append({n.split(".")[2] for n in names})
        received.update(stage)

    expected = {n: t for s in range(3) for n, t in _stage_tensors(s)}
    assert received.keys() == expected.keys()
    for name, tensor in expected.items():
        assert received[name].dtype == tensor.dtype
        assert torch.equal(received[name], tensor)
    # one broadcast per (owner, dtype), owners in stage order
    assert pp.broadcasts == [0, 0, 1, 1, 2, 2]
    assert seen_stages == [{"0", "1"}, {"2", "3"}, {"4", "5"}]


def test_single_stage_streams_the_export_itself_in_place(monkeypatch):
    pp = _FakePP(size=1, rank=0)
    pp.install(monkeypatch)
    local = _stage_tensors(0)
    pp.layout = hwi._gather_pp_adapter_layout(local)

    (stage,) = list(hwi._iter_pp_adapter_stages(local, pp.layout))

    assert stage is local
    assert pp.broadcasts == []


class _Exporter(hwi.MegatronHfWeightIteratorBase):
    def __init__(self, export):
        self.args = Namespace()
        self.placement = WeightUpdatePlacement(gather_pp=True)
        self._export = export

    def _export_pp_local_lora(self, adapter):
        return list(self._export)

    def _iter_hf_param_units(self, weights, *, materialize):
        return iter(())


def test_adapter_units_cover_all_stages_one_tensor_per_unit(monkeypatch):
    pp = _FakePP(size=2, rank=1)
    pp.install(monkeypatch)
    pp.layout = [[(n, tuple(t.shape), t.dtype) for n, t in _stage_tensors(s)] for s in range(2)]

    units = list(_Exporter(_stage_tensors(1))._iter_hf_adapter_units(None, materialize=True))

    assert all(len(unit) == 1 for unit in units)
    assert {n for (n, _), in units} == {n for s in range(2) for n, _ in _stage_tensors(s)}


def test_non_materializing_ranks_join_every_broadcast_and_yield_nothing(monkeypatch):
    pp = _FakePP(size=2, rank=0)
    pp.install(monkeypatch)
    pp.layout = [[(n, tuple(t.shape), t.dtype) for n, t in _stage_tensors(s)] for s in range(2)]

    units = list(_Exporter(_stage_tensors(0))._iter_hf_adapter_units(None, materialize=False))

    assert units == []
    assert pp.broadcasts == [0, 0, 1, 1]


def test_an_export_without_lora_names_fails_before_anything_streams(monkeypatch):
    pp = _FakePP(size=1, rank=0)
    pp.install(monkeypatch)
    pp.layout = None

    it = _Exporter([("model.layers.0.mlp.weight", torch.zeros(2))])._iter_hf_adapter_units(None, materialize=True)

    with pytest.raises(RuntimeError, match="no lora_A/lora_B"):
        next(it)
