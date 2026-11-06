"""``save_adapter_dir`` -> commit -> ``load_adapter_source`` restores weights, moments and step under a new split."""

import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.checkpoint.adapter import adapter_dir
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    HeadRows,
    load_adapter_source,
    save_adapter_dir,
    save_and_commit_adapter_dir,
    save_external_export,
    write_peft_export,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec
from miles.backends.training_utils.checkpoint.adapter.commit import (
    commit_adapter_dir,
    resolve_adapter_dir,
    tenant_digest,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind, ExternalWeightsReader
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

MODULE = "model.layers.0.self_attn.q_proj"
RUN = AdapterParametrization("m", 2, 4.0, "canonical_lora", (MODULE,), False)
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
FULL = {
    "lora_A": torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6),
    "lora_B": torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
}


class FakeMembers:
    """B split by rows across ranks (column-parallel), A replicated with rank 0 primary."""

    head_rows = None

    def __init__(self, scale: float = 1.0):
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        self.scale, self.imported, self.finished = scale, None, False

    def get_local_shard_specs(self, slot):
        rows = 4 // self.world
        return [
            LocalShardSpec(f"{MODULE}.lora_A", (2, 6), (0, 0), (2, 6), is_primary=self.rank == 0),
            LocalShardSpec(f"{MODULE}.lora_B", (4, 2), (self.rank * rows, 0), (rows, 2)),
        ]

    def export_adapter_state(self, slot):
        specs = self.get_local_shard_specs(slot)
        pieces = [
            FULL[s.key.rsplit(".", 1)[1]][s.offsets[0] : s.offsets[0] + s.local_shape[0]] * self.scale for s in specs
        ]
        return AdapterState(specs, pieces, [p * 2 for p in pieces], [p * 3 for p in pieces], step=5)

    def import_adapter_state(self, slot, state):
        self.imported = state

    def finish_import_adapter_state(self, slot):
        assert self.imported is not None, "finish runs after the import"
        self.finished = True

    def export_bf16_adapter(self, slot):
        raise NotImplementedError


def _save(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    written = save_adapter_dir(
        Path(root), "w", FakeMembers(), 0, RUN, tenant_digest=tenant_digest("alice"), training_state={"iteration": 3}
    )
    if rank == 0:
        commit_adapter_dir(Path(root), "w", written, overwrite=False)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def committed_dir():
    """One 2-rank save shared by the tests that load it under one process."""
    root = tempfile.mkdtemp()
    run_multiprocess(partial(_save, root=root))
    return resolve_adapter_dir(Path(root), "w", tenant="alice").path


def test_round_trip_with_optimizer_under_a_different_world_size(committed_dir):
    members = FakeMembers()  # one process now owns everything
    source = load_adapter_source(committed_dir, members, 0, RUN, TARGETS, load_optimizer=True)
    assert source == RUN and members.imported.step == 5
    for spec, weight, exp_avg in zip(
        members.imported.shard_specs, members.imported.weights, members.imported.exp_avgs, strict=True
    ):
        assert torch.equal(weight, FULL[spec.key.rsplit(".", 1)[1]])
        assert torch.equal(exp_avg, weight * 2)
    assert (committed_dir / "training_state.json").read_text().count('"iteration": 3') == 1


def test_weights_only_from_peft_upcasts_zeroes_moments_and_returns_the_source_alpha(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS, lora_dropout=0.0)
    members = FakeMembers()
    # a Tinker session adopts the source's alpha after loading
    source = load_adapter_source(tmp_path, members, 0, replace(RUN, lora_alpha=16.0), TARGETS, load_optimizer=False)
    assert source.lora_alpha == 4.0 and members.finished
    state = members.imported
    assert state.step == 0 and all(not m.any() for m in state.exp_avgs)
    assert all(w.dtype == torch.float32 for w in state.weights)
    assert torch.equal(state.weights[1], FULL["lora_B"].to(torch.bfloat16).float())


def test_weights_only_from_the_same_parametrization_reads_only_local_pieces(committed_dir, monkeypatch):
    def whole_tensor_read(*args, **kwargs):
        raise AssertionError("a same-parametrization load must not read whole tensors")

    monkeypatch.setattr(
        "miles.backends.training_utils.checkpoint.adapter.readers.MilesAdapterDirReader.read_weights",
        whole_tensor_read,
    )
    members = FakeMembers()
    load_adapter_source(committed_dir, members, 0, RUN, TARGETS, load_optimizer=False)
    assert members.imported.step == 0 and not members.imported.exp_avgs[0].any()
    assert torch.equal(members.imported.weights[1], FULL["lora_B"])


def test_a_same_parametrization_load_reads_fp32_under_a_bf16_default_dtype(committed_dir):
    members = FakeMembers()
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        load_adapter_source(committed_dir, members, 0, RUN, TARGETS, load_optimizer=True)
    finally:
        torch.set_default_dtype(default_dtype)
    assert torch.equal(members.imported.exp_avgs[1], FULL["lora_B"] * 2)


class InModelLoaderStub:
    """An in-model source with a fixed parametrization; records what it loads."""

    def __init__(self, parametrization, has_optimizer_state=True):
        self._parametrization, self._has_optimizer_state, self.loaded = parametrization, has_optimizer_state, []

    def parametrization(self, path):
        return self._parametrization, self._has_optimizer_state

    def prepare(self, path):
        return path

    def apply(self, prepared):
        self.loaded.append(prepared)


def _torchtitan_checkpoint(path) -> Path:
    torch.distributed.checkpoint.save(
        {"model.layers.0.attention.wq.lora_a.weight": torch.zeros(2)}, checkpoint_id=path, no_dist=True
    )
    return Path(path)


def _load_in_model(path, loader, run=RUN, *, load_optimizer):
    return load_adapter_source(
        path,
        FakeMembers(),
        0,
        run,
        TARGETS,
        load_optimizer=load_optimizer,
        in_model_loaders_by_kind={AdapterSourceKind.TORCHTITAN_CHECKPOINT: loader},
    )


_SHARED_K = replace(RUN, target_modules=(MODULE, "m.k_proj"), shared_lora_a_groups=((MODULE, "m.k_proj"),))


@pytest.mark.parametrize(
    "source, run, has_optimizer_state, load_optimizer, refusal",
    [
        (_SHARED_K, replace(_SHARED_K, shared_lora_a_groups=()), True, False, "cannot do"),
        (replace(RUN, lora_rank=RUN.lora_rank + 1), RUN, True, False, "adapter rank"),
        (RUN, RUN, False, True, "no optimizer state"),
    ],
    ids=["needs_a_conversion", "another_rank", "no_optimizer_state"],
)
def test_a_refused_in_model_source_never_touches_the_model(
    tmp_path, source, run, has_optimizer_state, load_optimizer, refusal
):
    loader = InModelLoaderStub(source, has_optimizer_state)
    with pytest.raises(RuntimeError, match=refusal):
        _load_in_model(_torchtitan_checkpoint(tmp_path), loader, run, load_optimizer=load_optimizer)
    assert loader.loaded == []


def test_an_accepted_in_model_source_without_alpha_loads_under_the_runs(tmp_path):
    loader = InModelLoaderStub(replace(RUN, lora_alpha=None), has_optimizer_state=False)
    source = _load_in_model(_torchtitan_checkpoint(tmp_path), loader, load_optimizer=False)
    assert loader.loaded == [tmp_path] and source.lora_alpha == RUN.lora_alpha


class WeightsOnlyMembers(FakeMembers):
    """A run saved with --no-save-optim."""

    def export_adapter_state(self, slot):
        state = super().export_adapter_state(slot)
        return AdapterState(state.shard_specs, state.weights, [], [], step=5, has_optimizer_state=False)


def test_a_weights_only_dir_loads_without_optimizer_and_refuses_with_it(tmp_path):
    written = save_adapter_dir(tmp_path, "w", WeightsOnlyMembers(), 0, RUN, tenant_digest=None)
    members = FakeMembers()
    load_adapter_source(written.path, members, 0, RUN, TARGETS, load_optimizer=False)
    assert torch.equal(members.imported.weights[1], FULL["lora_B"]) and members.imported.step == 0
    with pytest.raises(RuntimeError, match="no optimizer state"):
        load_adapter_source(written.path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=True)


class VocabPaddingMembers(FakeMembers):
    """Stores the rows past the vocabulary its head's B is padded to, as native Megatron does."""

    def __init__(self, padding_rows: int = 2):
        super().__init__()
        self.padding = LocalShardSpec(f"{MODULE}.lora_B.vocab_padding", (padding_rows, 2), (0, 0), (padding_rows, 2))

    def get_local_shard_specs(self, slot):
        return [*super().get_local_shard_specs(slot), self.padding]

    def export_adapter_state(self, slot):
        state = FakeMembers().export_adapter_state(slot)
        rows = torch.full(self.padding.local_shape, 7.0)
        return AdapterState(
            [*state.shard_specs, self.padding],
            [*state.weights, rows],
            [*state.exp_avgs, rows * 2],
            [*state.exp_avg_sqs, rows * 3],
            step=5,
        )


def test_vocab_padding_resumes_exactly_and_is_zero_for_any_other_run(tmp_path):
    written = save_adapter_dir(tmp_path, "w", VocabPaddingMembers(), 0, RUN, tenant_digest=None)
    resumed = VocabPaddingMembers()
    load_adapter_source(written.path, resumed, 0, RUN, TARGETS, load_optimizer=True)
    assert torch.equal(resumed.imported.weights[-1], torch.full((2, 2), 7.0))
    assert torch.equal(resumed.imported.exp_avgs[-1], torch.full((2, 2), 14.0))
    # a run padding to another size starts its padding at zero; one without padding ignores the source's
    repadded = VocabPaddingMembers(padding_rows=3)
    load_adapter_source(written.path, repadded, 0, RUN, TARGETS, load_optimizer=True)
    assert not repadded.imported.weights[-1].any() and not repadded.imported.exp_avgs[-1].any()
    unpadded = FakeMembers()
    load_adapter_source(written.path, unpadded, 0, RUN, TARGETS, load_optimizer=True)
    assert len(unpadded.imported.weights) == 2


class ExtraKeyMembers(FakeMembers):
    """Writes one tensor the run does not train, e.g. a second copy of a shared A."""

    def export_adapter_state(self, slot):
        state = super().export_adapter_state(slot)
        extra = LocalShardSpec("model.layers.0.self_attn.k_proj.lora_A", (2, 6), (0, 0), (2, 6))
        w = FULL["lora_A"].clone()
        return AdapterState(
            [*state.shard_specs, extra], [*state.weights, w], [*state.exp_avgs, w], [*state.exp_avg_sqs, w], step=5
        )


class NoPrimaryAMembers(FakeMembers):
    """Every rank marks lora_A a replica, so no rank writes it."""

    def get_local_shard_specs(self, slot):
        a, b = super().get_local_shard_specs(slot)
        return [replace(a, is_primary=False), b]


class PaddedRankMembers(FakeMembers):
    """A multi-LoRA backend that forgot to cut a max-rank slot to its real rank."""

    def get_local_shard_specs(self, slot):
        a, b = super().get_local_shard_specs(slot)
        return [a, replace(b, global_shape=(4, 3), local_shape=(4 // self.world, 3))]

    def export_adapter_state(self, slot):
        specs = self.get_local_shard_specs(slot)
        pieces = [FULL["lora_A"].clone(), torch.zeros(specs[1].local_shape)]
        return AdapterState(specs, pieces, pieces, pieces, step=1)


@pytest.mark.parametrize(
    "members, refusal",
    [
        (ExtraKeyMembers, "unexpected .*k_proj"),
        (NoPrimaryAMembers, "missing .*lora_A"),
        (PaddedRankMembers, "rank 3 != 2"),
    ],
    ids=["extra_key", "no_primary", "padded_rank"],
)
def test_a_backend_state_that_does_not_match_the_parametrization_is_refused_at_save(tmp_path, members, refusal):
    with pytest.raises(RuntimeError, match=refusal):
        save_adapter_dir(tmp_path, "w", members(), 0, RUN, tenant_digest=None)


def test_a_refused_source_is_reported_as_a_source_error(tmp_path):
    written = save_adapter_dir(tmp_path, "w", FakeMembers(), 0, RUN, tenant_digest=None)
    with pytest.raises(RuntimeError, match="adapter rank 2 != run rank 4") as refused:
        load_adapter_source(written.path, FakeMembers(), 0, replace(RUN, lora_rank=4), TARGETS, load_optimizer=False)
    assert isinstance(refused.value.__cause__, AdapterSourceError)
    with pytest.raises(RuntimeError) as unrecognized:
        load_adapter_source(tmp_path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=False)
    assert isinstance(unrecognized.value.__cause__, AdapterSourceError)


class OneRankCannotImport(FakeMembers):
    """Rank 1 fails writing its pieces; finishing the import on any rank would enter a collective."""

    def import_adapter_state(self, slot, state):
        assert dist.get_rank() != 1, "rank 1 cannot write its pieces"
        super().import_adapter_state(slot, state)

    def finish_import_adapter_state(self, slot):
        dist.barrier()
        super().finish_import_adapter_state(slot)


class OneRankCannotPrepare(InModelLoaderStub):
    """Rank 1 fails reading its part of the source; applying it on any rank would enter a collective."""

    def prepare(self, path):
        assert dist.get_rank() != 1, "rank 1 has no shard"
        return path

    def apply(self, prepared):
        dist.barrier()
        super().apply(prepared)


class SpecsFailOnRank1(FakeMembers):
    def get_local_shard_specs(self, slot):
        assert self.rank != 1, "rank 1 cannot describe its pieces"
        return super().get_local_shard_specs(slot)


def _one_rank_fails(rank, world_size, port, *, peft, torchtitan, scratch):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    rank = dist.get_rank()
    opened = []

    class CountingReader(adapter_dir.PeftAdapterReader):
        def __init__(self, *args, **kwargs):
            opened.append(rank)
            super().__init__(*args, **kwargs)

    adapter_dir.PeftAdapterReader = CountingReader
    members = FakeMembers()
    load_adapter_source(Path(peft), members, 0, RUN, TARGETS, load_optimizer=False)
    assert opened == ([0] if rank == 0 else []), "a source that needs conversion is read on rank 0 only"
    expected = [FULL["lora_A"], FULL["lora_B"][rank * 2 : rank * 2 + 2]]
    for got, want in zip(members.imported.weights, expected, strict=True):
        assert torch.equal(got, want.to(torch.bfloat16).float())
    # a file where the root directory should be makes rank 0's mkdir fail
    if rank == 0:
        Path(scratch, "blocker").write_text("")
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 0"):
        save_adapter_dir(
            Path(scratch, "blocker" if rank == 0 else "ok"), "w", FakeMembers(), 0, RUN, tenant_digest=None
        )

    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1"):
        load_adapter_source(
            Path(peft, "missing") if rank else Path(peft), FakeMembers(), 0, RUN, TARGETS, load_optimizer=False
        )

    with pytest.raises(RuntimeError, match="rank 1 cannot describe"):
        load_adapter_source(Path(peft), SpecsFailOnRank1(), 0, RUN, TARGETS, load_optimizer=False)

    members = OneRankCannotImport()
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: rank 1 cannot write"):
        load_adapter_source(Path(peft), members, 0, RUN, TARGETS, load_optimizer=False)
    assert not members.finished, "no rank finishes an import another rank could not write"

    loader = OneRankCannotPrepare(RUN)
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: rank 1 has no shard"):
        _load_in_model(Path(torchtitan), loader, load_optimizer=False)
    assert loader.loaded == [], "no rank applies a source another rank could not prepare"
    dist.destroy_process_group()


def test_a_rank_local_failure_fails_every_rank_before_the_next_collective(tmp_path):
    """A converted source is read on rank 0 only and scattered; save prepare, source read, shard specs, import and
    in-model prepare each agree before a collective."""
    peft = tmp_path / "peft"
    write_peft_adapter(peft, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS, lora_dropout=0.0)
    torchtitan = _torchtitan_checkpoint(tmp_path / "torchtitan")
    (tmp_path / "scratch").mkdir()
    worker = partial(_one_rank_fails, peft=str(peft), torchtitan=str(torchtitan), scratch=str(tmp_path / "scratch"))
    process = __import__("multiprocessing").get_context("spawn").Process(target=run_multiprocess, args=(worker,))
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


class DataParallelMembers(FakeMembers):
    """Two data-parallel replicas of a two-way row split: ranks 2 and 3 hold non-primary copies of ranks 0 and 1."""

    def get_local_shard_specs(self, slot):
        replica, rows = self.rank // 2, 2
        return [
            LocalShardSpec(f"{MODULE}.lora_A", (2, 6), (0, 0), (2, 6), is_primary=self.rank == 0),
            LocalShardSpec(f"{MODULE}.lora_B", (4, 2), (self.rank % 2 * rows, 0), (rows, 2), is_primary=replica == 0),
        ]


def _export_external_with_replicas(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    written = save_external_export(
        Path(root), "x", DataParallelMembers(scale=1 / 3), 0, RUN, TARGETS, tenant_digest=None, lora_dropout=0.0
    )
    if rank == 0:
        commit_adapter_dir(Path(root), "x", written, overwrite=False)
    dist.destroy_process_group()


def test_the_external_export_is_the_fp32_masters_each_from_its_primary_rank():
    """Not the bf16 weights upcast, and no replica's copy counted twice."""
    root = tempfile.mkdtemp()
    run_multiprocess(partial(_export_external_with_replicas, root=root), world_size=4)
    committed = resolve_adapter_dir(Path(root), "x", tenant=None)
    exported = ExternalWeightsReader(committed.path, RUN, TARGETS).read_weights()
    for factor, full in FULL.items():
        master = full * (1 / 3)
        assert not torch.equal(master, master.bfloat16().float()), "the masters must not be bf16-exact"
        assert torch.equal(exported[f"{MODULE}.{factor}"], master)


GDN = "model.layers.0.linear_attn"
GDN_TARGETS = get_hf_lora_targets(
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
GDN_ROWS = {"in_proj_qkv": 7, "in_proj_z": 3, "in_proj_b": 1, "in_proj_a": 1}


class GdnBridgeMembers:
    """Bridge's PEFT-named export: one A under every HF module of the group, in_proj_qkv fused."""

    bf16_adapter_key_style = "peft"

    def __init__(self, modules):
        self.a = torch.randn(4, 6)
        self.b = {m: torch.randn(GDN_ROWS[m], 4) for m in modules}

    def export_bf16_adapter(self, slot):
        for module, b in self.b.items():
            yield f"{GDN}.{module}.lora_A.weight", self.a.clone()
            yield f"{GDN}.{module}.lora_B.weight", b.clone()


@pytest.mark.parametrize(
    "modules", [("in_proj_qkv", "in_proj_z"), ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
)
def test_peft_export_of_a_fused_gdn_group_writes_one_a_per_exported_module(tmp_path, modules):
    from safetensors.torch import load_file

    members = GdnBridgeMembers(modules)
    components = [
        c for m in modules for c in (("in_proj_q", "in_proj_k", "in_proj_v") if m == "in_proj_qkv" else (m,))
    ]
    group = tuple(sorted(f"{GDN}.{c}" for c in components))
    run = AdapterParametrization("m", 4, 8.0, "lora", group, False, shared_lora_a_groups=(group,))
    write_peft_export(tmp_path, members, 0, run, GDN_TARGETS, lora_dropout=0.0)
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(tensors) == {f"base_model.model.{GDN}.{m}.lora_{f}.weight" for m in modules for f in "AB"}
    for module, b in members.b.items():
        assert torch.equal(tensors[f"base_model.model.{GDN}.{module}.lora_A.weight"], members.a.to(torch.bfloat16))
        assert torch.equal(tensors[f"base_model.model.{GDN}.{module}.lora_B.weight"], b.to(torch.bfloat16))


class CanonicalGdnSaveMembers(FakeMembers):
    """A canonical GDN component's own A and B: a checkpoint no PEFT adapter can name."""

    def get_local_shard_specs(self, slot):
        return [replace(s, key=s.key.replace(MODULE, f"{GDN}.in_proj_q")) for s in super().get_local_shard_specs(slot)]

    def export_bf16_adapter(self, slot):
        raise AssertionError("a parametrization with no plain-PEFT form gets no PEFT view")


def test_a_save_of_a_parametrization_with_no_plain_peft_form_commits_without_a_peft_view(tmp_path):
    run = AdapterParametrization("m", 2, 4.0, "canonical_lora", (f"{GDN}.in_proj_q",), False)
    committed = save_and_commit_adapter_dir(
        tmp_path, "adapter", CanonicalGdnSaveMembers(), 0, run, GDN_TARGETS, lora_dropout=0.0
    )
    assert resolve_adapter_dir(tmp_path, "adapter", tenant=None).path == committed.path
    assert not (committed.path / "adapter_model.safetensors").exists()


HEAD_RUN = AdapterParametrization("m", 2, 4.0, "canonical_lora", ("lm_head",), False)


class HeadMembers(FakeMembers):
    """A head that serves 6 rows of a vocab whose HF export has up to 8."""

    head_rows = HeadRows("lm_head.lora_B", max_source_rows=8)

    def get_local_shard_specs(self, slot):
        return [
            LocalShardSpec("lm_head.lora_A", (2, 6), (0, 0), (2, 6)),
            LocalShardSpec("lm_head.lora_B", (6, 2), (0, 0), (6, 2)),
        ]


@pytest.mark.parametrize("rows", [6, 8])
def test_a_head_source_over_the_served_vocab_loads_the_served_rows(tmp_path, rows):
    head_b = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    write_peft_adapter(
        tmp_path, {"lm_head.lora_A": FULL["lora_A"], "lm_head.lora_B": head_b}, HEAD_RUN, TARGETS, lora_dropout=0.0
    )
    members = HeadMembers()
    load_adapter_source(tmp_path, members, 0, HEAD_RUN, TARGETS, load_optimizer=False)
    assert torch.equal(members.imported.weights[1], head_b[:6].to(torch.bfloat16).float())


INKLING = get_hf_lora_targets({"model_type": "inkling_text", "num_hidden_layers": 1, "mlp_layer_types": ["dense"]})


# native names Inkling's head by the model's own path, as its probe does
INKLING_HEAD = "language_model.lm_head"
INKLING_HEAD_RUN = replace(HEAD_RUN, target_modules=(INKLING_HEAD,))


class InklingHeadMembers(FakeMembers):
    """Native Inkling's head: 6 served rows of a vocab whose released adapters hold 8."""

    head_rows = HeadRows(f"{INKLING_HEAD}.lora_B", max_source_rows=8)

    def get_local_shard_specs(self, slot):
        return [
            LocalShardSpec(f"{INKLING_HEAD}.lora_A", (2, 6), (0, 0), (2, 6)),
            LocalShardSpec(f"{INKLING_HEAD}.lora_B", (6, 2), (0, 0), (6, 2)),
        ]


def _write_inkling_head_adapter(path):
    weights = {f"{INKLING_HEAD}.lora_A": torch.full((2, 6), 3.0), f"{INKLING_HEAD}.lora_B": torch.full((8, 2), 7.0)}
    write_peft_adapter(path, weights, INKLING_HEAD_RUN, INKLING, lora_dropout=0.0)


def test_a_native_warm_start_loads_an_inkling_head_peft_dir(tmp_path):
    _write_inkling_head_adapter(tmp_path)
    members = InklingHeadMembers()
    load_adapter_source(tmp_path, members, 0, INKLING_HEAD_RUN, INKLING, load_optimizer=False)
    assert torch.equal(members.imported.weights[0], torch.full((2, 6), 3.0))
    assert torch.equal(members.imported.weights[1], torch.full((6, 2), 7.0))


def test_a_peft_dir_without_adapter_config_is_refused_with_a_remedy(tmp_path):
    # the load checks the adapter's rank, alpha and targets against the run from adapter_config.json
    _write_inkling_head_adapter(tmp_path)
    (tmp_path / "adapter_config.json").unlink()
    with pytest.raises(RuntimeError, match="no adapter_config.json.*save_pretrained"):
        load_adapter_source(tmp_path, InklingHeadMembers(), 0, INKLING_HEAD_RUN, INKLING, load_optimizer=False)


@pytest.mark.parametrize("rows", [5, 9])
def test_a_head_source_outside_the_declared_rows_is_refused(tmp_path, rows):
    head_b = torch.zeros(rows, 2)
    write_peft_adapter(
        tmp_path, {"lm_head.lora_A": FULL["lora_A"], "lm_head.lora_B": head_b}, HEAD_RUN, TARGETS, lora_dropout=0.0
    )
    with pytest.raises(RuntimeError, match=f"lm_head.lora_B: the source has {rows} rows; this run serves 6"):
        load_adapter_source(tmp_path, HeadMembers(), 0, HEAD_RUN, TARGETS, load_optimizer=False)
