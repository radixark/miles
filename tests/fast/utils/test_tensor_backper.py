from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from miles.utils.tensor_backper import MainCastContext, TensorBackuper, _TensorBackuperMainCast


@pytest.fixture(autouse=True)
def _cpu_fallback(monkeypatch):
    if torch.cuda.is_available():
        yield
        return
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    real_empty_like = torch.empty_like

    def empty_like_no_pin(tensor, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty_like(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", empty_like_no_pin)
    yield


class _FakeOptimizer:
    """Like _copy_main_params_to_model_params: writes only this rank's owned shard."""

    def __init__(self, mains, staging, shards):
        self._mains = mains
        self._staging = staging
        self._shards = shards

    def _copy_main_params_to_model_params(self):
        for name, main in self._mains.items():
            self._staging[name][self._shards[name]] = main.to(torch.bfloat16)


class _FakeModelChunk:
    """start_param_sync = param all-gather: staging buffers -> live params."""

    def __init__(self, params, staging):
        self._params = params
        self._staging = staging

    def start_param_sync(self, force_sync=False):
        assert force_sync
        for name, param in self._params.items():
            param.copy_(self._staging[name])


class _Setup:
    def __init__(self, num_params=3, numel=16, num_extras=1, check=False, shards=None):
        generator = torch.Generator().manual_seed(0)
        self.mains = {f"p{i}": torch.randn(numel, generator=generator) for i in range(num_params)}
        self.params = {name: main.to(torch.bfloat16) for name, main in self.mains.items()}
        self.staging = {name: param.clone() for name, param in self.params.items()}
        self.shards = shards or {name: slice(None) for name in self.mains}
        self.extras = {f"extra{i}": torch.randn(4, generator=generator) for i in range(num_extras)}
        owned = {name: main[self.shards[name]] for name, main in self.mains.items()}
        self.optimizer = SimpleNamespace(chained_optimizers=[_FakeOptimizer(owned, self.staging, self.shards)])
        self.model_chunk = _FakeModelChunk(self.params, self.staging)
        ctx = MainCastContext(
            cast_main_to_params=lambda: [
                opt._copy_main_params_to_model_params() for opt in self.optimizer.chained_optimizers
            ],
            model_chunks=[self.model_chunk],
            extras_getter=lambda: iter(self.extras.items()),
            rematerializable_ids={id(t) for t in self.params.values()},
            check=check,
        )
        self.backuper = TensorBackuper.create(
            source_getter=lambda: iter({**self.params, **self.extras}.items()),
            main_cast_ctx=ctx,
        )

    def corrupt_live_tensors(self):
        for param in self.params.values():
            param.fill_(float("nan"))
        for extra in self.extras.values():
            extra.fill_(float("nan"))


def test_create_returns_main_cast_variant():
    assert isinstance(_Setup().backuper, _TensorBackuperMainCast)


def test_round_trip_restores_bit_identical_weights():
    setup = _Setup()
    expected = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, expected[name]), name


def test_restore_only_covers_owned_shard_and_relies_on_param_sync():
    numel = 16
    shards = {f"p{i}": slice(0, numel // 2) for i in range(3)}
    setup = _Setup(numel=numel, shards=shards)
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    # The unowned staging halves stand in for the other DP rank's cast.
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_chained_optimizer_casts_every_inner_optimizer():
    setup = _Setup()
    names = list(setup.mains)
    inner = [_FakeOptimizer({n: setup.mains[n]}, setup.staging, setup.shards) for n in names]
    setup.optimizer.chained_optimizers = inner  # the ctx closure reads through
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_get_returns_pinned_backup_for_extras_and_live_tensors_for_params():
    setup = _Setup()
    setup.backuper.backup("actor")
    got = setup.backuper.get("actor")
    for name, param in setup.params.items():
        assert got[name].data_ptr() == param.data_ptr(), name
    for name, extra in setup.extras.items():
        # Extras are paused during update_weights, so get() must hand out the backup.
        assert got[name].data_ptr() != extra.data_ptr(), name
        assert torch.equal(got[name], extra), name


def test_check_verifies_first_cycles_and_raises_on_corruption():
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.mains["p0"][0] += 1.0
    with pytest.raises(RuntimeError, match="not bit-identical"):
        setup.backuper.restore("actor")


def test_check_stops_after_check_num_cycles():
    setup = _Setup(check=True)
    for _ in range(setup.backuper._check_num_cycles):
        setup.backuper.backup("actor")
        setup.backuper.restore("actor")
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_no_check_computes_no_hashes():
    setup = _Setup(check=False)
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_check_detects_tensor_set_change():
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.params["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    setup.staging["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="changed the tensor set"):
        setup.backuper.restore("actor")


def test_get_rejects_unknown_tensor_in_source():
    setup = _Setup()
    setup.backuper.backup("actor")
    setup.params["stray_buffer"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="stray_buffer"):
        setup.backuper.get("actor")


def test_non_actor_tag_keeps_full_pinned_copy():
    setup = _Setup()
    ref_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("ref")
    assert setup.backuper.backup_tags == ["actor", "ref"]
    setup.corrupt_live_tensors()
    setup.backuper.restore("ref")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, ref_values[name]), name
    got = setup.backuper.get("ref")
    for name in ref_values:
        # Pinned host copies, never the live tensors.
        assert got[name].data_ptr() != {**setup.params, **setup.extras}[name].data_ptr(), name


def test_actor_restore_wins_after_ref_switch():
    # A ref switch overwrites the live params; the actor restore must win them back.
    setup = _Setup()
    actor_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    for param in setup.params.values():
        param.fill_(7.0)  # stand-in for loading ref weights into the model
    setup.backuper.backup("ref")
    setup.backuper.restore("ref")
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, actor_values[name]), name


def test_normal_release_drops_pinned_snapshot_and_allows_rebackup():
    source = {"weight": torch.arange(4, dtype=torch.float32)}
    backuper = TensorBackuper.create(source_getter=lambda: iter(source.items()))

    backuper.backup("actor")
    backuper.release("actor")

    assert "actor" not in backuper.backup_tags
    assert not backuper.has_backup("actor")
    with pytest.raises(AssertionError, match="never backed up"):
        backuper.get("actor")

    source["weight"].add_(10)
    backuper.backup("actor")
    source["weight"].zero_()
    backuper.restore("actor")
    torch.testing.assert_close(source["weight"], torch.arange(4, dtype=torch.float32) + 10)


def test_main_cast_release_drops_only_actor_extras_and_allows_rebackup():
    setup = _Setup()
    setup.backuper.backup("actor")
    setup.backuper.backup("ref")

    setup.backuper.release("actor")

    assert setup.backuper.backup_tags == ["actor", "ref"]
    assert not setup.backuper.has_backup("actor")
    with pytest.raises(AssertionError, match="never backed up"):
        setup.backuper.get("actor")
    assert setup.backuper.get("ref")

    expected = {name: tensor.clone() for name, tensor in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, expected[name]), name
