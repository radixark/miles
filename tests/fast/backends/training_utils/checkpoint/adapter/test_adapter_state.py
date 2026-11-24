"""``adapter_state/`` reads back bitwise under any rank split, and each piece is written once."""

import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    LocalShardSpec,
    _load_adapter_state,
    _save_adapter_state,
    read_adapter_state_keys,
)
from miles.utils.distributed_utils import init_gloo_group

FULL = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)


def _state(rows, *, primary=True, fill=None, step=7):
    lo, hi = rows
    weight = FULL[lo:hi].clone() if fill is None else torch.full((hi - lo, 4), fill)
    spec = LocalShardSpec("m.lora_A", (6, 4), (lo, 0), (hi - lo, 4), primary)
    return AdapterState([spec], [weight], [weight * 2], [weight * 3], step=step)


def _save_then_reload(rank, world_size, port, *, directory, save_rows, load_rows, primaries):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    _save_adapter_state(Path(directory), _state(save_rows[rank], primary=primaries[rank]))
    template = _state(load_rows[rank], fill=-1.0, step=None)
    step = _load_adapter_state(Path(directory), template)
    lo, hi = load_rows[rank]
    assert step == 7
    assert torch.equal(template.weights[0], FULL[lo:hi])
    assert torch.equal(template.exp_avg_sqs[0], FULL[lo:hi] * 3)
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "save_rows, primaries",
    [([(0, 2), (2, 6)], [True, True]), ([(0, 6), (0, 6)], [True, False])],
    ids=["uneven_split", "replicated"],
)
def test_reload_under_a_different_split_is_bitwise(save_rows, primaries):
    directory = tempfile.mkdtemp()
    run_multiprocess(
        partial(
            _save_then_reload,
            directory=directory,
            save_rows=save_rows,
            load_rows=[(0, 4), (4, 6)],
            primaries=primaries,
        )
    )
    template = _state((0, 6), fill=-1.0, step=None)
    assert _load_adapter_state(Path(directory), template) == 7
    assert torch.equal(template.weights[0], FULL)
    assert read_adapter_state_keys(Path(directory)) == {"m.lora_A.weight", "m.lora_A.exp_avg", "m.lora_A.exp_avg_sq"}


def _banded_state(bands, *, fill=None, step=7):
    """Several row bands of one key on one rank, as a concatenation of separately gathered shards gives."""
    weights = [FULL[lo:hi].clone() if fill is None else torch.full((hi - lo, 4), fill) for lo, hi in bands]
    specs = [LocalShardSpec("m.lora_A", (6, 4), (lo, 0), (hi - lo, 4)) for lo, hi in bands]
    return AdapterState(specs, weights, [w * 2 for w in weights], [w * 3 for w in weights], step=step)


def _save_then_reload_bands(rank, world_size, port, *, directory, save_bands, load_bands):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    _save_adapter_state(Path(directory), _banded_state(save_bands[rank]))
    template = _banded_state(load_bands[rank], fill=-1.0, step=None)
    assert _load_adapter_state(Path(directory), template) == 7
    for (lo, hi), weight, exp_avg_sq in zip(load_bands[rank], template.weights, template.exp_avg_sqs, strict=True):
        assert torch.equal(weight, FULL[lo:hi]) and torch.equal(exp_avg_sq, FULL[lo:hi] * 3)
    dist.destroy_process_group()


_BANDS = [[(0, 1), (3, 4)], [(1, 3), (4, 6)]]
_ROWS = [[(0, 4)], [(4, 6)]]


@pytest.mark.parametrize(
    "save_bands, load_bands", [(_BANDS, _ROWS), (_ROWS, _BANDS)], ids=["save_bands", "load_bands"]
)
def test_several_pieces_of_one_key_on_a_rank_reload_bitwise(save_bands, load_bands):
    directory = tempfile.mkdtemp()
    run_multiprocess(
        partial(_save_then_reload_bands, directory=directory, save_bands=save_bands, load_bands=load_bands)
    )
    template = _state((0, 6), fill=-1.0, step=None)
    assert _load_adapter_state(Path(directory), template) == 7
    assert torch.equal(template.weights[0], FULL) and torch.equal(template.exp_avgs[0], FULL * 2)


def _one_rank_holds_nothing(rank, world_size, port, *, directory):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    state = _state((0, 6)) if rank == 0 else AdapterState([], [], [], [], step=None)
    _save_adapter_state(Path(directory), state)
    dist.destroy_process_group()


def test_a_rank_without_parameters_agrees_on_the_step():
    directory = tempfile.mkdtemp()
    run_multiprocess(partial(_one_rank_holds_nothing, directory=directory))
    assert _load_adapter_state(Path(directory), _state((0, 6), fill=-1.0, step=None)) == 7


def test_a_load_into_the_same_block_twice_is_refused(tmp_path):
    spec = LocalShardSpec("m.lora_B", (2, 2), (0, 0), (2, 2))
    saved = torch.ones(2, 2)
    _save_adapter_state(tmp_path, AdapterState([spec], [saved], [saved], [saved], step=1))
    blocks = [torch.zeros(2, 2), torch.zeros(2, 2)]
    template = AdapterState([spec, spec], blocks, [b.clone() for b in blocks], [b.clone() for b in blocks], step=None)
    # the first block would stay unfilled
    with pytest.raises(RuntimeError, match="several blocks"):
        _load_adapter_state(tmp_path, template)
