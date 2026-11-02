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
    load_adapter_state,
    read_adapter_state_keys,
    save_adapter_state,
)

FULL = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)


def _state(rows, *, primary=True, fill=None, step=7):
    lo, hi = rows
    weight = FULL[lo:hi].clone() if fill is None else torch.full((hi - lo, 4), fill)
    spec = LocalShardSpec("m.lora_A", (6, 4), (lo, 0), (hi - lo, 4), primary)
    return AdapterState([spec], [weight], [weight * 2], [weight * 3], step=step)


def _save_then_reload(rank, world_size, port, *, directory, save_rows, load_rows, primaries):
    init_gloo(rank, world_size, port=port)
    save_adapter_state(Path(directory), _state(save_rows[rank], primary=primaries[rank]))
    template = _state(load_rows[rank], fill=-1.0, step=None)
    step = load_adapter_state(Path(directory), template)
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
    assert load_adapter_state(Path(directory), template) == 7
    assert torch.equal(template.weights[0], FULL)
    assert read_adapter_state_keys(Path(directory)) == {"m.lora_A.weight", "m.lora_A.exp_avg", "m.lora_A.exp_avg_sq"}


def _disagree_on_step(rank, world_size, port, *, directory):
    init_gloo(rank, world_size, port=port)
    with pytest.raises(AssertionError, match="disagree on the optimizer step"):
        save_adapter_state(Path(directory), _state([(0, 3), (3, 6)][rank], step=rank))
    dist.destroy_process_group()


def test_ranks_must_agree_on_step():
    directory = tempfile.mkdtemp()
    run_multiprocess(partial(_disagree_on_step, directory=directory))


def test_only_fp32_is_stored():
    spec = LocalShardSpec("m.lora_A", (2, 2), (0, 0), (2, 2))
    weight = torch.zeros(2, 2, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="fp32 only"):
        AdapterState([spec], [weight], [weight], [weight], step=0)
