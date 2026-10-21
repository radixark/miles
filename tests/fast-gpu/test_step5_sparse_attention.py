"""Sparse attention must agree across TP/SP layouts and preserve selections during recomputation."""

from tests.ci.ci_register import register_cuda_ci

# the kernel is tuned for, and only run on, Blackwell
# the largest layout is TP4
register_cuda_ci(est_time=300, suite="stage-c-4-gpu-b200", labels=["miles-plugin"], hardware=["blackwell"], num_gpus=4)

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS

_WORKER = Path(__file__).with_name("_step5_sparse_attention_worker.py")
_REPO_ROOT = Path(__file__).parents[2]
_REFERENCE = (1, False)
_LAYOUTS = [_REFERENCE, (2, False), (2, True), (4, False), (4, True)]
_GRADIENTS = ("d_hidden", "d_q_proj", "d_k_proj", "d_v_proj", "d_g_proj", "d_o_proj", "d_q_norm", "d_k_norm")


def _layout_id(layout: tuple[int, bool]) -> str:
    tp_size, sequence_parallel = layout
    return f"tp{tp_size}{'-sp' if sequence_parallel else ''}"


def _run_worker(layout: tuple[int, bool], output: Path) -> dict:
    tp_size, sequence_parallel = layout
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={tp_size}",
        str(_WORKER),
        "--output",
        str(output),
    ]
    if sequence_parallel:
        command.append("--sequence-parallel")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    assert f"PASS {_layout_id(layout)}" in result.stdout
    return torch.load(output, weights_only=True)


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict[tuple[int, bool], dict]:
    directory = tmp_path_factory.mktemp("step5-sparse-attention")
    return {layout: _run_worker(layout, directory / f"{_layout_id(layout)}.pt") for layout in _LAYOUTS}


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    """Frobenius error relative to the reference's norm."""
    return ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)).item()


def _every_region(lengths: list[int], num_groups: int) -> torch.Tensor:
    rows = []
    for length in lengths:
        own = torch.arange(length)[:, None] // REGION_SIZE
        slot = torch.arange(MAX_SELECTED_REGIONS)[None]
        rows.append(torch.where(slot <= own, slot, -1))
    return torch.cat(rows).to(torch.int32)[:, None].expand(-1, num_groups, -1)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_layout_id)
def test_sparse_layer_is_the_dense_layer_while_every_region_is_selected(results, layout):
    short = results[layout]["short"]
    assert torch.equal(short["selection"], _every_region(short["lengths"], num_groups=4))

    errors = {key: _rel_err(short["sparse"][key], short["dense"][key]) for key in ("out", *_GRADIENTS)}
    # BF16 rounding inside each attention kernel puts two correct kernels ~3e-3 apart
    assert errors["out"] < 1e-2 and errors["d_hidden"] < 1e-2, errors
    assert all(err < 2e-2 for err in errors.values()), errors


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_layout_id)
@pytest.mark.parametrize("pack", ["short", "long"])
def test_recompute_selects_the_forwards_regions_bit_for_bit(results, layout, pack):
    result = results[layout][pack]
    assert result["selection_passes"] == 2, "the checkpointed forward and its recompute each select once"
    assert result["recomputed_selection_equal"]
    assert result["checkpointed_out_equal"]


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_layout_id)
@pytest.mark.parametrize("pack", ["short", "long"])
def test_reuse_preserves_selections_outputs_and_gradients(results, layout, pack):
    result = results[layout][pack]
    assert result["reuse_selection_passes"] == 1
    assert result["reuse_selection_equal"]
    assert result["reuse_out_equal"]
    assert result["reuse_gradient_errors"].keys() == set(_GRADIENTS)
    assert all(error < 1e-2 for error in result["reuse_gradient_errors"].values()), result["reuse_gradient_errors"]


def test_long_sequences_select_by_score(results):
    long = results[_REFERENCE]["long"]
    first = long["selection"][: long["lengths"][0]]
    scored = first[(TOP_REGIONS + 1) * REGION_SIZE :]
    assert (scored != -1).all(), "past position 4103 every row holds 512 scored regions and its own"
    # not merely the oldest 512 regions
    assert (scored[:, :, TOP_REGIONS - 1] > TOP_REGIONS - 1).float().mean() > 0.5


@pytest.mark.parametrize("layout", _LAYOUTS[1:], ids=_layout_id)
@pytest.mark.parametrize("pack", ["short", "long"])
def test_every_layout_agrees_with_tp1(results, layout, pack):
    got, want = results[layout][pack], results[_REFERENCE][pack]
    matching_rows = (got["selection"] == want["selection"]).all(-1).float().mean().item()
    errors = {key: _rel_err(got["sparse"][key], want["sparse"][key]) for key in ("out", *_GRADIENTS)}
    assert matching_rows == 1.0, f"{matching_rows:.6f} of the selection rows equal TP1's"
    assert errors["out"] < 1e-2 and errors["d_hidden"] < 1e-2, errors
    assert all(err < 2e-2 for err in errors.values()), errors
