"""Two-rank nested FSDP2 CPU offload parity, including uneven local shards.

The same-stream queued-work design is informed by verl PR #8187 (SelfDLer).
This Miles test is independently implemented; no verl test source is copied.
"""

import hashlib
import inspect
import json
import os
import platform
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tests.ci.ci_register import register_cuda_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard

from miles.backends.training_utils.torch_native.offload import move_train_state

# Unmeasured first-run upper bound. Replace after an authorized timing run.
register_cuda_ci(
    est_time=600, suite="stage-b-2-gpu-h200", labels=["fsdp", "torchtitan"], hardware=["hopper"], num_gpus=2
)

_WORLD_SIZE = 2
_CYCLES = 5
_QUEUED_MATMULS = 32
_ATOL = 2e-6
_RTOL = 2e-5


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * torch.tanh(self.linear(x))


class _TinyModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.entry = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList([_Block(dim), _Block(dim)])
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _make_model(dim: int) -> _TinyModel:
    model = _TinyModel(dim)
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)
            param.copy_(((values + 7 * index) % 31 - 15) / (32 * dim**0.5) + 0.25 * torch.eye(dim))
    return model.cuda()


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return _local(tensor).detach().cpu().clone()


def _train_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    state = dict(model.named_parameters())
    for name, param in model.named_parameters():
        for key, value in optimizer.state.get(param, {}).items():
            if isinstance(value, torch.Tensor):
                state[f"{name}.optimizer.{key}"] = value
    return state


def _check(name, actual, expected, report, output, *, exact=False):
    actual = _cpu(actual)
    report["parameter_or_output"] = name
    try:
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "nonfinite values"
        torch.testing.assert_close(actual, expected, atol=0 if exact else _ATOL, rtol=0 if exact else _RTOL)
    except AssertionError as error:
        report["status"] = "FAIL"
        report["first_difference"] = {"stage": report["stage"], "name": name, "cycle": report["cycle"]}
        torch.save({"actual": actual, "expected": expected}, output / f"rank-{report['rank']}-difference.pt")
        raise AssertionError(f"rank={report['rank']}, stage={report['stage']}, name={name}: {error}") from error


def _expected_local(tensor: torch.Tensor, reference: torch.Tensor, rank: int) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return reference
    assert tensor.placements == (Shard(0),)
    rows = (reference.shape[0] + _WORLD_SIZE - 1) // _WORLD_SIZE
    return reference.narrow(0, rank * rows, _local(tensor).shape[0])


def _reference_step(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    expected_output = _cpu(output)
    output.square().mean().backward()
    gradients = {name: _cpu(param.grad) for name, param in model.named_parameters()}
    optimizer.step()
    state = {name: _cpu(value) for name, value in _train_state(model, optimizer).items()}
    with torch.no_grad():
        next_output = _cpu(model(inputs))
    return expected_output, gradients, state, next_output


def _round_trip(model, optimizer, mover, workload, before, report, output, *, offload):
    producer = torch.cuda.current_stream()
    mover.wait_stream(producer)
    retained = [_local(value).detach() for value in _train_state(model, optimizer).values()]
    for value in retained:
        if value.is_cuda:
            value.record_stream(mover)
    with torch.cuda.stream(mover):
        # Real work on the conversion stream; all producer/consumer dependencies remain intact.
        for _ in range(_QUEUED_MATMULS):
            torch.mm(workload[0], workload[1], out=workload[2])
        if offload:
            report["stage"] = "model_d2h"
            move_train_state([model], [optimizer], to_host=True)
            if before is not None:
                report["stage"] = "cpu_parameters"
                for name, value in _train_state(model, optimizer).items():
                    assert _local(value).device.type == "cpu", name
                    _check(name, value, before[name], report, output, exact=True)
                report["cpu_pinned"] = {name: _local(param).is_pinned() for name, param in model.named_parameters()}
            report["stage"] = "model_h2d"
            move_train_state([model], [optimizer], to_host=False)
    # move_train_state already synchronizes. This also covers the no-offload control.
    producer.wait_stream(mover)
    return retained


def _exercise(model, reference, optimizer, ref_optimizer, options, report, output):
    dim, stream, probe, offload = options
    mover = torch.cuda.current_stream() if stream == "default" else torch.cuda.Stream()
    workload = [torch.ones((1024, 1024), device="cuda") for _ in range(3)]
    for _ in range(4):
        torch.mm(workload[0], workload[1], out=workload[2])
    for cycle in range(_CYCLES):
        report.update(cycle=cycle, stage="reference")
        values = torch.arange(4 * dim, device="cuda", dtype=torch.float32).reshape(4, dim)
        inputs = ((values + 3 * cycle) % 23 - 11) / 16  # Identical on every rank.
        expected, gradients, expected_state, expected_next = _reference_step(reference, ref_optimizer, inputs)
        optimizer.zero_grad(set_to_none=True)
        before = (
            {name: _cpu(value) for name, value in _train_state(model, optimizer).items()}
            if probe == "parameters-first"
            else None
        )
        torch.cuda.synchronize()  # Finish setup, before queuing the transfer-stream workload.
        retained = _round_trip(model, optimizer, mover, workload, before, report, output, offload=offload)
        if before is not None:
            report["stage"] = "reloaded_parameters"
            for name, value in _train_state(model, optimizer).items():
                _check(name, value, before[name], report, output, exact=True)
        report["stage"] = "first_forward"
        # No value probe, full_tensor(), logging or gather before this forward in forward-first mode.
        actual = model(inputs)
        _check("<model.output>", actual, expected, report, output)
        del retained
        report["stage"] = "backward"
        actual.square().mean().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            _check(name, param.grad, _expected_local(param.grad, gradients[name], report["rank"]), report, output)
        report["stage"] = "optimizer_step"
        optimizer.step()
        report["stage"] = "updated_state"
        actual_state = _train_state(model, optimizer)
        assert actual_state.keys() == expected_state.keys()
        for name, value in actual_state.items():
            _check(name, value, _expected_local(value, expected_state[name], report["rank"]), report, output)
        report["stage"] = "next_forward"
        with torch.no_grad():
            _check("<model.output>", model(inputs), expected_next, report, output)


def _worker(rank: int, init_file: str, output_directory: str, options: tuple) -> None:
    output = Path(output_directory)
    dim, stream, probe, offload = options
    report = {
        "rank": rank,
        "cycle": -1,
        "stage": "setup",
        "status": "ERROR",
        "first_difference": None,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dim": dim,
        "stream": stream,
        "probe": probe,
        "offload": offload,
    }
    try:
        source = Path(inspect.getsourcefile(inspect.unwrap(move_train_state))).resolve()
        assert source == Path(__file__).resolve().parents[2] / "miles/backends/training_utils/torch_native/offload.py"
        report["official_function_file"] = str(source)
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        assert os.environ.get("CUDA_LAUNCH_BLOCKING") in (None, "0"), "Do not serialize the asynchronous baseline"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.set_device(rank)
        report["gpu"] = torch.cuda.get_device_name()
        report["nccl"] = torch.cuda.nccl.version()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        dist.init_process_group(
            "nccl",
            init_method=Path(init_file).as_uri(),
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=120),
        )
        model, reference = _make_model(dim), _make_model(dim)
        mesh = init_device_mesh("cuda", (_WORLD_SIZE,))
        for block in model.blocks:
            fully_shard(block, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        report["shards"] = {}
        for name, param in model.named_parameters():
            rows = (dim + _WORLD_SIZE - 1) // _WORLD_SIZE
            expected_rows = min(rows, dim - rank * rows)
            assert isinstance(param, DTensor) and param.placements == (Shard(0),), name
            shape = list(_local(param).shape)
            report["shards"][name] = {"global": list(param.shape), "local": shape, "placement": "Shard(0)"}
            assert shape == [expected_rows, dim], (name, shape)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False, fused=False)
        report["stage"] = "warmup"
        inputs = torch.ones((4, dim), device="cuda") / 8
        for module, optim in ((model, optimizer), (reference, ref_optimizer)):
            module(inputs).square().mean().backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        expected_state = {name: _cpu(value) for name, value in _train_state(reference, ref_optimizer).items()}
        for name, value in _train_state(model, optimizer).items():
            _check(name, value, _expected_local(value, expected_state[name], rank), report, output)
        _exercise(model, reference, optimizer, ref_optimizer, options, report, output)
        report.update(status="PASS", stage="complete")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("dim", [64, 65])
@pytest.mark.parametrize("stream", ["default", "side"])
@pytest.mark.parametrize("probe", ["forward-first", "parameters-first"])
@pytest.mark.parametrize("offload", [False, True], ids=["resident", "transfer"])
def test_nested_fsdp2_offload_parity(tmp_path: Path, dim: int, stream: str, probe: str, offload: bool) -> None:
    assert torch.cuda.device_count() >= _WORLD_SIZE, "This test requires two visible GPUs; do not reduce its topology"
    mp.spawn(
        _worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), (dim, stream, probe, offload)),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    for rank in range(_WORLD_SIZE):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text())["status"] == "PASS"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
