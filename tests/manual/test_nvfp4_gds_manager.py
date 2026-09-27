"""GPU manager behavior with an in-memory I/O double, NOT GDS qualification.

Run: python tests/manual/test_nvfp4_gds_manager.py --receiver-path /path/to/sglang/srt/weight_sync/local_checkpoint.py
Canonical safetensors bytes are uploaded once by the test fixture. All simulated
baseline reads/writes then use GPU copies. This cannot qualify cuFile or storage.
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.nvfp4_gds import Nvfp4GdsDelta
from miles.utils.disk_delta import checkpoint_tensor_location, make_tensor_reader


class _GpuFileStore:
    def __init__(self, source, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.lock = threading.Lock()
        self.reads = []
        # Test setup only: stand in for the already existing canonical disk file.
        raw = np.frombuffer(source.read_bytes(), dtype=np.uint8).copy()
        self.buffers = {self.key(source): torch.from_numpy(raw).to(device)}
        torch.cuda.current_stream(device).synchronize()

    @staticmethod
    def key(path):
        info = os.stat(path)
        return info.st_dev, info.st_ino

    def backend(self, path, device, writable=False, *, executor):
        return _GpuFile(self, Path(path), device, writable, executor)


class _GpuFile:
    def __init__(self, store, path, device, writable, executor):
        self.store, self.path, self.device = store, path, device
        self.writable, self.executor = writable, executor
        self.futures = []

    def read_into(self, offset, dst, *, expected_bytes=None):
        assert offset % 4096 == dst.data_ptr() % 4096 == dst.numel() % 4096 == 0
        expected = dst.numel() if expected_bytes is None else expected_bytes
        if expected < dst.numel():
            assert offset + expected == os.stat(self.path).st_size
        self.store.reads.append((str(self.path), offset, dst.numel(), expected, dst.data_ptr()))
        ready = torch.cuda.current_stream(self.device).record_event()
        future = self.executor.submit(self._transfer, offset, dst, ready, expected, False)
        self.futures.append(future)
        return future

    def write_from(self, offset, src, ready_event):
        assert self.writable
        assert offset % 4096 == src.data_ptr() % 4096 == src.numel() % 4096 == 0
        future = self.executor.submit(self._transfer, offset, src, ready_event, src.numel(), True)
        self.futures.append(future)
        return future

    def _transfer(self, offset, tensor, ready, count, write):
        ready.synchronize()
        with self.store.lock, torch.cuda.device(self.device), torch.cuda.stream(self.store.stream):
            key = self.store.key(self.path)
            if write:
                size = os.stat(self.path).st_size
                old = self.store.buffers.get(key)
                if old is None or old.numel() < size:
                    data = torch.empty(size, dtype=torch.uint8, device=self.device)
                    if old is not None:
                        data[: old.numel()].copy_(old)
                    self.store.buffers[key] = data
                self.store.buffers[key][offset : offset + count].copy_(tensor)
            else:
                tensor[:count].copy_(self.store.buffers[key][offset : offset + count])
            self.store.stream.record_event().synchronize()
        return count

    def close(self):
        for future in self.futures:
            future.result()


def _reject_cpu(self, *args, **kwargs):
    raise AssertionError("Manager attempted a full tensor CPU transfer")


class _TransferGuard:
    """Permit only the codec's scalar metadata and compressed-payload D2H sites."""

    def __enter__(self):
        self.original_copy = torch.Tensor.copy_
        self.original_to = torch.Tensor.to

        def copy_(destination, source, *args, **kwargs):
            if destination.device.type == "cpu" and source.is_cuda:
                frame = inspect.currentframe().f_back
                assert frame.f_code.co_filename.endswith("miles/utils/gpu_delta.py")
                assert frame.f_code.co_name in ("_encode", "_copy_payloads")
                if frame.f_code.co_name == "_encode":
                    assert destination.ndim == 2 and destination.shape[1] == 2
            return self.original_copy(destination, source, *args, **kwargs)

        def to(tensor, *args, **kwargs):
            target = kwargs.get("device", args[0] if args else None)
            if isinstance(target, torch.Tensor):
                target = target.device
            if isinstance(target, (str, torch.device)) and tensor.is_cuda:
                assert torch.device(target).type == "cuda", "Manager attempted a full tensor CPU transfer"
            return self.original_to(tensor, *args, **kwargs)

        self.patches = [
            patch.object(torch.Tensor, "cpu", _reject_cpu),
            patch.object(torch.Tensor, "to", to),
            patch.object(torch.Tensor, "copy_", copy_),
        ]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *args):
        for item in reversed(self.patches):
            item.stop()


def _fixture(directory):
    rng = np.random.default_rng(20260927)
    weights = {}
    units = {}
    for expert in range(5):
        prefix = f"model.layers.0.mlp.experts.{expert}.gate_proj"
        packed = torch.from_numpy(rng.integers(0, 256, (33 + expert, 129), dtype=np.uint8))
        scale = torch.from_numpy(rng.integers(0, 256, (33 + expert, 17), dtype=np.uint8)).view(torch.float8_e4m3fn)
        family = {
            prefix + ".weight": packed,
            prefix + ".weight_scale": scale,
            prefix + ".weight_scale_2": torch.tensor(0.125 * (expert + 1)),
        }
        weights.update(family)
        units[f"expert-{expert}"] = list(family)
    source = directory / "canonical"
    source.mkdir()
    safetensors.torch.save_file(weights, source / "model.safetensors")
    assert any(checkpoint_tensor_location(str(source), name)[1] % 4096 for name in weights)
    return source, weights, units


def _raw(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().copy()


def _version(weights, version):
    result = {}
    for name, tensor in weights.items():
        raw = _raw(tensor)
        if version == 2:
            raw[::97] ^= np.uint8(0x81)
        elif version == 3:
            raw ^= np.uint8(0xFF)
        result[name] = torch.from_numpy(raw).view(tensor.dtype).reshape(tensor.shape)
    return result


def _apply_receiver(receiver, receiver_dir, publication, result, version, expected):
    publication.mkdir()
    if result.delta:
        safetensors.numpy.save_file(result.delta, publication / "delta.safetensors", metadata=result.checksums)
    index = {
        "metadata": {
            "version": version,
            "base_version": version - 1,
            "compression_format": "zstd",
            "delta_encoding": "xor",
            "checksum_format": "adler32",
        },
        "weight_map": {name: "delta.safetensors" for name in result.delta},
    }
    (publication / "model.safetensors.index.json").write_text(json.dumps(index))
    receiver._apply_delta(str(receiver_dir), str(publication))
    reader = make_tensor_reader(str(receiver_dir))
    for name, tensor in expected.items():
        assert np.array_equal(reader(name), _raw(tensor)), name
    assert receiver._read_applied_version(str(receiver_dir)) == version


def _test_versions(directory, device, receiver):
    source, initial, units = _fixture(directory)
    store = _GpuFileStore(source / "model.safetensors", device)
    receiver_dir = directory / "receiver"
    shutil.copytree(source, receiver_dir)
    receiver._write_applied_version(str(receiver_dir), 0)
    with patch("miles.utils.gds_io.GdsBackend", store.backend):
        manager = Nvfp4GdsDelta(
            str(source), str(directory / "baselines"), device, quantization_config={"quant_method": "nvfp4"}
        )
    previous = initial
    for version in range(4):
        expected = _version(initial, version)
        gpu = {name: tensor.to(device) for name, tensor in expected.items()}
        if version == 0:
            # Baseline capture must use checkpoint bytes, not these passed bytes.
            for tensor in gpu.values():
                tensor.reshape(-1).view(torch.uint8).bitwise_xor_(0x55)
        before_reads = len(store.reads)
        with _TransferGuard():
            manager.begin(capture_baseline=version == 0, weight_version=version)
            for key, names in units.items():
                manager.prefetch(key)
                unchanged = [
                    ("model.layers.0.mlp.shared_experts.weight", torch.ones(2, device=device)),
                    ("model.layers.0.mlp.experts.99.weight", torch.ones(2, dtype=torch.bfloat16, device=device)),
                ]
                remaining = manager.process(key, [(name, gpu[name]) for name in names] + unchanged)
                assert [name for name, _ in remaining] == [name for name, _ in unchanged]
                assert len(manager._pending) <= 2
            result = manager.finish()
            manager.commit()
        if version == 0:
            assert not result.delta
            assert any(expected < requested for _, _, requested, expected, _ in store.reads)
        else:
            reads = store.reads[before_reads:]
            assert len(reads) == len(units)
            assert len({pointer for *_, pointer in reads}) == 2, "Two slots must be reused across five units"
            total = sum(t.numel() * t.element_size() for t in expected.values())
            changed = sum(np.count_nonzero(_raw(previous[name]) ^ _raw(value)) for name, value in expected.items())
            assert (result.total_bytes, result.changed_bytes) == (total, changed)
            assert bool(result.delta) == (version != 1)
            _apply_receiver(receiver, receiver_dir, directory / f"delta-{version}", result, version, expected)
        assert manager._version == version
        previous = expected
    manager._executor.shutdown(wait=True)
    return source, initial, units, store


def _test_rejections(directory, device, source, initial, units, store):
    for case in ("wrong-device", "noncontiguous"):
        with patch("miles.utils.gds_io.GdsBackend", store.backend):
            manager = Nvfp4GdsDelta(
                str(source), str(directory / case), device, quantization_config={"quant_method": "nvfp4"}
            )
        key, names = next(iter(units.items()))
        tensors = [(name, initial[name].to(device)) for name in names]
        name, tensor = tensors[0]
        tensors[0] = (name, initial[name] if case == "wrong-device" else tensor.T.contiguous().T)
        manager.begin(capture_baseline=True, weight_version=0)
        manager.process(key, tensors)
        try:
            manager.finish()
        except RuntimeError:
            assert isinstance(manager.error, ValueError)
            assert ("implicit transfers" if case == "wrong-device" else "contiguous") in str(manager.error)
        else:
            raise AssertionError(f"Manager accepted {case}")
        manager._executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-path", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("qualified_delta_receiver", args.receiver_path)
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with tempfile.TemporaryDirectory(prefix="gpu-delta-manager-") as temporary:
        directory = Path(temporary)
        source, initial, units, store = _test_versions(directory, device, receiver)
        _test_rejections(directory, device, source, initial, units, store)
    print(
        json.dumps(
            {
                "status": "PASS",
                "versions": 4,
                "expert_units": 5,
                "rejections": 2,
                "io_backend": "GPU in-memory test double; NOT GDS qualification",
                "receiver": str(args.receiver_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
