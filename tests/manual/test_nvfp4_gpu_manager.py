"""Qualify the pinned-CPU baseline/GPU delta pipeline against SGLang's receiver.

Run with --receiver-path pointing to weight_sync/local_checkpoint.py. Uses real
CUDA copies and GPU compression; no I/O backend, disk cache, or test double.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.nvfp4_gpu import Nvfp4GpuDelta
from miles.utils.disk_delta import make_tensor_reader


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
    receiver_dir = directory / "receiver"
    shutil.copytree(source, receiver_dir)
    (receiver_dir / receiver.SYNC_DIR).mkdir()
    receiver._write_applied_version(str(receiver_dir), 0)
    with torch.device(device):
        manager = Nvfp4GpuDelta(str(source), device, quantization_config={"quant_method": "nvfp4"})
    producer = torch.cuda.Stream(device=device)
    previous = initial
    for version in range(4):
        expected = _version(initial, version)
        pointers = set()
        with torch.device(device):
            # Slot allocation and production intentionally use different streams.
            manager.begin(capture_baseline=version == 0, weight_version=version)
            with torch.cuda.stream(producer):
                for key, names in units.items():
                    manager.prefetch(key)
                    if version:
                        pointers.add(manager._prefetched[key][0][0].data_ptr())
                    torch.cuda._sleep(2_000_000)
                    converted = [(name, expected[name].to(device, non_blocking=True)) for name in names]
                    if version == 0:
                        # Capture must use the canonical checkpoint, not these bytes.
                        for _, tensor in converted:
                            tensor.reshape(-1).view(torch.uint8).bitwise_xor_(0x55)
                    assert manager.process(key, converted) == []
                    del converted
                    gc.collect()
                    pressure = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
                    assert len(manager._pending) <= 2
            result = manager.finish()
            assert manager.finish() is result
            del pressure
        for unit in manager._units.values():
            assert unit.snapshot.device.type == "cpu" and unit.snapshot.is_pinned()
            actual = unit.snapshot.numpy()
            for region in unit.regions:
                assert np.array_equal(
                    actual[region.offset : region.offset + region.nbytes], _raw(expected[region.name])
                )
        if version == 0:
            assert not result.delta
        else:
            assert len(pointers) == 2, "Two GPU slots must be reused across five units"
            total = sum(t.numel() * t.element_size() for t in expected.values())
            changed = sum(np.count_nonzero(_raw(previous[name]) ^ _raw(value)) for name, value in expected.items())
            assert (result.total_bytes, result.changed_bytes) == (total, changed)
            assert bool(result.delta) == (version != 1)
            _apply_receiver(receiver, receiver_dir, directory / f"delta-{version}", result, version, expected)
        try:
            manager.begin(capture_baseline=False, weight_version=version + 1)
        except RuntimeError as error:
            assert "uncommitted" in str(error)
        else:
            raise AssertionError("A prepared update must not be reused before publication commits")
        manager.commit()
        previous = expected
    return sum(unit.snapshot.numel() for unit in manager._units.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-path", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    spec = importlib.util.spec_from_file_location("qualified_delta_receiver", args.receiver_path)
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with tempfile.TemporaryDirectory(prefix="gpu-delta-manager-") as temporary:
        baseline_bytes = _test_versions(Path(temporary), device, receiver)
    print(
        json.dumps(
            {
                "status": "PASS",
                "versions": 4,
                "expert_units": 5,
                "baseline_bytes": baseline_bytes,
                "gpu_slots": 2,
                "baseline": "owner-local pinned CPU RAM",
                "receiver": str(args.receiver_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
