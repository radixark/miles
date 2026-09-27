"""Real local GDS manager qualification against the existing CPU receiver.

Run with --directory on the target local NVMe filesystem and --receiver-path
pointing to SGLang's weight_sync/local_checkpoint.py. A temporary child directory
holds all fixtures. Direct reads and writes are mandatory: missing cuFile support
or enabled compatibility mode fails the test. No I/O test doubles are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.nvfp4_gds import Nvfp4GdsDelta
from miles.utils.disk_delta import checkpoint_tensor_location, make_tensor_reader


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
    receiver_dir = directory / "receiver"
    shutil.copytree(source, receiver_dir)
    (receiver_dir / receiver.SYNC_DIR).mkdir()
    receiver._write_applied_version(str(receiver_dir), 0)
    manager = Nvfp4GdsDelta(
        str(source), str(directory / "baselines"), device, quantization_config={"quant_method": "nvfp4"}
    )
    previous = initial
    try:
        for version in range(4):
            expected = _version(initial, version)
            gpu = {name: tensor.to(device) for name, tensor in expected.items()}
            if version == 0:
                # Capture must load canonical disk bytes, not the passed tensors.
                for tensor in gpu.values():
                    tensor.reshape(-1).view(torch.uint8).bitwise_xor_(0x55)
            manager.begin(capture_baseline=version == 0, weight_version=version)
            for key, names in units.items():
                manager.prefetch(key)
                assert manager.process(key, [(name, gpu[name]) for name in names]) == []
            result = manager.finish()
            manager.commit()
            if version == 0:
                assert not result.delta
            else:
                total = sum(t.numel() * t.element_size() for t in expected.values())
                changed = sum(np.count_nonzero(_raw(previous[name]) ^ _raw(value)) for name, value in expected.items())
                assert (result.total_bytes, result.changed_bytes) == (total, changed)
                assert bool(result.delta) == (version != 1)
                _apply_receiver(receiver, receiver_dir, directory / f"delta-{version}", result, version, expected)
            previous = expected
    finally:
        manager._executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path, required=True, help="Existing directory on the target local NVMe mount"
    )
    parser.add_argument("--receiver-path", type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("--directory must already exist on the target local NVMe filesystem")
    spec = importlib.util.spec_from_file_location("qualified_delta_receiver", args.receiver_path)
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with tempfile.TemporaryDirectory(prefix="gpu-delta-manager-", dir=args.directory) as temporary:
        _test_versions(Path(temporary), device, receiver)
    print(
        json.dumps(
            {
                "status": "PASS",
                "versions": 4,
                "expert_units": 5,
                "io_backend": "strict cuFile direct read/write",
                "directory": str(args.directory.resolve()),
                "receiver": str(args.receiver_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
