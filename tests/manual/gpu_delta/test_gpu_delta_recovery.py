"""Native cached HF-base recovery publication, including target-buffer reuse.

python -m pytest tests/manual/gpu_delta/test_gpu_delta_recovery.py -q
Requires CUDA, nvCOMP >=5.3,<6 and the existing fixture's CPU codec oracles.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.manual.gpu_delta.test_gpu_delta_benchmark_fixture import _replay
from tests.manual.gpu_delta.test_gpu_delta_encoder import _snapshots

from miles.backends.training_utils.weight_update.protocols.gpu_delta.recovery import RecoveryPayload
from miles.utils.gpu_delta.encoder import GpuBatchEncoder
from miles.utils.gpu_delta.publication import CODECS, FRAME_BYTES, seal_publication


def _write_and_replay(payload, directory, plan, base, expected):
    metadata = dict(stream_id="recovery-test", base_version=0, target_version=7, plan_digest="b" * 64)
    shard = payload.write(directory, metadata, owner=0, plan=plan, base=base)
    descriptor = seal_publication(directory, [shard])
    assert descriptor["base_version"] == 0 and descriptor["target_version"] == 7
    state = {name: value.numpy().copy() for name, value in base.items()}
    # Existing independent CPU oracle authenticates the manifest/payload and
    # replays omitted frames, matrix XOR and full raw targets from the HF base.
    _replay(descriptor, state, payload.codec)
    for name in state:
        np.testing.assert_array_equal(state[name], expected[name])
    manifest = json.loads(Path(descriptor["manifest_path"]).read_bytes())
    assert manifest["frame_bytes"] == FRAME_BYTES
    return {entry["name"]: (directory / entry["name"]).read_bytes() for entry in manifest["files"]}


@pytest.mark.parametrize("codec", CODECS)
def test_cached_recovery_payload_survives_target_and_encoder_reuse(tmp_path, monkeypatch, codec):
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "0")
    old, current, inputs = _snapshots()
    names = [f"tensor-{i}" for i in range(len(inputs))]
    base = {name: pair[0] for name, pair in zip(names, inputs, strict=True)}
    target = {name: pair[1] for name, pair in zip(names, inputs, strict=True)}
    expected = dict(zip(names, current, strict=True))
    pointers = {name: value.data_ptr() for name, value in target.items()}
    matrices, raw = names[:4], names[4:]
    batches = [matrices[:2], matrices[2:]]
    plan = {
        name: {"dtype": "U8", "shape": [1, value.numel()] if name in matrices else [value.numel()], "views": []}
        for name, value in base.items()
    }
    encoder = GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()), codec=codec)
    cached = RecoveryPayload.encode(encoder, codec, batches, raw, base, target)
    assert encoder.stream.query()
    saved = [bytes(encoded[1]) for _, encoded in cached.matrices]
    first = _write_and_replay(cached, tmp_path / "first", plan, base, expected)

    # Reuse the same canonical target storage and encoder for a later step.
    # Recovery must still describe version 7, including its copied raw targets.
    for i, value in enumerate(target.values()):
        value.fill_(i + 41)
    later = RecoveryPayload.encode(encoder, codec, batches, raw, base, target)
    assert encoder.stream.query()
    assert later.raw != cached.raw
    del later
    assert [bytes(encoded[1]) for _, encoded in cached.matrices] == saved
    assert _write_and_replay(cached, tmp_path / "reused", plan, base, expected) == first
    for name, original in zip(names, old, strict=True):
        np.testing.assert_array_equal(base[name].numpy(), original)
        assert target[name].data_ptr() == pointers[name]
