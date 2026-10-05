"""Native nvCOMP compression/CPU interoperability on CUDA; no model required.

python tests/manual/gpu_delta/test_gpu_delta_nvcomp.py
Requires nvCOMP >=5.3,<6, zstandard, python-snappy and lz4. CPU decoding here is a
byte-exact test oracle, never a production fallback.
"""

import json

import lz4.block
import numpy as np
import snappy
import torch
import zstandard

from miles.utils.gpu_delta_nvcomp import NvcompCompressor


def qualify(codec, device):
    compressor = NvcompCompressor(codec, device)
    stream = torch.cuda.Stream(device=device)
    random = np.random.default_rng(20261002)
    raw = []
    for size in (1, 15, 65536, (1 << 20) - 1, 1 << 20):
        raw.extend([np.zeros(size, dtype=np.uint8), random.integers(0, 256, size, dtype=np.uint8)])
    with torch.cuda.stream(stream):
        inputs = [torch.from_numpy(item).to(device) for item in raw]
        first = compressor.compress(inputs, stream)
        second = compressor.compress(inputs[::-1], stream)
        ready = torch.cuda.Event()
        ready.record(stream)
    del inputs
    ready.synchronize()
    for batch, reference in ((first, raw), (second, raw[::-1])):
        sizes, statuses = batch.sizes.cpu().tolist(), batch.statuses.cpu().tolist()
        assert statuses == [0] * len(reference), statuses
        for output, size, expected in zip(batch.outputs, sizes, reference, strict=True):
            assert 0 < size <= output.numel()
            encoded = output[:size].cpu().numpy().tobytes()
            if codec == "lz4":
                decoded = lz4.block.decompress(encoded, uncompressed_size=expected.nbytes)
            elif codec == "snappy":
                decoded = snappy.decompress(encoded)
            else:
                decoded = zstandard.ZstdDecompressor().decompress(encoded)
            assert decoded == expected.tobytes()
    assert compressor.compress([], stream).outputs == []
    # Warm allocations, then prove submission does not synchronize queued GPU
    # work. No timing claim is made from this deliberately delayed test stream.
    with torch.cuda.stream(stream):
        value = torch.zeros(65536, dtype=torch.uint8, device=device)
        warm = compressor.compress([value], stream)
        stream.synchronize()
        torch.cuda._sleep(500_000_000)
        delay = torch.cuda.Event()
        delay.record(stream)
        pending = compressor.compress([value], stream)
        returned_before_delay = not delay.query()
        stream.synchronize()
    assert returned_before_delay, "compression submission synchronized its CUDA stream"
    assert pending.statuses.item() == 0 and warm.statuses.item() == 0
    return {"codec": codec, "nvcomp": compressor.version, "roundtrip_frames": 2 * len(raw), "async": True}


if __name__ == "__main__":
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    print(json.dumps({"status": "PASS", "results": [qualify(c, device) for c in ("zstd", "snappy", "lz4")]}))
