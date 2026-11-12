"""CPU nvCOMP ABI, codec admission, outer payload integrity and prepared metadata ownership."""

import ctypes
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zstandard as zstd

from miles.utils.gpu_delta import codec, memory, nvcomp
from miles.utils.gpu_delta.host_worker import HostPayloadPool, validate_codec

P, S = ctypes.c_void_p, ctypes.c_size_t


def _frame_table(batches):
    rows = [row for batch in batches for row in batch]
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T.copy()
    return table, list(map(len, batches))


@pytest.fixture
def fake_nvcomp(monkeypatch):
    """A recording nvCOMP 6 stand-in for both compression and decompression symbols."""
    fake = SimpleNamespace(calls=[], paths=[], functions={})
    fake.distribution = SimpleNamespace(version="6.0.0.12", locate_file=lambda path: path)

    class Function:
        def __init__(self, suffix):
            self.suffix = suffix

        def __call__(self, *args):
            fake.calls.append((self.suffix, args))
            if self.suffix == "GetRequiredAlignments":
                ctypes.cast(args[1], ctypes.POINTER(nvcomp._Alignments))[0] = nvcomp._Alignments(1, 16, 256)
            elif self.suffix == "GetTempSize":
                ctypes.cast(args[3], ctypes.POINTER(S))[0] = 4096
            elif self.suffix == "GetMaxOutputChunkSize":
                ctypes.cast(args[2], ctypes.POINTER(S))[0] = args[0] + 16
            return 0

    class Library:
        def __getattr__(self, name):
            for suffix in ("GetRequiredAlignments", "GetTempSize", "GetMaxOutputChunkSize", "Async"):
                if name.endswith(suffix):
                    return fake.functions.setdefault(name, Function(suffix))
            raise AttributeError(name)

    monkeypatch.setattr(nvcomp.importlib.metadata, "distribution", lambda _: fake.distribution)
    monkeypatch.setattr(nvcomp.ctypes, "CDLL", lambda path: fake.paths.append(path) or Library())
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    return fake


def _assert_bound_abi(fake, prefix, options, **signatures):
    """Plain fake callables never enforce argtypes; restate the nvCOMP 6 header signatures."""
    expected = {
        "GetTempSize": [S, S, options, ctypes.POINTER(S), S, P],
        "GetRequiredAlignments": [options, ctypes.POINTER(nvcomp._Alignments)],
        **signatures,
    }
    assert {name: (function.argtypes, function.restype) for name, function in fake.functions.items()} == {
        prefix + suffix: (argtypes, ctypes.c_int) for suffix, argtypes in expected.items()
    }


@pytest.mark.parametrize(
    "name,symbol,options",
    [
        ("snappy", "Snappy", nvcomp._Options),
        ("zstd", "Zstd", nvcomp._Options),
        ("lz4", "LZ4", nvcomp._LZ4Options),
        ("bitcomp", "Bitcomp", nvcomp._BitcompOptions),
    ],
)
def test_compressor_abi_and_stream_bound_workspace_queries(fake_nvcomp, name, symbol, options):
    compressor = nvcomp.NvcompCompressor(name, torch.device("cuda", 0))
    assert fake_nvcomp.paths == ["nvidia/libnvcomp/lib64/libnvcomp.so.6"]
    assert ctypes.sizeof(options) == 64
    if name == "bitcomp":
        assert ctypes.alignment(options) == 8
        assert [options.algorithm.offset, options.data_type.offset, options.delta.offset, options.mode.offset] == [
            0,
            4,
            8,
            16,
        ]
        assert compressor._options.algorithm == 1 and compressor._options.data_type == 1
        assert compressor._options.mode == 0 and compressor._options.delta == 0.0
        assert bytes(compressor._options)[20:] == bytes(44)
    else:
        assert bytes(compressor._options) == bytes(64)
        assert ctypes.alignment(options) == (ctypes.alignment(ctypes.c_int) if name == "lz4" else 1)
    if name == "lz4":
        assert options.data_type.offset == 0 and options.bitshuffle_mode.offset == 4 and options.reserved.offset == 8
        assert compressor._options.data_type == 0  # NVCOMP_TYPE_CHAR, not FLOAT8_E4M3.
    assert ctypes.sizeof(nvcomp._Alignments) == 24 and ctypes.alignment(nvcomp._Alignments) == 8
    assert [nvcomp._Alignments.input.offset, nvcomp._Alignments.output.offset, nvcomp._Alignments.temp.offset] == [
        0,
        8,
        16,
    ]
    _assert_bound_abi(
        fake_nvcomp,
        f"nvcompBatched{symbol}Compress",
        options,
        GetMaxOutputChunkSize=[S, options, ctypes.POINTER(S)],
        Async=[P, P, S, S, P, S, P, P, options, P, P],
    )
    assert (compressor.alignments.input, compressor.alignments.output, compressor.alignments.temp) == (1, 16, 256)
    assert compressor.temporary_bytes((2, 1024, 1536), 17) == 4096
    assert compressor.temporary_bytes((2, 1024, 1536), 17) == 4096
    assert compressor.temporary_bytes((2, 1024, 1536), 23) == 4096
    assert compressor.temporary_bytes((3, 1024, 2048), 23) == 4096
    queries = [args for suffix, args in fake_nvcomp.calls if suffix == "GetTempSize"]
    assert [(args[0], args[1], args[4], args[5]) for args in queries] == [
        (2, 1024, 1536, 17),
        (2, 1024, 1536, 23),
        (3, 1024, 2048, 23),
    ]
    assert compressor._output_bound(1024) == compressor._output_bound(1024) == 1040
    assert sum(suffix == "GetMaxOutputChunkSize" for suffix, _ in fake_nvcomp.calls) == 1


@pytest.mark.parametrize(
    "name,algorithm,capability,hardware,sorting",
    [
        ("snappy", "Snappy", (10, 0), True, "0"),
        ("snappy", "Snappy", (10, 0), True, "1"),
        ("lz4", "LZ4", (10, 3), True, "0"),
        ("lz4", "LZ4", (10, 3), True, "1"),
        # Requested sorting must stay off outside DE.
        ("snappy", "Snappy", (9, 0), False, "1"),
        ("lz4", "LZ4", (9, 0), False, "1"),
        ("bitcomp", "Bitcomp", (10, 3), False, "1"),
        ("bitcomp", "Bitcomp", (9, 0), False, "1"),
    ],
)
def test_decoder_abi_and_frozen_hardware_options(
    fake_nvcomp, monkeypatch, request, name, algorithm, capability, hardware, sorting
):
    codec.decoder_backend.cache_clear()
    request.addfinalizer(codec.decoder_backend.cache_clear)
    # Match nvCOMP 6 snappy.h, lz4.h and bitcomp.h: 64-byte options led by the backend
    # enum and the DE sorting flag (reserved in Bitcomp); zero later bytes select LZ4
    # CHAR without bitshuffle.
    options = codec._DecompressOptions
    assert ctypes.sizeof(options) == 64 and ctypes.alignment(options) == ctypes.alignment(ctypes.c_int)
    assert options.backend.offset == 0 and options.sort_before_hw_decompress.offset == 4
    assert options.reserved.offset == 8
    monkeypatch.setenv("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", sorting)
    device_queries = []
    attributes = {136: {"Snappy": 2, "LZ4": 4, "Bitcomp": 0}[algorithm], 137: 4 << 20}

    def device_attribute(attribute, device):
        device_queries.append((attribute, device))
        return 0, attributes[attribute]

    monkeypatch.setitem(
        sys.modules,
        "miles.utils.gpu_delta.memory",
        SimpleNamespace(
            _check=memory._check,
            _driver=lambda: SimpleNamespace(
                CUdevice_attribute=SimpleNamespace(
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK=136,
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH=137,
                ),
                CUmemDecompressAlgorithm=SimpleNamespace(
                    CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY=2,
                    CU_MEM_DECOMPRESS_ALGORITHM_LZ4=4,
                ),
                cuDeviceGetAttribute=device_attribute,
            ),
        ),
    )
    capability_queries = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability_queries.append(device) or capability,
    )
    for allocator_api in (
        "get_allocator_backend",
        "_snapshot",
        "_set_allocator_settings",
    ):
        monkeypatch.setattr(
            torch.cuda.memory,
            allocator_api,
            lambda *a: pytest.fail("global allocator access"),
        )
    decoder = codec.NvcompDecoder(torch.device("cuda", 0), name)
    assert fake_nvcomp.paths == ["nvidia/libnvcomp/lib64/libnvcomp.so.6"]
    assert decoder.inner_codec == name
    assert decoder.backend == ("hardware" if hardware else "cuda")
    assert codec.decoder_backend(decoder.device, name) == decoder.backend
    assert codec.decoder_backend(decoder.device, name) == decoder.backend
    assert capability_queries == ([decoder.device] if name != "bitcomp" else [])
    assert decoder.maximum_chunk_bytes == ((4 << 20) if hardware else codec._INT64_MAX)
    assert decoder._options.backend == (1 if hardware else 2)
    assert decoder._options.sort_before_hw_decompress == (int(sorting) if hardware else 0)
    assert bytes(decoder._options)[8:] == bytes(56)
    _assert_bound_abi(
        fake_nvcomp,
        f"nvcompBatched{algorithm}Decompress",
        options,
        Async=[P, P, P, P, S, P, S, P, options, P, P],
    )
    assert (decoder.alignments.input, decoder.alignments.output, decoder.alignments.temp) == (1, 16, 256)
    # Alignment and workspace queries must receive the same admitted options.
    assert decoder.temporary_bytes((1, 64, 64), 7) == decoder.temporary_bytes((1, 64, 64), 7) == 4096
    [(first, align), (second, query)] = fake_nvcomp.calls
    assert (first, second) == ("GetRequiredAlignments", "GetTempSize")
    assert align[0] is query[2] is decoder._options
    assert device_queries == ([(136, 0), (137, 0)] if hardware else [])
    # Allocation capability alone does not prove this codec exists. Frame
    # lengths are checked against the cached device limit during preparation.
    if hardware:
        attributes[136] = 0
        with pytest.raises(RuntimeError, match="no hardware"):
            codec.NvcompDecoder(decoder.device, name)
        attributes[136] = {"Snappy": 2, "LZ4": 4}[algorithm]
        attributes[137] = 1 << 20
        assert codec.NvcompDecoder(decoder.device, name).maximum_chunk_bytes == 1 << 20
        attributes[137] = 0
        with pytest.raises(RuntimeError, match="positive hardware decompression limit"):
            codec.NvcompDecoder(decoder.device, name)


@pytest.mark.parametrize("version", ["5.3.0.16", "7.0.0.0"])
@pytest.mark.parametrize(
    "create",
    [lambda device: nvcomp.NvcompCompressor("snappy", device), lambda device: codec.NvcompDecoder(device, "snappy")],
    ids=["compressor", "decoder"],
)
def test_other_nvcomp_major_versions_fail_before_loading_library(fake_nvcomp, create, version):
    fake_nvcomp.distribution.version = version
    with pytest.raises(RuntimeError, match="64-bit nvCOMP 6 ABI"):
        create(torch.device("cuda", 0))
    assert fake_nvcomp.paths == []


def test_workspace_query_cache_uses_geometry_and_stream_and_preserves_failure():
    # Exercise the real shared query/cache code without a CUDA device guard,
    # allocation or library; prepare_batches already selects the device.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder._label = "snappy/hardware"
    decoder._options = codec._DecompressOptions()
    decoder._temporary_sizes = {}
    calls, fail = [], True

    def query(count, maximum, options, size, total, stream):
        calls.append((count, maximum, total, stream))
        if total == 31 and fail:
            return 1
        # Zero is also a valid cache entry; it must not trigger another query.
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    decoder._temporary = query

    assert decoder.temporary_bytes((0, 0, 0), 7) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert calls == [(2, 20, 30, 7)]
    # Same count and maximum, different total: the native query must run again.
    with pytest.raises(RuntimeError, match="status=1"):
        decoder.temporary_bytes((2, 20, 31), 7)
    fail = False
    assert decoder.temporary_bytes((2, 20, 31), 7) == 0
    assert decoder.temporary_bytes((2, 20, 31), 7) == 0
    # Query the actual DE stream once, then retain its own zero-sized cache entry.
    assert decoder.temporary_bytes((2, 20, 30), 9) == 0
    assert decoder.temporary_bytes((2, 20, 30), 9) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert calls == [(2, 20, 30, 7), (2, 20, 31, 7), (2, 20, 31, 7), (2, 20, 30, 9)]


def test_prepared_stream_scratch_and_slot_status_ownership(monkeypatch):
    # Real tensor views/pointers exercise nonzero, non-aligned native workspace
    # requirements without a device. Native tests cover actual DE and ring reuse.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.hardware = True
    decoder.alignments = nvcomp._Alignments(16, 16, 16)
    decoder._options = codec._DecompressOptions()
    decoder.temporary_bytes = lambda geometry, stream: geometry[0] * 13
    recorded = []
    # DE outputs were admitted by device_empty; binding queries no capability.
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.gpu_delta.memory",
        SimpleNamespace(
            device_empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
            record_device_stream=lambda tensor, stream: recorded.append(
                (tensor.untyped_storage().data_ptr(), stream.cuda_stream)
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    stream = SimpleNamespace(device=decoder.device, cuda_stream=1)
    counts = [1, 2, 0, 3] * 3
    workspace = decoder._allocate_workspace([(count, 32, count * 32) for count in counts], stream.cuda_stream)
    metadata = torch.zeros((4, sum(counts)), dtype=torch.int64)
    host_metadata = metadata.clone()
    plan = codec.PreparedDecodePlan(
        decoder,
        counts,
        torch.empty(128, dtype=torch.uint8),
        workspace,
        host_metadata,
        metadata,
        stream,
        np.zeros(sum(counts), dtype=np.int64),
        [128, 128],
        [0, 0],
    )
    assert workspace.temporary.shape == (39,)
    assert workspace.temporary.data_ptr() % decoder.alignments.temp == 0
    assert workspace.statuses.shape == workspace.actual_sizes.shape == (2, 3)
    for index, batch in enumerate(plan.batches):
        slot = index % 2
        assert batch.stream is stream
        assert batch._arguments[-1] == stream.cuda_stream
        assert batch._arguments[5:7] == (
            workspace.temporary.data_ptr(),
            workspace.temporary.numel(),
        )
        assert batch.statuses.storage_offset() == slot * 3
        assert batch.actual_sizes.storage_offset() == slot * 3
        assert batch._arguments[3] == batch.actual_sizes.data_ptr()
        assert batch._arguments[9] == batch.statuses.data_ptr()
        if not counts[index]:
            batch.enqueue()  # Empty metadata must not call the native decoder.
    outputs = [torch.empty(128, dtype=torch.uint8) for _ in range(2)]
    assert plan.bind_outputs(outputs) is plan.batches
    consumer = SimpleNamespace(device=decoder.device, cuda_stream=2)
    plan.record_stream(consumer)
    with pytest.raises(ValueError, match="consumer stream/device mismatch"):
        plan.record_stream(SimpleNamespace(device=torch.device("cuda", 0)))
    for tensor in (
        metadata,
        workspace.temporary,
        workspace.statuses,
        workspace.actual_sizes,
        *outputs,
    ):
        assert {(tensor.untyped_storage().data_ptr(), s.cuda_stream) for s in (stream, consumer)} <= set(recorded)
    offset = 0
    for index, count in enumerate(counts):
        assert metadata[3, offset : offset + count].tolist() == [outputs[index % 2].data_ptr()] * count
        offset += count


def test_host_input_plans_reject_invalid_geometry_before_device_work(monkeypatch):
    # Native tests exercise successful slab construction/slot reuse. Keep the
    # vector arithmetic and output-lease failures here without a fake CUDA API.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.hardware = True
    decoder.inner_codec = "snappy"
    decoder.maximum_chunk_bytes = 4 << 20
    decoder.alignments = nvcomp._Alignments(16, 16, 16)
    table, counts = _frame_table([[(0, 20, 64, 0), (32, 17, 32, 128)]])
    geometry, bounds, remainders = decoder._frame_geometry(table, counts, 0, 512)
    assert (geometry, bounds, remainders) == (
        [(2, 64, 96)],
        [160, 0],
        [0, None],
    )
    for rows, error in (
        ([(496, 32, 64, 0)], "outside input"),
        ([(np.iinfo(np.int64).max, 32, 64, 0)], "outside input"),
        ([(0, 32, 64, np.iinfo(np.int64).max)], "Output arithmetic"),
        ([(1, 32, 64, 0)], "Misaligned encoded"),
        ([(0, 32, 64, 0), (32, 32, 64, 32)], "Overlapping"),
        ([(0, 32, 64, 0), (32, 32, 64, 65)], "Misaligned decoded"),
    ):
        with pytest.raises(ValueError, match=error):
            decoder._frame_geometry(*_frame_table([rows]), 0, 512)
    for invalid in (table.astype(np.int32), table[:, ::-1], table[:, :-1]):
        with pytest.raises(ValueError, match="contiguous int64"):
            decoder._frame_geometry(invalid, counts, 0, 512)
    maximum = decoder.maximum_chunk_bytes
    # A compressible full 4 MiB output is legal; either actual length exceeding
    # hardware's bound is not. This is independent of the inner algorithm.
    assert decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512)[1] == [maximum, 0]
    for encoded, decoded in ((maximum + 1, 64), (20, maximum + 1)):
        with pytest.raises(ValueError, match="device limit"):
            decoder._frame_geometry(*_frame_table([[(0, encoded, decoded, 0)]]), 0, 512)
    decoder.maximum_chunk_bytes = 1 << 20
    with pytest.raises(ValueError, match="device limit"):
        decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512)
    assert decoder._frame_geometry(*_frame_table([[], []]), 0, 512) == (
        [(0, 0, 0), (0, 0, 0)],
        [0, 0],
        [None, None],
    )

    plan = object.__new__(codec.PreparedDecodePlan)
    plan.decoder = decoder
    plan.output_bounds, plan.output_remainders = [128, 128], [0, 0]
    plan.decoded_slots = []
    # Every malformed output must fail before CUDA context, allocation or upload.
    monkeypatch.setattr(torch.cuda, "device", lambda _: pytest.fail("entered CUDA"))
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.gpu_delta.memory",
        SimpleNamespace(
            device_empty=lambda *a, **kw: pytest.fail("allocated device storage"),
            record_device_stream=lambda *a: pytest.fail("recorded device"),
        ),
    )
    with pytest.raises(ValueError, match="Decoder stream/device mismatch"):
        decoder.prepare_batches(
            table,
            counts,
            torch.empty(512, dtype=torch.uint8),
            SimpleNamespace(device=torch.device("cuda", 0)),
        )
    a, b = [torch.empty(256, dtype=torch.uint8) for _ in range(2)]
    for slots, error in (
        ((a, b[:64]), "Out-of-bounds"),
        ((a, b[1:]), "Misaligned decoded"),
        ((a, a[16:]), "must not overlap"),
        ((a,), "count differs"),
        ((a, b, a), "count differs"),
    ):
        with pytest.raises(ValueError, match=error):
            plan.bind_outputs(slots)
        assert not plan.decoded_slots


def test_cuda_input_packing_preserves_frame_bytes_and_alignment():
    # Reordered, duplicate and disjoint frames must share merged H2D spans
    # without copying large holes in the persistent host arena.
    inputs = np.array([160, 0, 16, 160, 1 << 20], dtype=np.int64)
    sizes = np.array([8, 7, 20, 8, 9], dtype=np.int64)
    offsets, ranges, capacity = codec._pack_inputs(inputs, sizes, 16)
    assert ranges == ((0, 0, 36), (160, 48, 8), (1 << 20, 64, 9))
    assert capacity == 73
    assert offsets.tolist() == [48, 0, 16, 48, 64]
    source = np.arange((1 << 20) + 9, dtype=np.uint8)
    destination = np.zeros(capacity, dtype=np.uint8)
    for start, offset, length in ranges:
        destination[offset : offset + length] = source[start : start + length]
    for start, offset, size in zip(inputs, offsets, sizes, strict=False):
        np.testing.assert_array_equal(destination[offset : offset + size], source[start : start + size])
    assert codec._pack_inputs(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 16)[1:] == ((), 0)


class TestOuterZstd(unittest.TestCase):
    def test_manifest_codec_and_frame_geometry_admission(self):
        admitted = dict(codec="snappy-zstd", frame_bytes=1 << 20)
        for codec_name in ("snappy-zstd", "lz4-zstd", "lz4", "bitcomp", "bitcomp-zstd"):
            for size in (1 << 16, 192 << 10, 1 << 19, 1 << 20, 2 << 20, 4 << 20):
                with self.subTest(codec=codec_name, size=size):
                    validate_codec(admitted | {"codec": codec_name, "frame_bytes": size})
        for patch_value in (
            {"codec": "zstd"},
            {"codec": None},
            {"frame_bytes": 0},
            {"frame_bytes": -1},
            {"frame_bytes": (4 << 20) + 1},
            {"frame_bytes": 1.5},
            {"frame_bytes": True},
        ):
            with (
                self.subTest(patch=patch_value),
                self.assertRaisesRegex(ValueError, "GPU delta requires"),
            ):
                validate_codec(admitted | patch_value)

    def test_gpu_outer_chunks_decode_directly_to_one_destination(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for known_size in (False, True):
            payload, chunks = bytearray(), []
            for index, value in enumerate(values):
                payload.extend(bytes((-len(payload)) % 16))
                compressed = zstd.ZstdCompressor(level=1, write_content_size=known_size).compress(value)
                chunks.append(
                    dict(
                        encoded_offset=len(payload),
                        encoded_bytes=len(compressed),
                        decoded_offset=index * (1 << 20),
                        decoded_bytes=len(value),
                    )
                )
                payload.extend(compressed)
            target = bytearray(sum(map(len, values)))
            pool.executor.submit(
                pool.decode_zstd,
                zstd.BufferWithSegments(
                    payload,
                    np.asarray(
                        [[chunk["encoded_offset"], chunk["encoded_bytes"]] for chunk in chunks],
                        dtype=np.uint64,
                    ),
                ),
                [chunk["decoded_bytes"] for chunk in chunks],
                [chunk["decoded_offset"] for chunk in chunks],
                memoryview(target),
            ).result()
            self.assertEqual(target, b"".join(values))

    def test_invalid_frames_fail_without_publishing_a_decoded_result(self):
        raw = bytes(range(64)) * 128
        encoded = zstd.ZstdCompressor(write_checksum=True).compress(raw)
        corrupt = bytearray(encoded)
        corrupt[-1] ^= 1
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for payload, size in (
            (encoded, len(raw) - 1),
            (encoded, len(raw) + 1),
            (encoded[:-1], len(raw)),
            (corrupt, len(raw)),
            (b"not-zstd", len(raw)),
        ):
            target = bytearray(2 * len(raw) + 1)
            with self.subTest(size=size, payload=bytes(payload[:8])):
                with self.assertRaises(zstd.ZstdError):
                    pool.executor.submit(
                        pool.decode_zstd,
                        zstd.BufferWithSegments(
                            encoded + bytes(payload),
                            np.asarray(
                                [[0, len(encoded)], [len(encoded), len(payload)]],
                                dtype=np.uint64,
                            ),
                        ),
                        [len(raw), size],
                        [0, len(raw)],
                        memoryview(target),
                    ).result()
                self.assertEqual(target, bytes(len(target)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
