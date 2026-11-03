"""GPU-delta baseline admission and native codec ABI contracts."""

import ctypes
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.utils.gpu_delta import nvcomp
from miles.utils.gpu_delta.admission import validate_gpu_delta_args


def _args(tmp_path, **overrides):
    return Namespace(
        **(
            {
                "update_weight_transfer_mode": "gpu-delta",
                "update_weight_delta_encoding": "xor",
                "colocate": False,
                "train_backend": "megatron",
                "megatron_to_hf_mode": "raw",
                "lora_rank": 0,
                "prefill_num_servers": None,
                "pause_generation_mode": "retract",
                "check_weight_update_equal": False,
                "update_weight_disk_dir": str(tmp_path / "updates"),
                "hf_checkpoint": str(tmp_path),
                "mtp_num_layers": None,
                "sglang_load_format": "auto",
                "sglang_speculative_draft_load_format": None,
            }
            | overrides
        )
    )


@pytest.mark.parametrize(
    "field",
    [
        "sglang_load_format",
        "sglang_speculative_draft_load_format",
        "eval_sglang_load_format",
        "eval_sglang_speculative_draft_load_format",
    ],
)
def test_dummy_target_or_draft_cannot_establish_delta_baseline(tmp_path, field):
    with pytest.raises(ValueError, match="checkpoint|safetensors"):
        validate_gpu_delta_args(_args(tmp_path, **{field: "dummy"}))


def test_trainable_draft_and_invalid_initial_codec_fail_before_startup(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="static receiver MTP"):
        validate_gpu_delta_args(_args(tmp_path, mtp_num_layers=1))
    monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", "invalid")
    with pytest.raises(ValueError, match="CODEC"):
        validate_gpu_delta_args(_args(tmp_path))


@pytest.mark.parametrize(
    "codec,symbol,options",
    [("snappy", "Snappy", nvcomp._Options), ("zstd", "Zstd", nvcomp._Options), ("lz4", "LZ4", nvcomp._LZ4Options)],
)
def test_nvcomp6_abi_and_stream_bound_workspace_queries(monkeypatch, codec, symbol, options):
    calls, libraries = [], []

    class Function:
        def __init__(self, suffix):
            self.suffix = suffix

        def __call__(self, *args):
            calls.append((self.suffix, args))
            if self.suffix == "GetRequiredAlignments":
                alignments = ctypes.cast(args[1], ctypes.POINTER(nvcomp._Alignments)).contents
                alignments.input, alignments.output, alignments.temp = 1, 16, 256
            elif self.suffix == "GetTempSize":
                ctypes.cast(args[3], ctypes.POINTER(ctypes.c_size_t))[0] = 4096
            elif self.suffix == "GetMaxOutputChunkSize":
                ctypes.cast(args[2], ctypes.POINTER(ctypes.c_size_t))[0] = args[0] + 16
            return 0

    functions = {
        suffix: Function(suffix)
        for suffix in ("GetRequiredAlignments", "GetTempSize", "GetMaxOutputChunkSize", "Async")
    }
    library = SimpleNamespace(
        **{"nvcompBatched" + symbol + "Compress" + suffix: value for suffix, value in functions.items()}
    )
    monkeypatch.setattr(
        nvcomp.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(version="6.0.0.12", locate_file=lambda path: path),
    )
    monkeypatch.setattr(nvcomp.ctypes, "CDLL", lambda path: libraries.append(path) or library)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    compressor = nvcomp.NvcompCompressor(codec, torch.device("cuda", 0))
    assert libraries == ["nvidia/libnvcomp/lib64/libnvcomp.so.6"]
    assert ctypes.sizeof(options) == 64 and bytes(compressor._options) == bytes(64)
    assert ctypes.alignment(options) == (ctypes.alignment(ctypes.c_int) if codec == "lz4" else 1)
    assert ctypes.sizeof(nvcomp._Alignments) == 24 and ctypes.alignment(nvcomp._Alignments) == 8
    assert [nvcomp._Alignments.input.offset, nvcomp._Alignments.output.offset, nvcomp._Alignments.temp.offset] == [
        0,
        8,
        16,
    ]
    if codec == "lz4":
        assert options.data_type.offset == 0 and options.bitshuffle_mode.offset == 4 and options.reserved.offset == 8
        assert compressor._options.data_type == 0  # NVCOMP_TYPE_CHAR, not FLOAT8_E4M3.
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    assert functions["GetRequiredAlignments"].argtypes == [options, ctypes.POINTER(nvcomp._Alignments)]
    assert functions["GetTempSize"].argtypes == [size, size, options, ctypes.POINTER(size), size, pointer]
    assert functions["GetMaxOutputChunkSize"].argtypes == [size, options, ctypes.POINTER(size)]
    assert functions["Async"].argtypes == [
        pointer,
        pointer,
        size,
        size,
        pointer,
        size,
        pointer,
        pointer,
        options,
        pointer,
        pointer,
    ]
    assert all(function.restype is ctypes.c_int for function in functions.values())
    first, second = SimpleNamespace(cuda_stream=17), SimpleNamespace(cuda_stream=23)
    assert compressor._temporary_bytes(2, 1024, 1536, first) == 4096
    assert compressor._temporary_bytes(2, 1024, 1536, first) == 4096
    assert compressor._temporary_bytes(2, 1024, 1536, second) == 4096
    assert compressor._temporary_bytes(3, 1024, 2048, second) == 4096
    queries = [args for suffix, args in calls if suffix == "GetTempSize"]
    assert [(args[0], args[1], args[4], args[5]) for args in queries] == [
        (2, 1024, 1536, 17),
        (2, 1024, 1536, 23),
        (3, 1024, 2048, 23),
    ]
    assert compressor._output_bound(1024) == compressor._output_bound(1024) == 1040
    assert sum(suffix == "GetMaxOutputChunkSize" for suffix, _ in calls) == 1


@pytest.mark.parametrize("version", ["5.3.0.16", "7.0.0.0"])
def test_nvcomp_other_major_versions_fail_before_loading_library(monkeypatch, version):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(nvcomp.importlib.metadata, "distribution", lambda _: SimpleNamespace(version=version))
    monkeypatch.setattr(nvcomp.ctypes, "CDLL", lambda _: pytest.fail("loaded incompatible nvCOMP"))
    with pytest.raises(RuntimeError, match="64-bit nvCOMP 6 ABI"):
        nvcomp.NvcompCompressor("snappy", torch.device("cuda", 0))
