"""Focused GPU coverage for streamed optimizer main-param initialization."""

import os
from types import SimpleNamespace

import torch
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _resize, _Stager

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["miles-plugin"],
)


def test_bucketwise_main_initialization_preserves_bytes_and_releases_cuda_storage(tmp_path):
    model_param = torch.nn.Parameter(torch.arange(600_000, dtype=torch.float32, device="cuda").to(torch.bfloat16))
    main_param = torch.empty_like(model_param, dtype=torch.float32)
    _resize(main_param, 0)
    bucket = _Bucket(
        str(tmp_path / "bucket.bin"),
        entries=[_Entry(model_param, main_param, 0)],
        adam=None,
        stager=_Stager(1024 * 1024),
        dtypes={segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")},
    )
    param_range = SimpleNamespace(start=0, end=model_param.numel(), size=model_param.numel())
    dist_opt = SimpleNamespace(
        model_fp32_groups=[],
        shard_fp32_groups=[],
        _get_model_param_range_map=lambda _param: {"param": param_range},
    )
    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt = dist_opt
    store.buckets = [bucket]

    nbytes = main_param.numel() * main_param.element_size()
    assert store.initialize_main_from_model_params() == nbytes
    assert main_param.untyped_storage().nbytes() == 0
    restored = torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, 0)), dtype=torch.float32)
    torch.testing.assert_close(restored, model_param.float().cpu(), atol=0, rtol=0)
    os.close(bucket.fd)
