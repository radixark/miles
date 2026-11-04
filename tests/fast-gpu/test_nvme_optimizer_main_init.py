"""Focused GPU coverage for streamed optimizer main-param initialization and bucket writeback."""

import os
from types import SimpleNamespace

import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.fast.optimizers.nvme_setup_fixtures import require_evictable, resident_pages

from miles_plugins.optimizers.nvme_stream import (
    NVMeOptimizerStateStore,
    _Bucket,
    _Entry,
    _PageCacheWindow,
    _resize,
    _Stager,
)

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)
register_rocm_ci(est_time=30, suite="nightly-stage-c-2-gpu-mi350", labels=["miles-plugin"])


def _initialized_store(tmp_path):
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
    store._page_cache = None
    nbytes = main_param.numel() * main_param.element_size()
    assert store.initialize_main_from_model_params() == nbytes
    return bucket, model_param, main_param, nbytes


def test_bucketwise_main_initialization_preserves_bytes_and_releases_cuda_storage(tmp_path):
    bucket, model_param, main_param, nbytes = _initialized_store(tmp_path)

    assert main_param.untyped_storage().nbytes() == 0
    restored = torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, 0)), dtype=torch.float32)
    torch.testing.assert_close(restored, model_param.float().cpu(), atol=0, rtol=0)
    os.close(bucket.fd)


def test_uncapped_initialization_leaves_nothing_cached(tmp_path):
    require_evictable(tmp_path)
    bucket, *_ = _initialized_store(tmp_path)

    assert resident_pages(bucket.fd) == 0
    os.close(bucket.fd)


@torch.no_grad()
def test_capped_page_cache_drops_older_buckets_and_matches_a_resident_adam(tmp_path):
    """The cap drops older buckets once written back; the state stays bit-exact through it."""
    require_evictable(tmp_path)
    params = [torch.nn.Parameter(torch.linspace(-1, 1, 1_048_576, device="cuda") * (i + 1)) for i in range(2)]
    references = [torch.nn.Parameter(param.detach().clone()) for param in params]
    resident_adams = [torch.optim.Adam([reference], lr=1e-3) for reference in references]
    stager = _Stager(1024 * 1024)
    buckets = [
        _Bucket(
            str(tmp_path / f"bucket{index}.bin"),
            entries=[_Entry(param, param, 0)],
            adam=torch.optim.Adam([param], lr=1e-3),
            stager=stager,
            dtypes={segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")},
        )
        for index, param in enumerate(params)
    ]
    window = _PageCacheWindow(cap_bytes=buckets[0].nbytes)
    try:
        for bucket in buckets:
            bucket.flush(segments=("main",), page_cache=window)
        for step in range(2):
            for bucket, param, reference, resident_adam in zip(
                buckets, params, references, resident_adams, strict=True
            ):
                bucket.fetch()
                gradient = torch.full_like(reference, 0.1 * (step + 1))
                param.grad = gradient.clone()
                reference.grad = gradient
                bucket.adam.step()
                resident_adam.step()
                bucket.flush(page_cache=window)
                assert param.untyped_storage().nbytes() == 0
            # Flushing the second bucket pushed the first past the cap, so it is written back and dropped.
            assert resident_pages(buckets[0].fd) == 0
            for bucket, param, reference, resident_adam in zip(
                buckets, params, references, resident_adams, strict=True
            ):
                bucket.fetch()
                torch.testing.assert_close(param, reference, atol=0, rtol=0)
                for segment in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        bucket.adam.state[param][segment], resident_adam.state[reference][segment], atol=0, rtol=0
                    )
                bucket.flush(page_cache=window)
    finally:
        window.drain()
        for bucket in buckets:
            os.close(bucket.fd)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
