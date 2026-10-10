"""The streamed step builds fp32 grads per bucket and must match Megatron's eager path bit for bit.

Megatron copies every shard's BF16 grad to fp32 up front, clips the fp32 grads in place and steps;
the NVMe store converts one bucket at a time and clips it with the same clip_grad_by_total_norm_fp32
and total norm, and reports the grad norm from the BF16 shards. Same inputs, same Adam, so the
resulting main params (on disk and copied into the param buffer) and the norm must agree.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _resize, _Stager

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

SIZES = (1000, 3000, 500)
LR = 1e-3
DTYPES = {segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")}


def _adam(params):
    from megatron.core.optimizer import Adam

    return Adam([{"params": params, "lr": LR, "weight_decay": 0.0}], adam_w_mode=True)


def _setup(tmp_path, clip_grad: float):
    torch.manual_seed(0)
    model_params, entries, ranges, offset = [], [], {}, 0
    param_data = torch.zeros(sum(SIZES), dtype=torch.bfloat16, device="cuda")
    for numel in SIZES:
        param = torch.nn.Parameter(torch.randn(numel, device="cuda").to(torch.bfloat16))
        param.main_grad = torch.randn(numel, device="cuda").to(torch.bfloat16)
        main = torch.empty(numel, dtype=torch.float32, device="cuda")
        main.tensor_model_parallel = True  # a unique shard, so the norm counts it without TP groups
        _resize(main, 0)
        ranges[param] = {
            "param": SimpleNamespace(start=0, end=numel, size=numel),
            "gbuf_world_in_bucket": SimpleNamespace(start=offset, end=offset + numel, size=numel),
        }
        offset += numel
        model_params.append(param)
        entries.append(_Entry(param, main, 0))
    dist_opt = SimpleNamespace(
        model_fp32_groups=[],
        shard_fp32_groups=[],
        optimizer=SimpleNamespace(param_groups=[{"lr": LR, "weight_decay": 0.0}]),
        model_param_gbuf_map={p: (0, torch.bfloat16, 0) for p in model_params},
        buffers=[SimpleNamespace(buckets=[SimpleNamespace(param_data=param_data)])],
        _get_model_param_range_map=lambda p: ranges[p],
        config=SimpleNamespace(clip_grad=clip_grad),
    )
    stager = _Stager(64 * 1024)
    # Two buckets, so the per-bucket attach/detach runs more than once.
    groups = [entries[:2], entries[2:]]
    buckets = [
        _Bucket(str(tmp_path / f"bucket{i}.bin"), group, _adam([e.main_param for e in group]), stager, DTYPES)
        for i, group in enumerate(groups)
    ]
    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt, store.buckets = dist_opt, buckets
    store._fp32_adam, store._fp32_group_indices = None, []
    store._grad_norm = None
    store.initialize_main_from_model_params()
    return store, model_params, entries, param_data


def _eager_reference(model_params, grad_norm: torch.Tensor | float, clip_grad: float) -> list[torch.Tensor]:
    """Megatron's path: fp32 grads for every shard, clip_grad_by_total_norm_fp32, then Adam."""
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

    mains = [p.detach().float().clone() for p in model_params]
    for main, param in zip(mains, model_params, strict=True):
        main.grad = param.main_grad.float()
    if clip_grad > 0.0:
        clip_grad_by_total_norm_fp32(mains, clip_grad, grad_norm)
    _adam(mains).step()
    return mains


def _read_main(bucket: _Bucket, index: int, numel: int) -> torch.Tensor:
    raw = os.pread(bucket.fd, numel * 4, bucket.offsets["main"][index])
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


@pytest.mark.parametrize("clip_grad", [1.0, 0.0, 1.0e6])
def test_streamed_grads_match_eager_fp32_grads(tmp_path, clip_grad):
    store, model_params, entries, param_data = _setup(tmp_path, clip_grad)

    (norm,) = store.grads_for_norm()
    expected_norm = torch.linalg.vector_norm(torch.cat([p.main_grad.float() for p in model_params]))
    torch.testing.assert_close(norm, expected_norm.reshape(1), rtol=1e-6, atol=0)

    from megatron.core.optimizer.clip_grads import multi_tensor_scale_tensor_impl

    # The type Megatron's get_grad_norm_fp32 returns, which selects clip_grad_by_total_norm_fp32's path.
    grad_norm = norm if multi_tensor_scale_tensor_impl is not None else float(norm)
    store.record_grad_norm(grad_norm)
    assert store.step()
    reference = _eager_reference(model_params, grad_norm, clip_grad)

    offset = 0
    reference_of = {id(entry.main_param): ref for entry, ref in zip(entries, reference, strict=True)}
    for bucket in store.buckets:
        for index, entry in enumerate(bucket.entries):
            numel = entry.main_param.numel()
            ref = reference_of[id(entry.main_param)]
            torch.testing.assert_close(_read_main(bucket, index, numel), ref.cpu(), atol=0, rtol=0)
            assert entry.main_param.grad is None
            assert entry.main_param.untyped_storage().nbytes() == 0
    for param, ref in zip(model_params, reference, strict=True):
        numel = param.numel()
        torch.testing.assert_close(param_data[offset : offset + numel], ref.to(torch.bfloat16), atol=0, rtol=0)
        offset += numel
    for bucket in store.buckets:
        os.close(bucket.fd)


def test_step_without_a_recorded_norm_fails_loudly(tmp_path):
    store, *_ = _setup(tmp_path, clip_grad=1.0)
    with pytest.raises(AssertionError, match="grad norm was not recorded"):
        store.step()
    for bucket in store.buckets:
        os.close(bucket.fd)


def test_megatron_norm_and_clip_leave_streamed_mains_to_the_store(tmp_path):
    """Outside step() the streamed mains have no .grad: Megatron's norm takes the store's, its clip skips them."""
    from types import MethodType

    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, multi_tensor_scale_tensor_impl
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from miles_plugins.optimizers.nvme_stream import _bind

    store, _, entries, _ = _setup(tmp_path, clip_grad=1.0)
    try:
        resident = torch.zeros(4, dtype=torch.float32, device="cuda")
        resident.tensor_model_parallel = True
        resident.grad = torch.ones(4, device="cuda")
        dist_opt = store.dist_opt
        dist_opt.optimizer.param_groups[0]["params"] = [entry.main_param for entry in entries] + [resident]
        dist_opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
        dist_opt.config.use_precision_aware_optimizer = False
        dist_opt.get_parameters = MethodType(MegatronOptimizer.get_parameters, dist_opt)
        dist_opt._filter_grads_for_norm = MethodType(MegatronOptimizer._filter_grads_for_norm, dist_opt)
        _bind(dist_opt, store)

        grads = dist_opt.get_grads_for_grad_norm()
        assert len(grads) == 2 and grads[0] is resident.grad
        expected = torch.linalg.vector_norm(torch.cat([e.model_param.main_grad.float() for e in entries]))
        torch.testing.assert_close(grads[1], expected.reshape(1), rtol=1e-6, atol=0)

        # ChainedOptimizer.step clips each child's get_parameters(); an all-streamed child has no grad to
        # scale, so the clip must return without reaching TE's kernel (it segfaults on an empty list).
        total_norm = torch.full((1,), 100.0, device="cuda") if multi_tensor_scale_tensor_impl is not None else 100.0
        clip_grad_by_total_norm_fp32([entry.main_param for entry in entries], 1.0, total_norm)
    finally:
        for bucket in store.buckets:
            os.close(bucket.fd)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
