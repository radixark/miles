"""Focused GPU coverage for streamed optimizer main-param initialization."""

import errno
import json
import os
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.optimizers.nvme_stream import (
    NVMeOptimizerStateStore,
    _Bucket,
    _drop_file_cache,
    _Entry,
    _live_dir_root,
    _purge_legacy_rank_dir,
    _resize,
    _Stager,
)

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["miles-plugin"],
)


def test_bucketwise_main_initialization_preserves_bytes_and_releases_cuda_storage(tmp_path):
    model_param = torch.nn.Parameter(torch.arange(600_000, dtype=torch.float32, device="cuda").to(torch.bfloat16))
    main_param = torch.empty_like(model_param, dtype=torch.float32)
    _resize(main_param, 0)
    entry = _Entry(model_param, main_param, 0)
    dtypes = {
        "main": torch.float32,
        "exp_avg": torch.float32,
        "exp_avg_sq": torch.float32,
    }
    bucket = _Bucket(
        str(tmp_path / "bucket.bin"),
        entries=[entry],
        adam=None,
        stager=_Stager(1024 * 1024),
        dtypes=dtypes,
    )
    param_range = SimpleNamespace(start=0, end=model_param.numel(), size=model_param.numel())
    dist_opt = SimpleNamespace(
        model_fp32_groups=[],
        shard_fp32_groups=[],
        _get_model_param_range_map=lambda _param: {"param": param_range},
        _build_model_param_to_state_dict_param_map=lambda _state_dict: {model_param: torch.full_like(model_param, 7)},
    )
    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt = dist_opt
    store.buckets = [bucket]

    with patch(
        "miles_plugins.optimizers.nvme_stream._drop_file_cache",
        wraps=_drop_file_cache,
    ) as drop_file_cache:
        written = store.refresh_main_from_model_params()

    nbytes = main_param.numel() * main_param.element_size()
    assert written == nbytes
    assert main_param.untyped_storage().nbytes() == 0
    assert drop_file_cache.call_count == 1
    assert drop_file_cache.call_args.args[2] == bucket.offsets["exp_avg"][0]
    assert drop_file_cache.call_args.kwargs["sync"] is True

    restored = torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, 0)), dtype=torch.float32)
    torch.testing.assert_close(restored, model_param.float().cpu(), atol=0, rtol=0)

    with patch(
        "miles_plugins.optimizers.nvme_stream._drop_file_cache",
        wraps=_drop_file_cache,
    ) as drop_file_cache:
        reloaded = store.refresh_main_from_model_params(state_dict=object())

    assert reloaded == nbytes
    assert main_param.untyped_storage().nbytes() == 0
    assert drop_file_cache.call_count == 1
    restored = torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, 0)), dtype=torch.float32)
    torch.testing.assert_close(restored, torch.full_like(restored, 7), atol=0, rtol=0)
    bucket.close()


def test_bucket_close_releases_file_descriptor(tmp_path):
    bucket = _Bucket(
        str(tmp_path / "empty.bin"),
        entries=[],
        adam=None,
        stager=None,
        dtypes={},
    )
    fd = bucket.fd

    bucket.close()
    bucket.close()

    with pytest.raises(OSError) as exc_info:
        os.fstat(fd)
    assert exc_info.value.errno == errno.EBADF


def test_store_indices_produce_stable_checkpoint_paths():
    relative_dirs = []
    for store_index in range(2):
        store = object.__new__(NVMeOptimizerStateStore)
        store._rank = 0
        store._instance = 0
        store.store_index = store_index
        relative_dirs.append(store.relative_dir)

    assert relative_dirs == ["rank00000/opt0_0", "rank00000/opt0_1"]


def test_live_store_paths_are_namespaced_by_role(tmp_path):
    args = Namespace(offload_train_disk_dir=str(tmp_path))

    assert _live_dir_root(args, "actor").endswith("optimizer_state/actor")
    assert _live_dir_root(args, "critic").endswith("optimizer_state/critic")
    with pytest.raises(AssertionError, match="path-unsafe"):
        _live_dir_root(args, "actor/replica")
    with pytest.raises(AssertionError, match="role must not be empty"):
        _live_dir_root(args, "")


def test_legacy_live_rank_directory_is_removed(tmp_path):
    legacy_rank_dir = tmp_path / "optimizer_state" / "rank00003"
    legacy_rank_dir.mkdir(parents=True)
    (legacy_rank_dir / "bucket00000.bin").write_bytes(b"stale")

    with patch("torch.distributed.get_rank", return_value=3):
        _purge_legacy_rank_dir(str(tmp_path))

    assert not legacy_rank_dir.exists()


def test_store_checkpoint_round_trip_preserves_main_moments_and_step(tmp_path):
    model_param = torch.nn.Parameter(torch.arange(1024, dtype=torch.float32, device="cuda").to(torch.bfloat16))
    main_param = torch.empty_like(model_param, dtype=torch.float32)
    _resize(main_param, 0)
    entry = _Entry(model_param, main_param, 0)
    adam = SimpleNamespace(param_groups=[{"step": 0}], state={})
    dtypes = {
        "main": torch.float32,
        "exp_avg": torch.float32,
        "exp_avg_sq": torch.float32,
    }
    (tmp_path / "live").mkdir()
    bucket = _Bucket(
        str(tmp_path / "live" / "bucket.bin"),
        entries=[entry],
        adam=adam,
        stager=_Stager(1024 * 1024),
        dtypes=dtypes,
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
    store._stager = bucket._stager
    store.dtypes = dtypes
    store._fp32_adam = None
    store._allow_fresh_state = False
    store._rank = 0
    store._instance = 0
    store.store_index = 0

    store.refresh_main_from_model_params()
    bucket.allocate_moments()
    bucket.fetch()
    adam.state[main_param]["exp_avg"].fill_(2)
    adam.state[main_param]["exp_avg_sq"].fill_(3)
    bucket.flush()
    adam.param_groups[0]["step"] = 7
    store.save_to(str(tmp_path / "checkpoint"))

    os.pwrite(bucket.fd, bytes(bucket.nbytes), 0)
    adam.param_groups[0]["step"] = 0
    assert store.load_from(str(tmp_path / "checkpoint"))
    bucket.fetch()

    torch.testing.assert_close(main_param, model_param.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        adam.state[main_param]["exp_avg"],
        torch.full_like(main_param, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        adam.state[main_param]["exp_avg_sq"],
        torch.full_like(main_param, 3),
        atol=0,
        rtol=0,
    )
    assert adam.param_groups[0]["step"] == 7

    adam.param_groups[0]["step"] = 0
    store.save_to(str(tmp_path / "checkpoint_step_zero"))
    adam.param_groups[0]["step"] = 7
    assert store.load_from(str(tmp_path / "checkpoint_step_zero"))
    assert adam.param_groups[0]["step"] == 0
    bucket.close()


def test_load_rejects_missing_native_fp32_optimizer_sidecar(tmp_path):
    store = object.__new__(NVMeOptimizerStateStore)
    store.dtypes = {
        "main": torch.float32,
        "exp_avg": torch.float32,
        "exp_avg_sq": torch.float32,
    }
    store.buckets = []
    store._fp32_adam = SimpleNamespace(load_state_dict=lambda _state: None)
    store._allow_fresh_state = False
    store._rank = 0
    store._instance = 0
    store.store_index = 0
    checkpoint_dir = tmp_path / "checkpoint" / store.relative_dir
    checkpoint_dir.mkdir(parents=True)
    with open(checkpoint_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "dtypes": {segment: str(dtype) for segment, dtype in store.dtypes.items()},
                "has_fp32_resident_optimizer": True,
                "buckets": [],
            },
            f,
        )

    with pytest.raises(AssertionError, match="checkpoint is incomplete"):
        store.load_from(str(tmp_path / "checkpoint"))
