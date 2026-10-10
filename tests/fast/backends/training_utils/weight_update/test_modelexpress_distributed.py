import json
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import safetensors.numpy
import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zstandard

pytest.importorskip("modelexpress_rl")

from modelexpress_rl import WeightPayloadFormat, refit_pb2
from modelexpress_rl.train import runtime as mx_runtime
from miles.backends.training_utils.weight_update.protocols import modelexpress as mx

from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils import distributed_utils


class _LocalS3:
    def __init__(self, root):
        self.root = root

    def put(self, *, uri, data):
        path = self.root / uri.removeprefix("s3://test/run/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def close(self):
        pass


def _run_publisher_rank(rank, directory):
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)
        patch.setenv("MX_REFIT_DELTA_WORKERS", "1")
        patch.setattr(mx_runtime, "S3Client", lambda **kwargs: _LocalS3(root / "objects"))

        def create_version(**kwargs):
            uid = kwargs.get("uid") or kwargs["idempotency_key"].rsplit("/", 1)[-1]
            payloads = {
                WeightPayloadFormat.FULL_TENSOR: refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
                WeightPayloadFormat.FULL_HF_CHECKPOINT: refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT,
                WeightPayloadFormat.XOR_DELTA: refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            }
            version = refit_pb2.WeightVersion(
                uid=uid,
                model_name=kwargs["model_name"],
                payload_format=payloads[kwargs["payload_format"]],
                object_storage=refit_pb2.ObjectStorageSource(
                    uri=kwargs["object_storage"].uri, storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3
                ),
            )
            if "base_version_id" in kwargs:
                version.base_version_id = kwargs["base_version_id"]
            (root / f"{uid}.pb").write_bytes(version.SerializeToString())
            return SimpleNamespace(version_id=uid)

        def get_version(request, **kwargs):
            return refit_pb2.GetWeightVersionResponse(
                version=refit_pb2.WeightVersion.FromString((root / f"{request.uid}.pb").read_bytes())
            )

        def mark_ready(uid, state):
            assert (root / "objects" / uid / "model.safetensors.index.json").is_file()

        patch.setattr(
            mx.ModelExpressControlClient,
            "connect",
            lambda **kwargs: SimpleNamespace(
                create_weight_version=create_version, update_weight_version_state=mark_ready
            ),
        )
        patch.setattr(
            mx.ModelExpressTrainerClient,
            "_service",
            property(lambda self: SimpleNamespace(GetWeightVersion=get_version)),
        )

        pp_groups = [dist.new_group(ranks=ranks, backend="gloo") for ranks in ([0, 2], [1, 3])]
        parallel_state = SimpleNamespace(pp=SimpleNamespace(group=pp_groups[rank % 2], size=2))
        step = 0

        def iter_weights(weights, *, materialize=True, **kwargs):
            for bucket_id in range(2):
                # All four ranks must join every gather, including non-senders.
                values = [None] * 4
                dist.all_gather_object(values, rank)
                assert values == list(range(4))
                if materialize:
                    yield [(f"stage{rank // 2}.weight{bucket_id}", torch.tensor([float(step + bucket_id)]))]

        iterator = SimpleNamespace(
            placement=SimpleNamespace(gather_pp=False), weight_update_selector="all", iter_hf_weights=iter_weights
        )
        args = Namespace(
            update_weight_transfer_mode="modelexpress",
            colocate=False,
            pause_generation_mode="abort",
            check_lora_weight_equal=False,
            modelexpress_config={
                "model_name": "miles-test",
                "server_url": "unused:8001",
                "object_storage_uri_prefix": "s3://test/run",
                "initial_base_version_id": "v0",
                "seed_checkpoint_path": str(root / "seed"),
                "full_hf_checkpoint_interval": 2,
            },
        )
        engine = SimpleNamespace(
            pause_generation=AsyncMock(),
            flush_cache=AsyncMock(),
            update_weights_from_modelexpress=AsyncMock(return_value={"success": True}),
            update_weight_version=AsyncMock(),
            continue_generation=AsyncMock(),
        )
        updater = WeightUpdater(
            args,
            [],
            weights_getter=lambda: {},
            model_name="test",
            quantization_config=None,
            iterator_factory=lambda *a, **kw: iterator,
            parallel_state=parallel_state,
            is_lora=False,
        )
        protocol = updater.protocol
        try:
            updater.connect_rollout_engines([engine])
            assert protocol.is_sender == (rank in (0, 2))
            if protocol.is_sender:
                assert dist.get_process_group_ranks(protocol._publisher_group) == [0, 2]
            else:
                assert protocol._trainer is None

            for step in range(4):  # Seed, delta, full checkpoint, then delta from the new base.
                updater.update_weights()
                assert protocol._current_version_id == f"v{step}"
                if not protocol.is_sender:
                    assert protocol._staged is None
            (root / f"rank{rank}.done").touch()
        finally:
            if protocol._trainer is not None:
                protocol._trainer.close()
            dist.destroy_process_group()


def test_non_senders_gather_while_only_senders_publish(tmp_path):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    expected = {
        f"stage{stage}.weight{bucket}": torch.tensor([float(bucket)]) for stage in range(2) for bucket in range(2)
    }
    safetensors.torch.save_file(expected, seed_dir / "model.safetensors")
    processes = mp.spawn(_run_publisher_rank, args=(str(tmp_path),), nprocs=4, join=False)
    try:
        deadline = time.monotonic() + 90
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("publication or weight gathering hung with non-sender ranks")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join()

    assert all((tmp_path / f"rank{rank}.done").is_file() for rank in range(4))
    previous = {name: tensor.view(torch.uint8).numpy() for name, tensor in expected.items()}
    for step in range(1, 4):
        version_dir = tmp_path / "objects" / f"v{step}"
        index = json.loads((version_dir / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == set(expected)
        assert len(set(index["weight_map"].values())) == 2
        for name, filename in index["weight_map"].items():
            shard = (version_dir / filename).read_bytes()
            if step == 2:
                actual = safetensors.torch.load(shard)[name].view(torch.uint8).numpy()
            else:
                encoded = safetensors.numpy.load(shard)[name]
                delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(encoded), dtype=np.uint8)
                actual = np.bitwise_xor(previous[name], delta)
            assert np.array_equal(actual, (expected[name] + step).view(torch.uint8).numpy())
            previous[name] = actual
