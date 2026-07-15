from __future__ import annotations

import json
import logging
import os
import queue
import shutil
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from miles.utils.distributed_utils import get_gloo_group

from ..common import _check_weight_sync_results
from .mixin import DistBucketedWeightUpdateMixin

logger = logging.getLogger(__name__)


class UpdateWeightFromDiskDelta(DistBucketedWeightUpdateMixin):
    """
    Delta weight sync over a shared filesystem. Source ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    each engine's /pull_weights fans the apply out to every host it spans, then the engine reloads
    the patched local checkpoint via the ordinary update_weights_from_disk path. miles only ever
    talks to one endpoint per engine, so multi-node serving needs nothing extra.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        assert not is_lora, "LoRA weight sync is not supported for disk-delta weight transfer."
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False
        self.delta_dir = args.update_weight_disk_dir
        os.makedirs(self.delta_dir, exist_ok=True)
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        # Post-write hook: object-store-backed shared filesystems lack cross-host
        # read-after-write consistency, so written files need an explicit step
        # (e.g. uploading them to the backing object store) before the engines can see them.
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.misc import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)
        self._init_lora(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )

    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        # No NCCL groups: the transport is the shared filesystem. The rollout_engine_lock the
        # NCCL path uses isn't needed either — the engine-side apply is serialized by a per-host
        # flock behind /pull_weights.
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self._group_name = "miles-disk-delta"

    @property
    def _is_source(self):
        """If it's the source gpu producing the gathered HF tensors this rank publishes."""
        return get_parallel_state().intra_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0

    @torch.no_grad()
    def update_weights(self) -> None:
        # The first call only captures the baseline snapshot the next sync diffs against.
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return

        self.weight_version += 1
        self._publish()
        self._reload_engines()
        self._record_metrics()

    def _capture_baseline(self) -> None:
        """Capture the baseline snapshot the first delta diffs against (no publish), and clear any
        stale stream from a prior run. Seeds from hf_checkpoint — what each host materializes its
        base from — so the invariant ``snapshot == engine base`` holds even where the megatron->HF
        round-trip trims vocab-padding rows (embed/lm_head). A tensor absent there (rare) falls back
        to the gathered value. pull_weights(0) makes each host materialize its local base now,
        overlapped with the snapshot gather, so the first real sync only pays the delta apply."""
        # a prior run's versions would apply against the wrong base; start the dir clean
        pulls = []
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self.delta_dir, list(self.rollout_engines))
            pulls = [engine.pull_weights.remote(target_version=0) for engine in self.rollout_engines]
        dist.barrier(group=get_gloo_group())

        read_hf = make_tensor_reader(self.args.hf_checkpoint)  # index the HF headers once

        def seed_bucket(converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None) -> None:
            for name, tensor in converted_named_tensors:
                try:
                    self._snapshot[name] = read_hf(name)
                except KeyError:
                    self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
                    logger.warning("seed: %s absent from hf_checkpoint; seeding from current weights", name)

        self._for_each_hf_bucket(seed_bucket)
        if dist.get_rank() == 0:
            _check_weight_sync_results(ray.get(pulls), is_lora=False)
            if self.args.check_weight_update_equal:
                # The weights checker resets engine tensors at startup and compares after the
                # first sync, expecting it to rewrite every tensor. The baseline publishes
                # nothing, so reload the just-pulled base checkpoint to restore engine state
                # (and set the engine weight version the CI equality check expects).
                results = ray.get(
                    [
                        engine.update_weights_from_disk.remote(
                            model_path=self.args.update_weight_local_checkpoint_dir,
                            weight_version=str(self.weight_version),
                        )
                        for engine in self.rollout_engines
                    ]
                )
                _check_weight_sync_results(results, is_lora=False)
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )

    def _for_each_hf_bucket(self, bucket_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None]) -> None:
        """Feed every gathered HF bucket through ``bucket_func``: the base-class TP pass then the
        EP pass. All ranks join the gathers; ``bucket_func`` only runs on source ranks."""
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None
        self._gather_and_update_non_expert_weights(bucket_func, pbar)
        dist.barrier(group=get_gloo_group())
        self._gather_and_update_expert_weights(bucket_func, pbar)
        dist.barrier(group=get_gloo_group())

    def _publish(self) -> None:
        """Encode this version's changed tensors (source ranks), then write it as a canonical HF dir."""
        self._encode_delta()
        self._write_delta_files()

    def _drop_duplicate_names(self, group, world: int, rank: int) -> None:
        """A parameter Megatron replicates across PP stages — the word embedding on the last stage
        when it hosts an MTP block, or tied embeddings — is gathered and diffed by one source rank
        per stage, so the same HF tensor lands in several ranks' shards. The published artifact
        must hold each tensor exactly once (the XOR apply is an involution: applied twice it
        reverts), so keep the lowest-rank copy and drop the rest. The replicas are gradient-synced
        and byte-identical; a checksum divergence means the sync is broken — never publish it."""
        all_checksums: list = [None] * world
        dist.all_gather_object(all_checksums, self._checksums, group=group)
        for other_rank, other in enumerate(all_checksums[:rank]):
            for name in self._delta.keys() & other.keys():
                if other[name] != self._checksums[name]:
                    raise RuntimeError(
                        f"{name!r} published by rank {other_rank} and rank {rank} with different bytes; "
                        "PP-replicated parameters must stay identical across stages."
                    )
                del self._delta[name]
                del self._checksums[name]

    def _write_delta_files(self) -> None:
        """Write this rank's changed tensors as one canonical model-NNNNN.safetensors, and on rank
        0 the HF index. The sequential file numbers and the index are coordinated over gloo (small
        object gathers), not the filesystem — a non-POSIX shared filesystem may not surface one
        rank's writes to another until commit."""
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()

        self._drop_duplicate_names(group, world, rank)

        # number the files sequentially across only the ranks that have one (no gaps)
        counts: list = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        offset, total = sum(counts[:rank]), sum(counts)

        fname = None
        self.wire_bytes = 0
        if self._delta:
            fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, fname), blob)

        maps: list = [None] * world
        dist.all_gather_object(maps, {name: fname for name in self._delta}, group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{self.weight_version:06d}",
                    "base_version": f"{self.weight_version - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {name: f for m in maps for name, f in m.items()},
            }
            _atomic_write(os.path.join(self._version_dir, "model.safetensors.index.json"), json.dumps(index).encode())
        dist.barrier(group=group)

    def _reload_engines(self) -> None:
        """Commit the published files, have each engine pull the delta onto every host it spans
        (checksum-verified), then reload the engines. The pull is disk-only, so it runs before
        pause and overlaps generation."""
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            pulls = ray.get([engine.pull_weights.remote(self.weight_version) for engine in self.rollout_engines])
            _check_weight_sync_results(pulls, is_lora=False)
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            if mode not in ("in_place"):
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            results = ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            _check_weight_sync_results(results, is_lora=False)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _encode_delta(self) -> None:
        """Diff each gathered HF tensor against the snapshot, keeping the changed ones (compressed)
        in self._delta with their checksums. The GPU->CPU gather is pipelined into a compute pool:
        each bucket callback copies one tensor at a time to a pinned buffer and submits it; pool
        workers diff and compress in parallel (each is a few big GIL-releasing numpy/zstd calls)."""
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        if self._is_source:
            os.makedirs(self._version_dir, exist_ok=True)
        snapshot = self._snapshot
        self._delta: dict[str, np.ndarray] = {}  # changed tensor name -> compressed diff
        self._checksums: dict[str, str] = {}  # changed tensor name -> new-state checksum
        self.changed_bytes = self.total_bytes = 0

        # Pinned host-buffer pool: a pinned non_blocking GPU->CPU copy is far faster than .cpu().
        max_bytes = max((int(v.nbytes) for v in snapshot.values()), default=0)
        free_q: queue.Queue = queue.Queue()
        use_pinned = True
        try:
            for _ in range(max(4, min(2 * NUM_WORKERS, (32 << 30) // max(max_bytes, 1)))):
                free_q.put(torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True))
        except RuntimeError as e:  # low memlock limit
            logger.warning("pinned host buffers unavailable (%s); using pageable .cpu()", e)
            use_pinned = False

        def diff_and_compress(name, buf, nbytes, pinned):
            if pinned:  # copy out and free the pinned buffer before the heavy diff/compress
                new = np.empty(nbytes, dtype=np.uint8)
                np.copyto(new, buf.numpy()[:nbytes])
                free_q.put(buf)
            else:
                new = buf
            old = snapshot[name]
            if self.delta_encoding == "xor":
                diff = new ^ old
                changed = int(np.count_nonzero(diff))
            elif self.delta_encoding == "overwrite":
                mask = new != old
                changed = int(np.count_nonzero(mask))
                diff = overwrite_encode(new, mask)
            else:
                raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
            if not changed:
                return name, new, None, None, 0
            compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            return name, new, compressed, checksum(self.checksum_algorithm, new), changed

        def collect(fut):
            name, new, compressed, digest, changed = fut.result()
            snapshot[name] = new  # becomes the next sync's base
            if changed:
                self.changed_bytes += changed
                self._delta[name] = compressed
                self._checksums[name] = digest

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        inflight: deque = deque()

        def encode_bucket(converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None) -> None:
            for name, tensor in converted_named_tensors:
                flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = int(flat.numel())
                if use_pinned and nbytes <= max_bytes:
                    buf = free_q.get()  # blocks when all buffers are in flight -> backpressures the gather
                    buf[:nbytes].copy_(flat, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    payload, pinned = buf, True
                else:
                    payload, pinned = flat.cpu().numpy(), False
                self.total_bytes += nbytes
                inflight.append(pool.submit(diff_and_compress, name, payload, nbytes, pinned))
                if len(inflight) >= 2 * NUM_WORKERS:
                    collect(inflight.popleft())

        try:
            self._for_each_hf_bucket(encode_bucket)
            while inflight:
                collect(inflight.popleft())
        finally:
            pool.shutdown()

    def _record_metrics(self) -> None:
        """All-reduce the byte counts and record changed-fraction / wire size; the actor drains
        update_weight_metrics onto the step log."""
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        self.update_weight_metrics = {
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
        }
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta v=%s] density=%.2f%% wire=%.2f GB",
                self.weight_version,
                100.0 * changed / max(total, 1),
                wire / 1e9,
            )


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
