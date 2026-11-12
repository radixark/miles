"""Engine-host sharing, DE-local preparation, and failure lifetimes."""

import hashlib
import json
import mmap
import multiprocessing
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zstandard as zstd
from safetensors.numpy import load as load_columns

from miles.utils.gpu_delta import host as host
from miles.utils.gpu_delta import host_worker as worker
from miles.utils.gpu_delta.host_worker import HostPayloadPool
from miles.utils.gpu_delta.publication import PublicationWriter, seal_publication

TENSOR_IDS = {name: i for i, name in enumerate(("dense", "expert", "foreign", "raw"))}
DENSE = bytes(range(251)) * 100
FOREIGN = b"foreign-inner-frame"


def matrix(data, nbytes=None, parts=1, wrapped=True):
    """An identity inner frame in outer Zstd chunks, shaped like GpuBatchEncoder.finish_device output."""
    frames = [dict(encoded_offset=0, encoded_bytes=len(data), decoded_offset=0, decoded_bytes=len(data))]
    payload, outer, step = bytearray(), [], -(-len(data) // parts)
    for offset in range(0, len(data), step) if wrapped else ():
        chunk = data[offset : offset + step]
        encoded = zstd.ZstdCompressor(write_checksum=True).compress(chunk)
        outer.append(
            dict(
                encoded_offset=len(payload),
                encoded_bytes=len(encoded),
                decoded_offset=offset,
                decoded_bytes=len(chunk),
            )
        )
        payload.extend(encoded)
    payload = bytes(payload) if wrapped else data
    return dict(
        frames=frames,
        payload=payload,
        outer=dict(encoded_bytes=len(payload), decoded_bytes=len(data), frames=outer),
        changed_bytes=len(data),
        nbytes=nbytes or len(data),
    )


def write_publication(directory, owners, version=1, codec="snappy-zstd", payload_hash=True):
    """Seal one publication with the production writer, one payload file per owner.

    Owners map names to raw XOR masks (bytes) or matrix() records. The writer
    creates every file exclusively, so each publication needs a fresh directory.
    """
    # The producer reads its payload hash policy when the writer is constructed.
    with patch.dict(os.environ, {"GPU_DELTA_SKIP_PAYLOAD_HASH": str(int(not payload_hash))}):
        writers = [
            PublicationWriter(
                directory, "stream", version - 1, version, "fixture-plan", codec, owner=owner, publication_id="fixture"
            )
            for owner in range(len(owners))
        ]
    for writer, tensors in zip(writers, owners, strict=True):
        for name, tensor in tensors.items():
            if isinstance(tensor, bytes):
                writer.add_raw_xor_tensor(name, tensor, dict(dtype="U8", shape=[len(tensor)]))
            else:
                writer.add_encoded_tensor(
                    name,
                    tensor["frames"],
                    tensor["payload"],
                    tensor["outer"],
                    tensor["changed_bytes"],
                    dict(dtype="U8", shape=[1, tensor["nbytes"]]),
                )
    descriptor = seal_publication(directory, [writer.finish_shard() for writer in writers])
    return Path(descriptor["manifest_path"]), descriptor["manifest_sha256"]


def fixture(directory, version=1, codec="snappy-zstd", dense=DENSE, foreign=None, payload_hash=True):
    """Owner 0 writes dense (a 1 MiB row with a trailing gap), expert and raw; owner 1 writes foreign.

    The returned bytes cover the tensors a rank selects by default; the shared image also prepares foreign.
    """
    expected = {"dense": dense, "expert": b"snappy-bytes" * 50, "raw": b"abcd"}
    owners = [
        {"dense": matrix(dense, nbytes=1 << 20), "expert": matrix(expected["expert"]), "raw": expected["raw"]},
        {"foreign": foreign or matrix(FOREIGN)},
    ]
    path, digest = write_publication(directory, owners, version, codec, payload_hash)
    return path, digest, expected


def arena_inputs(path):
    """Catalog names, raw mask, payload backing, file spans and columns as the CPU child maps them."""
    manifest = json.loads(path.read_bytes())
    catalog = json.loads((path.parent / "catalog.json").read_bytes())["tensors"]
    payloads = [(path.parent / record["name"]).read_bytes() for record in manifest["files"]]
    sizes = np.array([len(payload) for payload in payloads], dtype=np.int64)
    return (
        [entry["name"] for entry in catalog],
        np.array([entry["encoding"] == "xor_raw_bytes" for entry in catalog]),
        memoryview(b"".join(payloads)),
        np.column_stack((np.cumsum(sizes) - sizes, sizes)),
        load_columns((path.parent / "metadata.safetensors").read_bytes()),
    )


def prepare_snapshot(arena, path, digest, names, pool, timings, metadata, cancelled=None):
    if arena._worker is not None:
        arena._worker.builder.pool = pool
    publication = arena.prepare_publication(
        metadata | dict(manifest_path=str(path), manifest_sha256=digest),
        timings,
        cancelled,
    )
    return arena.prepare_local(publication, [TENSOR_IDS[name] for name in names], timings)


def one_poll():
    """A cancellation event that aborts a follower after its first state poll."""
    polled = threading.Event()
    return SimpleNamespace(is_set=polled.is_set, wait=lambda timeout: polled.set())


def fake_fallocate(fd, offset, length):
    os.ftruncate(fd, offset + length)


class FakeHostAllocation:
    """Rank-private CPU backing; CUDA allocator admission is tested separately."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.view = memoryview(bytearray(capacity))

    def close(self):
        self.view.release()


class FakeHostMapping:
    """Keep real shared mmap semantics without registering it with CUDA."""

    def __init__(self, mapping):
        self.registered = False
        self.mapping = mapping
        self.capacity = len(mapping)
        self.view = memoryview(mapping)

    def pin(self):
        self.registered = True

    def close(self):
        self.view.release()
        self.mapping = None


class InlineCpuWorker:
    """Use the real CPU builder for deterministic unit fault injection."""

    def __init__(self, directory, pool):
        with patch.object(worker, "HostPayloadPool", return_value=pool):
            self.builder = worker.HostBuilder(directory, pool.workers)

    def prepare(self, expected, namespace, metrics, cancelled):
        result = self.builder.prepare(expected, namespace)
        metrics.update(result["metrics"])
        return result["index"]

    def release(self, token):
        return self.builder.release(token)

    def close(self):
        self.builder.close()


def shared_participants(engine="a"):
    return [
        dict(
            engine_id=engine,
            rank_id=f"{engine}-original-{rank}",
            host_cache_id="shared-host-" + engine,
            pid=100 + rank,
            tp_rank=rank,
            dp_rank=0,
            pp_rank=0,
        )
        for rank in range(2)
    ]


def _shared_child(rank, publications, barrier, output):
    participants = shared_participants()
    arena = host.HostArena("a", 0, participants[rank])
    records, reads = [], []
    read_bytes = Path.read_bytes

    def observed_read(path):
        reads.append(path.name)
        return read_bytes(path)

    try:
        with (
            patch.object(host, "configured_cpu_workers", return_value=2),
            patch("torch.cuda.set_device") as set_device,
            patch.object(Path, "read_bytes", new=observed_read),
        ):
            for version, (path, digest, expected) in enumerate(publications, 1):
                hardware = version == 4
                barrier.wait(timeout=10)
                metrics = {}
                reads.clear()
                publication = arena.prepare_publication(
                    metadata(version) | dict(manifest_path=str(path), manifest_sha256=digest),
                    metrics,
                )
                assert all(
                    not array.flags.writeable
                    for array in (
                        publication.tensors,
                        publication.frames,
                        publication.gaps,
                    )
                )
                if version == 1:
                    assert [e["name"] for e in publication.read_catalog()["tensors"]] == list(TENSOR_IDS)
                names = ["dense", "raw"] if rank == 0 else ["dense", "expert"]
                snapshot = arena.prepare_local(publication, [TENSOR_IDS[name] for name in names], metrics, hardware)
                assert {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in names} == {
                    name: expected[name] for name in names
                }
                assert arena.shared_allocation.registered is not hardware
                records.append(
                    dict(
                        metrics=metrics,
                        capacity=arena.capacity.copy(),
                        shared_capacity=arena.shared_capacity.copy(),
                        metadata_capacity=publication.index["metadata"].copy(),
                        tensors=publication.tensors.copy(),
                        frames=publication.frames.copy(),
                        gaps=publication.gaps.copy(),
                        offsets=snapshot.offsets.copy(),
                        pool_id=id(arena._worker.builder.pool) if arena._worker else None,
                        set_device_calls=set_device.call_count,
                        manifest_reads=reads.count(path.name),
                        catalog_reads=reads.count("catalog.json"),
                    )
                )
                barrier.wait(timeout=10)
                snapshot.close(True)
                barrier.wait(timeout=10)
        output.put((rank, records))
    finally:
        arena.close()


def metadata(version=1, engine="a", validate_payload_hash=True):
    return dict(
        stream_id="stream",
        session_id=f"session-{version}",
        base_version=version - 1,
        target_version=version,
        plan_digest="fixture-plan",
        validate_payload_hash=validate_payload_hash,
        participants=shared_participants(engine),
    )


class TestHostSnapshot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.pool = HostPayloadPool(2)
        self.addCleanup(self.pool.close)
        for replacement in (
            patch.object(host, "_cache_base", return_value=self.cache),
            patch.object(host, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(worker, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(host, "CpuWorker", side_effect=lambda directory: InlineCpuWorker(directory, self.pool)),
            patch.object(host, "HostAllocation", FakeHostAllocation),
            patch.object(host, "HostMapping", FakeHostMapping),
            patch("torch.cuda.set_device"),
            patch.object(os, "posix_fallocate", side_effect=fake_fallocate, create=True),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def arena(self, engine="a", rank=0):
        arenas = [host.HostArena(engine, 0, identity) for identity in shared_participants(engine)]
        for arena in arenas:
            self.addCleanup(arena.close)
        return arenas[rank]

    def held_build(self, build, while_held=lambda: None, *, after_read=False):
        """Run build() while one payload read is held.

        Default: owner 1's read waits before its I/O; owner 0's read has finished by while_held().
        after_read: owner 0's read holds its pool worker after its I/O; owner 1's read starts only
        then and must finish meanwhile, so the two owner-file reads must run in parallel.
        """
        held, release, first_read, outcome = threading.Event(), threading.Event(), threading.Event(), []
        original_read = worker._read_verify_payload
        held_name = "owner-00000.bin" if after_read else "owner-00001.bin"

        def gated_read(source, destination, record, skip_payload_hash):
            if source.name == held_name:
                result = original_read(source, destination, record, skip_payload_hash) if after_read else None
                held.set()
                assert release.wait(5)
                return result if after_read else original_read(source, destination, record, skip_payload_hash)
            if after_read:
                assert held.wait(5)
            try:
                return original_read(source, destination, record, skip_payload_hash)
            finally:
                first_read.set()

        def run():
            try:
                outcome.append(build())
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=run)
        with (
            patch.object(worker, "_read_verify_payload", side_effect=gated_read),
            patch.object(self.pool, "decode_zstd", wraps=self.pool.decode_zstd) as decode,
        ):
            thread.start()
            try:
                self.assertTrue(held.wait(5) and first_read.wait(5))
                # Shared decode starts only after every payload read drains, including after a failure.
                self.assertTrue(thread.is_alive())
                decode.assert_not_called()
                while_held()
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        return outcome

    def test_empty_publication_authenticates_sidecars_without_allocating_payload(self):
        path, digest = write_publication(self.root / "empty", [{}], codec="bitcomp-zstd")
        wire = json.loads(path.read_bytes())
        for kind in ("catalog", "metadata"):
            sidecar = path.parent / wire[kind]["file"]
            original = sidecar.read_bytes()
            sidecar.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "metadata changed or SHA256"):
                self.arena(kind).prepare_publication(
                    metadata(engine=kind) | dict(manifest_path=str(path), manifest_sha256=digest),
                    {},
                )
            sidecar.write_bytes(original)
        arena, timings = self.arena(), {}
        publication = arena.prepare_publication(
            metadata() | dict(manifest_path=str(path), manifest_sha256=digest), timings
        )
        snapshot = arena.prepare_local(publication, [], timings, hardware=False)
        self.assertEqual(publication.index["metadata_counts"], (0, 0, 0))
        self.assertEqual(timings["host_rank_arena_bytes"], 0)
        self.assertEqual(timings["host_rank_outer_zstd_frames"], 0)
        self.assertEqual(timings["host_prepared_cache_capacity_bytes"], 0)
        snapshot.close(True)
        self.assertEqual(
            json.loads((arena.directory / "state.json").read_bytes())["state"],
            "REUSABLE",
        )

    def test_verified_cache_precedes_shared_decode_and_failures_never_release(self):
        path, digest, expected = fixture(self.root / "verified")
        arena, metrics = self.arena("verified"), {}

        def while_held():
            self.assertIsNone(arena.allocation)
            self.assertEqual(json.loads((arena.directory / "state.json").read_bytes())["state"], "BUILDING")
            (path.parent / "owner-00000.bin").write_bytes(b"changed after retained read")

        (snapshot,) = self.held_build(
            lambda: prepare_snapshot(arena, path, digest, expected, self.pool, metrics, metadata(engine="verified")),
            while_held,
            after_read=True,
        )
        self.assertGreaterEqual(metrics["host_prepared_cache_build_s"], metrics["host_rank_outer_zstd_decode_s"])
        self.assertGreaterEqual(metrics["host_encoded_cache_access_s"], metrics["host_encoded_cache_wait_s"])
        self.assertIs(snapshot.get(TENSOR_IDS["dense"]).obj, arena.mapping.obj)
        self.assertEqual({name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected}, expected)
        publication = snapshot.publication
        snapshot.close()
        catalog = publication.directory / publication.index["catalog"]["file"]
        catalog.write_bytes(b"changed catalog")
        with self.assertRaisesRegex(ValueError, "shared canonical catalog changed"):
            publication.read_catalog()

        path, digest, expected = fixture(self.root / "corrupt")
        payload = path.parent / "owner-00000.bin"
        corrupted = bytearray(payload.read_bytes())
        corrupted[-1] ^= 1
        payload.write_bytes(corrupted)
        arena = self.arena("corrupt")
        outcome = self.held_build(
            lambda: prepare_snapshot(arena, path, digest, expected, self.pool, {}, metadata(engine="corrupt"))
        )
        self.assertEqual([str(error) for error in outcome], ["delta payload SHA256 mismatch"])
        # The child never retries a publication that failed before READY.
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            prepare_snapshot(arena, path, digest, expected, self.pool, {}, metadata(engine="corrupt"))

        # Verification alone cannot publish READY when shared outer decoding fails.
        foreign = matrix(FOREIGN)
        foreign["payload"] = b"not-zstd".ljust(len(foreign["payload"]), b" ")
        path, digest, _ = fixture(self.root / "bad-decode", foreign=foreign)
        arena = self.arena("bad-decode")
        with self.assertRaises(zstd.ZstdError):
            prepare_snapshot(arena, path, digest, ["foreign"], self.pool, {}, metadata(engine="bad-decode"))
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(arena, path, digest, ["foreign"], self.pool, {}, metadata(2, "bad-decode"))

    def test_payload_hash_policy_follows_publication_and_requires_peer_agreement(self):
        strict, peer = self.arena("strict"), self.arena("strict", rank=1)
        for version, validate in enumerate((True, False, True), 1):
            path, digest, expected = fixture(self.root / f"strict-{version}", version)
            metrics = {}
            # A legacy inherited environment cannot override the request policy.
            with patch.dict(os.environ, {"GPU_DELTA_SKIP_PAYLOAD_HASH": str(int(validate))}):
                snapshot = prepare_snapshot(
                    strict,
                    path,
                    digest,
                    expected,
                    self.pool,
                    metrics,
                    metadata(version, "strict", validate),
                )
            self.assertEqual(metrics["host_encoded_cache_created"], 1)
            self.assertEqual(metrics["host_encoded_cache_skip_payload_hash"], int(not validate))
            self.assertEqual(metrics["host_encoded_cache_hash_files"], 2 if validate else 0)
            self.assertEqual(
                {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected},
                expected,
            )
            # A peer requesting the other policy polls for its own token, never adopting this one.
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                prepare_snapshot(
                    peer, path, digest, expected, self.pool, {}, metadata(version, "strict", not validate), one_poll()
                )
            snapshot.close(True)

        # The manifest itself still has an exact digest; only payload SHA is skipped.
        path, digest, expected = fixture(self.root / "unchecked", payload_hash=False)
        for arena, created in ((self.arena(), 1), (self.arena(rank=1), 0)):
            metrics = {}
            snapshot = prepare_snapshot(
                arena,
                path,
                digest,
                expected,
                self.pool,
                metrics,
                metadata(validate_payload_hash=False),
            )
            self.assertEqual(
                {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected},
                expected,
            )
            self.assertEqual(metrics["host_encoded_cache_skip_payload_hash"], 1)
            self.assertEqual(metrics["host_encoded_cache_created"], created)
            for field in ("hash_files", "hash_bytes", "sha256_worker_sum_s"):
                self.assertEqual(metrics["host_encoded_cache_" + field], 0)
            snapshot.close()
        with self.assertRaisesRegex(ValueError, "payload checksum omitted"):
            prepare_snapshot(
                self.arena("require-checksum"),
                path,
                digest,
                expected,
                self.pool,
                {},
                metadata(engine="require-checksum"),
            )

    def test_shared_preparation_has_one_owner_and_reuses_or_grows_one_inode(self):
        variants = {
            # Owner 1 keeps foreign's outer payload but omits its inner frames, so nothing is prepared.
            2: dict(foreign=dict(matrix(FOREIGN), frames=[], changed_bytes=0)),
            3: dict(dense=DENSE + b"d" * 80000),
            4: dict(codec="snappy-zstd"),
        }
        publications = [
            fixture(
                self.root / f"version-{version}", version, **({"codec": "bitcomp-zstd"} | variants.get(version, {}))
            )
            for version in range(1, 6)
        ]
        context = multiprocessing.get_context("fork")
        barrier, output = context.Barrier(2), context.Queue()
        children = [
            context.Process(
                target=_shared_child,
                args=(rank, publications, barrier, output),
            )
            for rank in range(2)
        ]
        for child in children:
            child.start()
        try:
            records = dict(output.get(timeout=30) for _ in children)
            for child in children:
                child.join(timeout=10)
                self.assertEqual(child.exitcode, 0)
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=10)
        for version in range(5):
            owner, peer = records[0][version], records[1][version]
            self.assertEqual(owner["shared_capacity"], peer["shared_capacity"])
            self.assertEqual(owner["metadata_capacity"], peer["metadata_capacity"])
            for key in ("tensors", "frames", "gaps"):
                np.testing.assert_array_equal(owner[key], peer[key])
            self.assertEqual(owner["manifest_reads"], 1)
            self.assertEqual(peer["manifest_reads"], 0)
            for row in (owner, peer):
                self.assertEqual(row["catalog_reads"], int(version == 0))
            self.assertEqual(owner["metrics"]["host_prepared_cache_created"], 1)
            self.assertEqual(peer["metrics"]["host_prepared_cache_reused"], 1)
            self.assertEqual(
                owner["metrics"]["host_rank_outer_zstd_tensors"],
                2 if version == 1 else 3,
            )
            self.assertEqual(peer["metrics"]["host_rank_outer_zstd_tensors"], 0)
            self.assertEqual(owner["metrics"]["host_encoded_cache_hash_files"], 2)
            self.assertEqual(peer["metrics"]["host_encoded_cache_hash_files"], 0)
            self.assertEqual(owner["metrics"]["host_rank_cpu_workers"], 2)
            self.assertEqual(peer["metrics"]["host_rank_cpu_workers"], 0)
            if version != 3:
                self.assertEqual(owner["capacity"], peer["capacity"])
                for row in (owner, peer):
                    self.assertEqual(row["metrics"]["host_rank_private_capacity_bytes"], 0)
                    self.assertEqual(row["metrics"]["host_rank_staging_copy_bytes"], 0)
        self.assertEqual(len({row["pool_id"] for row in records[0]}), 1)
        self.assertTrue(all(row["pool_id"] is None for row in records[1]))
        self.assertEqual(records[0][1]["tensors"][TENSOR_IDS["foreign"]]["prepared_nbytes"], 0)
        self.assertGreater(records[0][2]["tensors"][TENSOR_IDS["foreign"]]["prepared_nbytes"], 0)
        de_owner, de_peer = records[0][3], records[1][3]
        self.assertNotEqual(de_owner["capacity"]["identity"], de_peer["capacity"]["identity"])
        for row, names in (
            (de_owner, ["dense", "raw"]),
            (de_peer, ["dense", "expert"]),
        ):
            self.assertGreater(row["metrics"]["host_rank_private_capacity_bytes"], 0)
            self.assertEqual(
                row["metrics"]["host_rank_staging_copy_bytes"],
                sum(len(publications[3][2][name]) for name in names),
            )
        for rows in records.values():
            first, warm, grown, de, restored = rows
            self.assertEqual([row["set_device_calls"] for row in rows], [1, 1, 2, 2, 3])
            self.assertEqual(first["shared_capacity"], warm["shared_capacity"])
            self.assertEqual(warm["metrics"]["host_rank_mapping_reused"], 1)
            self.assertEqual(warm["metrics"]["host_rank_allocation_calls"], 0)
            self.assertNotEqual(
                first["shared_capacity"]["identity"],
                grown["shared_capacity"]["identity"],
            )
            self.assertEqual(grown["shared_capacity"]["generation"], 2)
            self.assertGreater(
                grown["shared_capacity"]["capacity"],
                first["shared_capacity"]["capacity"],
            )
            self.assertEqual(grown["shared_capacity"], de["shared_capacity"])
            self.assertEqual(grown["shared_capacity"], restored["shared_capacity"])

    def test_failed_shared_creator_cannot_publish_or_release_prepared_bytes(self):
        path, digest, _ = fixture(self.root / "v1", codec="bitcomp-zstd")
        participants = shared_participants()
        owner, peer = [host.HostArena("a", 0, identity) for identity in participants]
        self.addCleanup(owner.close)
        self.addCleanup(peer.close)
        request = metadata() | dict(participants=participants)
        prior_metrics = {}
        prior_publication = owner.prepare_publication(
            request | dict(manifest_path=str(path), manifest_sha256=digest),
            prior_metrics,
        )
        prior = owner.prepare_local(prior_publication, [TENSOR_IDS["dense"]], prior_metrics, False)
        prior.close(True)
        path, digest, _ = fixture(self.root / "v2", 2, "bitcomp-zstd")
        request = metadata(2) | dict(participants=participants)
        with patch.object(worker, "_prepare_arena", side_effect=RuntimeError("decode failed")):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                owner.prepare_publication(request | dict(manifest_path=str(path), manifest_sha256=digest), {})
        state = json.loads((owner.directory / "state.json").read_text())
        self.assertEqual(state["state"], "BUILDING")
        prior.close(True)
        self.assertEqual(json.loads((owner.directory / "state.json").read_text()), state)
        prepared_path = owner.directory / owner.shared_capacity["file"]
        owner.close()
        self.assertTrue(prepared_path.exists())
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            peer.prepare_publication(request | dict(manifest_path=str(path), manifest_sha256=digest), {})
        self.assertIsNone(peer.allocation)
        self.assertIsNone(peer._worker)
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            owner.prepare_publication(
                metadata(3)
                | dict(
                    participants=participants,
                    manifest_path=str(path),
                    manifest_sha256=digest,
                ),
                {},
            )

    def test_payload_reader_handles_short_reads_and_failure(self):
        path = self.root / "payload.bin"

        class ShortReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def readinto(self, destination):
                with destination[:64] as part:
                    return self.stream.readinto(part)

            def read(self, count):
                return self.stream.read(count)

        for data in (b"", bytes(range(251)) * 9, b"x" * 256):
            with self.subTest(bytes=len(data)):
                path.write_bytes(data)
                reader = ShortReader(path.open("rb", buffering=0))
                with (
                    patch.object(Path, "open", return_value=reader),
                    memoryview(bytearray(len(data))) as destination,
                ):
                    times = worker._read_verify_payload(
                        path,
                        destination,
                        {
                            "nbytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        },
                        False,
                    )
                    self.assertEqual(destination, data)
                    self.assertTrue(all(value >= 0 for value in times))
                self.assertTrue(reader.stream.closed)
        path.write_bytes(b"a" * 80)
        with memoryview(bytearray(100)) as destination:
            with self.assertRaisesRegex(ValueError, "truncated delta payload"):
                worker._read_verify_payload(path, destination, {"nbytes": 80, "sha256": "0" * 64}, True)
        with mmap.mmap(-1, 80) as mapping, memoryview(mapping) as destination:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                worker._read_verify_payload(path, destination, {"nbytes": 80, "sha256": "0" * 64}, False)
            _, hash_s = worker._read_verify_payload(path, destination, {"nbytes": 80, "sha256": "0" * 64}, True)
            self.assertEqual(destination, b"a" * 80)
            self.assertEqual(hash_s, 0)
            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                worker._read_verify_payload(path, destination, {"nbytes": 79, "sha256": "0" * 64}, True)

    def test_followers_wait_for_shared_ready_without_blocking_other_engines(self):
        path, digest, _ = fixture(self.root / "v1")
        first, follower, other = self.arena(), self.arena(rank=1), self.arena("b")
        previous = prepare_snapshot(first, path, digest, ["dense"], self.pool, {}, metadata())
        previous.close(True)
        path, digest, expected = fixture(self.root / "v2", 2)
        slow_pool = HostPayloadPool(2)
        self.addCleanup(slow_pool.close)
        entered, release, follower_started, follower_finished = [threading.Event() for _ in range(4)]
        snapshots, errors, metrics = [], [], {}
        decode = slow_pool.decode_zstd
        stale_checked, cancelled = threading.Event(), threading.Event()
        reads, locks = [], []
        read_bytes, flock = Path.read_bytes, host.fcntl.flock

        def observed_read(path):
            current = threading.current_thread()
            if current in threads:
                reads.append((current, path.name))
                if reads.count((threads[1], "state.json")) >= 3:
                    stale_checked.set()
            return read_bytes(path)

        def observed_lock(file, mode):
            current = threading.current_thread()
            if current in threads and mode != host.fcntl.LOCK_UN:
                locks.append((current, mode))
            return flock(file, mode)

        def blocked_decode(*args):
            entered.set()
            assert release.wait(5)
            return decode(*args)

        def build(arena, pool, selected, finished=None):
            try:
                if finished is not None:
                    follower_started.set()
                snapshots.append(
                    prepare_snapshot(
                        arena,
                        path,
                        digest,
                        selected,
                        pool,
                        metrics if finished is not None else {},
                        metadata(2),
                        cancelled,
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                if finished is not None:
                    finished.set()

        threads = [
            threading.Thread(target=build, args=(first, slow_pool, ["dense"])),
            threading.Thread(
                target=build,
                args=(follower, self.pool, ["expert"], follower_finished),
            ),
        ]
        with (
            patch.object(slow_pool, "decode_zstd", side_effect=blocked_decode),
            patch.object(Path, "read_bytes", new=observed_read),
            patch.object(host.fcntl, "flock", new=observed_lock),
        ):
            threads[1].start()
            try:
                self.assertTrue(follower_started.wait(5))
                self.assertTrue(stale_checked.wait(5))
                self.assertNotIn((threads[1], "index.json"), reads)
                threads[0].start()
                self.assertTrue(entered.wait(5))
                third = prepare_snapshot(other, path, digest, ["expert"], self.pool, {}, metadata(2, engine="b"))
                self.assertEqual(bytes(third.get(TENSOR_IDS["expert"])), expected["expert"])
                self.assertFalse(release.is_set())
                self.assertFalse(follower_finished.is_set())
                self.assertNotEqual(first.directory, other.directory)
                third.close()
            finally:
                release.set()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(5)
                if any(thread.is_alive() for thread in threads):
                    cancelled.set()
                    for thread in threads:
                        if thread.ident is not None:
                            thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(metrics["host_prepared_cache_reused"], 1)
        self.assertEqual(metrics["host_rank_outer_zstd_tensors"], 0)
        self.assertEqual(metrics["host_rank_cpu_workers"], 0)
        self.assertEqual(reads.count((threads[1], "index.json")), 1)
        self.assertNotIn((threads[0], "index.json"), reads)
        for thread, mode in locks:
            self.assertEqual(
                mode,
                host.fcntl.LOCK_EX if thread is threads[0] else host.fcntl.LOCK_SH | host.fcntl.LOCK_NB,
            )
        for snapshot in snapshots:
            snapshot.close()

    def test_rank_capacity_reuse_growth_and_late_encoded_release(self):
        path, digest, expected = fixture(self.root / "v1")
        arena, first_metrics = self.arena(), {}
        first = prepare_snapshot(arena, path, digest, expected, self.pool, first_metrics, metadata())
        original = arena.capacity.copy()
        self.assertEqual(
            original["capacity"],
            (first_metrics["host_rank_arena_bytes"] + 1023) // 1024 * 1024,
        )
        self.assertEqual(
            first_metrics["host_encoded_cache_capacity_bytes"],
            (sum(file["nbytes"] for file in json.loads(path.read_bytes())["files"]) + 1023) // 1024 * 1024,
        )
        request = metadata() | dict(manifest_path=str(path), manifest_sha256=digest)
        # The child returns its own unreleased READY image again; nothing is rebuilt.
        reused = {}
        with patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            again = arena.prepare_publication(request, reused)
        self.assertEqual(again.index, first.publication.index)
        self.assertEqual([call.args[0] for call in read.call_args_list], [arena.directory / "state.json"])
        for kind in ("encoded", "prepared"):
            self.assertEqual((reused[f"host_{kind}_cache_reused"], reused[f"host_{kind}_cache_created"]), (1, 0))
        first.close(True)
        # A released publication is never rebuilt, so its applied XOR mask cannot be served again.
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            arena.prepare_publication(request, {})
        path, digest, _ = fixture(self.root / "v2", 2)
        warm = {}
        second = prepare_snapshot(arena, path, digest, expected, self.pool, warm, metadata(2))
        self.assertEqual(arena.capacity, original)
        self.assertEqual(warm["host_rank_allocation_calls"], 0)
        self.assertEqual(warm["host_encoded_cache_allocation_calls"], 0)
        self.assertEqual(warm["host_rank_mapping_reused"], 1)
        first.close(True)
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(arena, path, digest, expected, self.pool, {}, metadata(3))
        second.close(True)
        large = hashlib.shake_256(b"growth").digest(original["capacity"] * 2 + 1)
        path, digest, expected = fixture(self.root / "v3", 3, dense=large)
        growth = {}
        third = prepare_snapshot(arena, path, digest, expected, self.pool, growth, metadata(3))
        self.assertEqual(bytes(third.get(TENSOR_IDS["dense"])), large)
        self.assertNotEqual(arena.capacity["identity"], original["identity"])
        self.assertEqual(arena.capacity["generation"], 2)
        self.assertEqual(growth["host_rank_allocation_calls"], 1)
        self.assertEqual(
            arena.capacity["capacity"],
            (2 * growth["host_rank_arena_bytes"] + 1023) // 1024 * 1024,
        )
        self.assertEqual(
            growth["host_encoded_cache_capacity_bytes"],
            (2 * sum(file["nbytes"] for file in json.loads(path.read_bytes())["files"]) + 1023) // 1024 * 1024,
        )
        third.close()
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(arena, path, digest, expected, self.pool, {}, metadata(4))
        arena.close()
        self.assertIsNone(arena.mapping)


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4-zstd", "lz4"])
def test_bounded_jobs_decode_raw_and_compressed_bytes_exactly(codec, tmp_path):
    expected = {f"t{i}": bytes([i % 251]) * (31 + i * 31) for i in range(97)}
    expected["t33"] = bytes(j % 251 for j in range(len(expected["t33"])))
    owners = [{}, {}]
    for i, (name, value) in enumerate(expected.items()):
        # Every third matrix lives in the second owner file; t33 spans two outer frames.
        owners[int(i % 3 == 0)][name] = matrix(value, parts=2 if i == 33 else 1, wrapped=codec != "lz4")
    expected |= {"raw": b"raw-target-value", "raw-large": bytes(range(256)) * 256}
    owners[0]["raw"], owners[1]["raw-large"] = expected["raw"], expected["raw-large"]
    path, _ = write_publication(tmp_path / "publication", owners, codec=codec)
    names, raw, encoded, file_spans, columns = arena_inputs(path)
    tensors = np.zeros(len(names), dtype=worker.TENSOR_DTYPE)
    tensors["prepared_nbytes"] = [len(expected[name]) for name in names]
    aligned = (tensors["prepared_nbytes"] + 15) // 16 * 16
    tensors["prepared_offset"] = np.cumsum(aligned) - aligned
    backing = bytearray(
        b"left" + bytes(int(np.max(tensors["prepared_offset"] + tensors["prepared_nbytes"]))) + b"right"
    )
    destination = memoryview(backing)[4:-5]
    order = np.arange(len(names))
    metrics = {name: 0 for name in worker._PREPARE_METRICS}
    pool = HostPayloadPool(2)
    try:
        with (
            patch.object(pool.executor, "submit", wraps=pool.executor.submit) as submit,
            patch.object(pool, "decode_zstd", wraps=pool.decode_zstd) as decode,
        ):
            worker._prepare_arena(
                memoryview(destination), tensors, raw, encoded, file_spans, columns, order, pool, metrics, codec
            )
        assert submit.call_count == 4 * pool.workers
        assert metrics["host_rank_outer_zstd_tensors"] == (0 if codec == "lz4" else 97)
        assert metrics["host_rank_encoded_copy_tensors"] == (97 if codec == "lz4" else 0)
        if codec == "lz4":
            decode.assert_not_called()
            assert metrics["host_rank_encoded_copy_bytes"] == sum(
                len(v) for k, v in expected.items() if not k.startswith("raw")
            )
        else:
            assert decode.call_count == submit.call_count
        for ordinal, name in enumerate(names):
            row = tensors[ordinal]
            start, count = int(row["prepared_offset"]), int(row["prepared_nbytes"])
            assert destination[start : start + count] == expected[name]
        assert backing[:4] == b"left" and backing[-5:] == b"right"
        # A span cannot read the following logical file in the shared backing. The production
        # writer never emits such an offset, so a copy of its columns is mutated by hand.
        for name in ("t1", "raw"):
            ordinal = names.index(name)
            offsets = columns["tensors.payload_offset"].copy()
            offsets[ordinal] = file_spans[columns["tensors.payload_file"][ordinal], 1]
            escaped = columns | {"tensors.payload_offset": offsets}
            with pytest.raises(ValueError, match="span escapes declared file"):
                worker._prepare_arena(
                    destination, tensors, raw, encoded, file_spans, escaped, order, pool, metrics, codec
                )
    finally:
        pool.close()


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4"])
def test_failed_tensor_drains_other_groups_before_releasing_views(codec, tmp_path):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    attempted, errors = [], []

    def decode_one(payload, destination):
        index = payload[0]
        attempted.append(index)
        if index == 0:
            raise ValueError("intentional bad frame")
        if index == 19:
            entered.set()
            assert release.wait(5)
        destination[:] = payload

    def decode(payloads, sizes, offsets, destination):
        for payload, size, offset in zip(payloads, sizes, offsets, strict=True):
            decode_one(memoryview(payload), destination[offset : offset + size])
        return 0.0

    frame = dict(encoded_offset=0, encoded_bytes=1, decoded_offset=0, decoded_bytes=1)
    outer = dict(encoded_bytes=1, decoded_bytes=1, frames=[frame] if codec.endswith("-zstd") else [])
    # One-byte payloads name their tensor; zero padding keeps sorted ordinals in that order.
    owner = {
        f"{i:02d}": dict(frames=[frame], payload=bytes([i]), outer=outer, changed_bytes=1, nbytes=1) for i in range(20)
    }
    path, _ = write_publication(tmp_path / "publication", [owner], codec=codec)
    _, raw, encoded, file_spans, columns = arena_inputs(path)
    tensors = np.zeros(20, dtype=worker.TENSOR_DTYPE)
    tensors["prepared_offset"] = np.arange(20)
    tensors["prepared_nbytes"] = 1
    pool = SimpleNamespace(workers=2, executor=ThreadPoolExecutor(max_workers=2), decode_zstd=decode)

    def run():
        try:
            worker._prepare_arena(
                memoryview(bytearray(20)),
                tensors,
                raw,
                encoded,
                file_spans,
                columns,
                np.arange(20),
                pool,
                {name: 0 for name in worker._PREPARE_METRICS},
                codec,
            )
        except ValueError as error:
            errors.append(str(error))
        finally:
            finished.set()

    def copy(destination, source):
        decode_one(memoryview(source), memoryview(destination))

    copy_patch = patch.object(worker, "copy_bytes", side_effect=copy)
    copy_patch.start()
    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    assert not finished.is_set()
    release.set()
    thread.join(5)
    pool.executor.shutdown(wait=True)
    copy_patch.stop()
    assert finished.is_set() and errors == ["intentional bad frame"]
    assert sorted(attempted) == [i for i in range(20) if i not in {8, 16}]


@pytest.mark.parametrize("finish", ["close", "failure"])
def test_cpu_process_reuses_exact_publications_without_torch(tmp_path, monkeypatch, finish):
    monkeypatch.setattr(host, "configured_cpu_workers", lambda: 2)
    cache = tmp_path / "cache"
    cache.mkdir()
    guard = tmp_path / "import-guard"
    launches, popen = [], subprocess.Popen

    def launch(command, **kwargs):
        # Bootstrap the actual isolated entry file; do not import this test module.
        code = f"""
import runpy, sys
from pathlib import Path
def guard_import(event, args):
    if event == 'import' and args[0].split('.')[0] in {{'torch', 'miles', 'sglang', 'flashinfer'}}:
        raise ImportError('GPU/model imports are forbidden in the CPU worker')
sys.addaudithook(guard_import)
Path({str(guard)!r}).touch()
entry = sys.argv.pop(1)
sys.argv[0] = entry
runpy.run_path(entry, run_name='__main__')
"""
        entry = command.index(str(Path(host.__file__).with_name("host_worker.py")))
        process = popen(command[:entry] + ["-c", code] + command[entry:], **kwargs)
        launches.append(process)
        return process

    monkeypatch.setattr(host.subprocess, "Popen", launch)
    namespace = {"stream_id": "stream", "participants": shared_participants()}
    client = host.CpuWorker(cache)
    previous, identities, metrics_by_version = None, [], []
    try:
        for version in (1, 2):
            path, digest, expected = fixture(tmp_path / str(version), version, "bitcomp-zstd")
            expected["foreign"] = FOREIGN
            expected_publication = {
                "manifest_path": str(path),
                "manifest_sha256": digest,
                "session_id": f"publication-{version}",
                "base_version": version - 1,
                "target_version": version,
                "plan_digest": "fixture-plan",
                "skip_payload_hash": False,
            }
            metrics = {}
            current = client.prepare(expected_publication, namespace, metrics, None)
            assert current["cpu_worker"]["pid"] == client.process.pid != os.getpid()
            identities.append(current["cpu_worker"])
            catalog = json.loads((cache / current["catalog"]["file"]).read_bytes())
            numeric = (cache / current["metadata"]["file"]).read_bytes()
            tensors, frames, gaps = worker._metadata_views(numeric, current["metadata_counts"])
            prepared = (cache / current["prepared"]["file"]).read_bytes()
            for ordinal, entry in enumerate(catalog["tensors"]):
                row = tensors[ordinal]
                start, size = int(row["prepared_offset"]), int(row["prepared_nbytes"])
                assert prepared[start : start + size] == expected[entry["name"]]
            assert frames.shape[0] == 4 and gaps.shape[1] == 2
            if previous is not None:
                assert all(current[k] == previous[k] for k in ("encoded", "prepared", "metadata"))
                assert all(
                    metrics[k] == 0
                    for k in (
                        "host_encoded_cache_allocation_calls",
                        "host_prepared_cache_allocation_calls",
                        "host_metadata_cache_allocation_calls",
                        "host_cpu_worker_spawn_s",
                    )
                )
            metrics_by_version.append(metrics)
            assert json.loads((cache / "state.json").read_text()) == {"token": current["token"], "state": "READY"}
            client.release(current["token"])
            assert json.loads((cache / "state.json").read_text()) == {"token": current["token"], "state": "REUSABLE"}
            previous = current
        assert identities[0] == identities[1] and guard.exists()
        assert metrics_by_version[0]["host_cpu_worker_spawn_s"] > 0
        if finish == "failure":
            bad = expected_publication | {
                "session_id": "bad-publication",
                "manifest_sha256": "0" * 64,
                "base_version": 2,
                "target_version": 3,
            }
            with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
                client.prepare(bad, namespace, {}, None)
            with pytest.raises(RuntimeError, match="cannot restart"):
                client.prepare(expected_publication, namespace, {}, None)
        client.close()
        assert client.process.poll() is not None and len(launches) == 1
        with pytest.raises(ChildProcessError):
            os.waitpid(client.process.pid, os.WNOHANG)
    finally:
        if client.process.poll() is None:
            client.close(terminate=True)
        client.close()


@pytest.mark.parametrize("abort", ["cancel", "child-exit"])
def test_cpu_process_failure_stops_writer_and_never_releases_publication(tmp_path, monkeypatch, abort):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(host, "_cache_base", lambda: cache)
    monkeypatch.setattr(host, "configured_cpu_workers", lambda: 2)
    ready_r, ready_w = os.pipe()
    release_r, release_w = os.pipe()
    cancelled, finished = threading.Event(), threading.Event()
    launches, errors, popen = [], [], subprocess.Popen

    def launch(command, **kwargs):
        code = f"""
import fcntl, json, os, sys
from multiprocessing.connection import Connection
from pathlib import Path
fd = int(sys.argv[sys.argv.index('--socket-fd') + 1])
connection = Connection(fd)
directory = Path(json.loads(connection.recv_bytes())['directory'])
# Like the real child, consume the prepare request before building.
assert json.loads(connection.recv_bytes())['command'] == 'prepare'
with (directory / '.lock').open('a+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    (directory / 'state.json').write_text(json.dumps({{'state': 'BUILDING'}}))
    with (directory / 'inflight.bin').open('wb') as pending:
        pending.write(b'started')
        pending.flush()
        os.write({ready_w}, b'1')
        os.read({release_r}, 1)
        pending.write(b'finished')
        pending.flush()
    os._exit(7)
"""
        entry = command.index(str(Path(host.__file__).with_name("host_worker.py")))
        kwargs["pass_fds"] += (ready_w, release_r)
        process = popen(command[:entry] + ["-c", code] + command[entry:], **kwargs)
        launches.append(process)
        return process

    monkeypatch.setattr(host.subprocess, "Popen", launch)
    path, digest, _ = fixture(tmp_path / "publication", codec="bitcomp-zstd")
    arena = host.HostArena("a", 0, shared_participants()[0])

    def build():
        try:
            arena.prepare_publication(
                metadata() | dict(manifest_path=str(path), manifest_sha256=digest),
                {},
                cancelled,
            )
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=build)
    thread.start()
    try:
        assert select.select([ready_r], [], [], 10)[0] and os.read(ready_r, 1) == b"1"
        assert not finished.is_set()
        with (arena.directory / ".lock").open("a+b") as lock:
            with pytest.raises(BlockingIOError):
                host.fcntl.flock(lock, host.fcntl.LOCK_EX | host.fcntl.LOCK_NB)
            assert json.loads((arena.directory / "state.json").read_text())["state"] == "BUILDING"
            if abort == "cancel":
                cancelled.set()
            else:
                os.write(release_w, b"1")
            thread.join(10)
            assert not thread.is_alive() and len(errors) == 1
            host.fcntl.flock(lock, host.fcntl.LOCK_EX | host.fcntl.LOCK_NB)
            host.fcntl.flock(lock, host.fcntl.LOCK_UN)
            assert json.loads((arena.directory / "state.json").read_text())["state"] == "BUILDING"
        assert launches[0].poll() is not None
        assert (arena.directory / "inflight.bin").read_bytes() == (
            b"started" if abort == "cancel" else b"startedfinished"
        )
        with pytest.raises(ChildProcessError):
            os.waitpid(launches[0].pid, os.WNOHANG)
        client = arena._worker
        arena.close()
        assert arena._worker is client
        with pytest.raises(RuntimeError, match="cannot restart"):
            client.prepare({}, {}, {}, None)
        assert len(launches) == 1
    finally:
        cancelled.set()
        thread.join(10)
        arena.close()
        for fd in (ready_r, ready_w, release_r, release_w):
            os.close(fd)


@pytest.mark.parametrize("reason", ["eof", "parent-death"])
def test_cpu_process_stops_on_eof_or_exact_parent_death(reason):
    # Keep the socket peer alive for parent-death, so EOF cannot explain exit.
    parent = subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    local, child = socket.socketpair()
    parent_fd = os.pidfd_open(parent.pid)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            worker.__file__,
            "--socket-fd",
            str(child.fileno()),
            "--parent-fd",
            str(parent_fd),
        ],
        pass_fds=(child.fileno(), parent_fd),
    )
    child.close()
    child_fd = os.pidfd_open(process.pid)
    try:
        if reason == "eof":
            local.close()
        else:
            parent.terminate()
            parent.wait(timeout=10)
            assert local.fileno() >= 0
        assert select.select([child_fd], [], [], 10)[0]
        assert process.wait(timeout=10) == (0 if reason == "eof" else 1)
    finally:
        local.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if parent.poll() is None:
            parent.terminate()
        parent.wait(timeout=10)
        parent.stdin.close()
        os.close(parent_fd)
        os.close(child_fd)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
