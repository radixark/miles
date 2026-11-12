"""Native host-arena and complete receiver checks.

Manual-only: Linux, CUDA, nvCOMP 6, Torch, Snappy, LZ4 and Zstandard are required.
Bitcomp uses one GPU and the Miles producer, without requiring DE. The two-engine
DE test requires two SM100/SM103 GPUs; its barriers coordinate the oracle, not
production preparation.
"""

import json
import mmap
import multiprocessing
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import lz4.block
import numpy as np
import pytest
import snappy
import zstandard as zstd


def storage_identity(value):
    return value.data_ptr(), value.untyped_storage().data_ptr(), value.stride(), value.device


def tiny_receiver(monkeypatch, directory, rank_id, weights, extra):
    """A DeepseekV3-shaped receiver with one U8 layer per `weights` matrix plus `extra` parameters.

    Every parameter keeps its tensor's device, shape and strides, so CPU-resident
    and padded views reach the receiver unchanged. F32 parameters keep an F32
    canonical dtype; every other parameter is U8. Parameter storage identities are
    taken before the backend exists, so admission cannot rebind one unnoticed.
    """
    import torch

    from miles.backends.sglang_utils.gpu_delta import checkpoint
    from miles.backends.sglang_utils.gpu_delta.receiver import GpuDeltaBackend
    from miles.utils.gpu_delta.host import host_cache_id

    device = torch.device("cuda", torch.cuda.current_device())
    model = torch.nn.Module()
    model.config = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], num_hidden_layers=len(weights))
    model.num_fused_shared_experts = 0
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in weights])
    for layer, weight in zip(model.model.layers, weights, strict=True):
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    for name, value in extra.items():
        setattr(model.model, name, torch.nn.Parameter(value, requires_grad=False))
    inventory = {
        name: {"dtype": "F32" if value.dtype == torch.float32 else "U8", "shape": list(value.shape)}
        for name, value in model.named_parameters()
    }
    monkeypatch.setattr(checkpoint, "read_canonical_checkpoint_inventory", lambda _: inventory)
    for name, value in (
        ("GPU_DELTA_HOST_CACHE_DIR", str(Path(directory) / "cache")),
        ("GPU_DELTA_CPU_WORKERS", "4"),
        ("GPU_DELTA_LAYERS_PER_BATCH", "1"),
        # The producer reads its payload hash policy when each writer is constructed.
        ("GPU_DELTA_SKIP_PAYLOAD_HASH", "0"),
    ):
        monkeypatch.setenv(name, value)
    identity = dict(
        engine_id=directory,
        rank_id=rank_id,
        host_cache_id=host_cache_id(directory),
        pid=os.getpid(),
        tp_rank=0,
        dp_rank=0,
        pp_rank=0,
    )
    # The serving worker's CUDA device comes from its model runner, never from a parameter.
    runner = SimpleNamespace(model=model, device="cuda", gpu_id=device.index)
    identities = {name: storage_identity(value) for name, value in model.named_parameters()}
    return model, inventory, identity, GpuDeltaBackend(runner, identity), identities


def publish(encoder, codec, directory, stream_id, base_version, target_version, inventory, before, after):
    """Seal the canonical XOR publication between `before` and `after` byte views of every tensor."""
    from miles.utils.gpu_delta.publication import PublicationWriter, plan_digest

    definitions = [
        dict(name=name, **meta, encoding="xor_raw_bytes" if len(meta["shape"]) <= 1 else "xor_bytes")
        for name, meta in inventory.items()
    ]
    matrices = [row for row in definitions if row["encoding"] == "xor_bytes"]
    encoded = encoder.finish_device(
        encoder.encode_device([(before[row["name"]], after[row["name"]]) for row in matrices])
    )
    writer = PublicationWriter(
        directory,
        stream_id,
        base_version,
        target_version,
        plan_digest(definitions),
        codec,
        frame_bytes=encoder.frame_bytes,
    )
    try:
        for row, result in zip(matrices, encoded, strict=True):
            writer.add_encoded_tensor(row["name"], *result[:4], row)
        for row in definitions:
            if row["encoding"] == "xor_raw_bytes":
                mask = np.bitwise_xor(before[row["name"]].numpy(), after[row["name"]].numpy())
                writer.add_raw_xor_tensor(row["name"], mask, row)
        return writer.finish()
    finally:
        writer.close()


@pytest.mark.parametrize("codec", ["bitcomp", "bitcomp-zstd"])
@pytest.mark.parametrize("residency", ["cuda", "mixed"])
def test_bitcomp_receiver_double_buffer_exact_updates_and_stale_target(codec, residency, monkeypatch):
    """Real producer/receiver, five batches, omitted frames and small XOR masks."""
    import torch

    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    device = torch.device("cuda", torch.cuda.current_device())
    frame_bytes = 1 << 16
    # Five one-layer batches; each row ends in a 140-byte frame after an unchanged middle frame.
    weights = [torch.zeros((1, 2 * frame_bytes + 140), dtype=torch.uint8, device=device) for _ in range(5)]
    # model.raw is the first parameter (direct parameters precede the layers); it
    # does not determine the serving worker's CUDA device.
    raw_device = "cpu" if residency == "mixed" else device
    host_storage = host_padding = None
    if residency == "mixed":
        columns, pitch, offset = frame_bytes + 72, frame_bytes + 88, 4
        host_storage = torch.full((2 * pitch + offset,), 165, dtype=torch.uint8)
        weight = host_storage.as_strided((2, columns), (pitch, 1), offset)
        weight.zero_()
        weights[0] = weight
        host_padding = torch.ones_like(host_storage, dtype=torch.bool)
        host_padding.as_strided(weight.shape, weight.stride(), offset).fill_(False)
    encoder = GpuBatchEncoder(device, codec=codec, frame_bytes=frame_bytes)
    with (
        tempfile.TemporaryDirectory(prefix="gpu-delta-bitcomp-", dir="/dev/shm") as directory,
        ThreadPoolExecutor(max_workers=1) as prepare_executor,
    ):
        model, inventory, identity, backend, identities = tiny_receiver(
            monkeypatch,
            directory,
            "bitcomp-original-rank",
            weights,
            extra={
                "raw": torch.zeros(17, dtype=torch.uint8, device=raw_device),
                "scalar": torch.zeros((), dtype=torch.uint8, device=raw_device),
            },
        )
        prepared = None
        try:
            for version in range(1, 4):
                before = {
                    name: value.detach().cpu().flatten().pin_memory() for name, value in model.named_parameters()
                }
                after = {name: value.clone().pin_memory() for name, value in before.items()}
                for name, value in after.items():
                    if name in {"model.raw", "model.scalar"}:
                        value.bitwise_xor_(version)
                    else:
                        value[:frame_bytes:4096].bitwise_xor_(version)
                        value[-140:].bitwise_xor_(version + 7)
                descriptor = publish(
                    encoder,
                    codec,
                    Path(directory) / str(version),
                    "tiny-receiver",
                    version - 1,
                    version,
                    inventory,
                    before,
                    after,
                )
                metadata = dict(
                    descriptor,
                    session_id="tiny-receiver",
                    validate_payload_hash=True,
                    participants=[identity],
                )
                # Match the scheduler's persistent preparation worker: CUDA
                # allocations on this foreground thread do not bind its context.
                prepared = prepare_executor.submit(backend.prepare, metadata).result()
                assert not prepared.hardware
                assert not prepared.decode_plan.decoded_slots and not prepared.decode_plan.input_slots
                assert backend.host_arena.tensor.is_pinned()
                assert prepared.timings["compressed_batches"] == 5
                assert prepared.timings["decoded_zero_bytes"] == 5 * frame_bytes
                assert prepared.timings["host_batch_plan_reused"] == int(version > 1)
                for name, value in model.named_parameters():
                    assert torch.equal(value.cpu().flatten(), before[name])
                if version == 1 and residency == "mixed":
                    prepared.close()
                    for name, value in model.named_parameters():
                        assert torch.equal(value.cpu().flatten(), before[name])
                    prepared = prepare_executor.submit(backend.prepare, metadata).result()
                if version == 3:
                    model.model.layers[0].weight = torch.nn.Parameter(
                        model.model.layers[0].weight.clone(), requires_grad=False
                    )
                    with pytest.raises(RuntimeError, match="parameter identity changed"):
                        prepared.apply()
                    for name, value in model.named_parameters():
                        assert torch.equal(value.cpu().flatten(), before[name])
                    prepared.close()
                else:
                    result = prepared.apply()
                    assert result["applied"] and result["target_version"] == version
                    assert result["timings"]["compressed_input_buffers"] == 2
                    assert result["timings"]["decoded_buffers"] == 2
                    assert result["timings"]["compressed_input_h2d_bytes"] > 0
                    assert result["timings"]["de_host_input_bytes"] == 0
                    for name, value in model.named_parameters():
                        assert torch.equal(value.cpu().flatten(), after[name])
                        assert storage_identity(value) == identities[name]
                    prepared.close(True)
                if host_storage is not None:
                    assert torch.all(host_storage[host_padding] == 165)
                prepared = None
        finally:
            if prepared is not None:
                prepared.close()
            backend.close()


def test_hf_relative_snapshots_are_reversible_with_the_same_update_api(monkeypatch):
    """Apply A, cancel A, skip B, apply C, cancel C, and apply an HF no-op."""
    import shutil
    from unittest.mock import patch

    import torch

    from miles.backends.sglang_utils.gpu_delta.bindings import DerivedImage
    from miles.backends.sglang_utils.gpu_delta.derived import DerivedUpdate, plan_updates
    from miles.backends.sglang_utils.gpu_delta.session import DeltaSession
    from miles.utils.gpu_delta import apply
    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    device = torch.device("cuda", torch.cuda.current_device())
    frame_bytes = 1 << 16
    encoder = GpuBatchEncoder(device, codec="bitcomp-zstd", frame_bytes=frame_bytes)
    with tempfile.TemporaryDirectory(prefix="gpu-delta-snapshot-", dir="/dev/shm") as directory:
        root = Path(directory)
        # 7-byte tail frames, a CPU raw vector and F32 scalars on CPU and CUDA
        # whose derived values are refreshed.
        model, inventory, identity, backend, identities = tiny_receiver(
            monkeypatch,
            directory,
            "snapshot-original",
            [torch.zeros((1, frame_bytes + 7), dtype=torch.uint8, device=device) for _ in range(2)],
            extra={
                "vector": torch.arange(17, dtype=torch.uint8),
                "scalar": torch.tensor(1.5),
                "scalar_gpu": torch.tensor(1.5, device=device),
            },
        )
        derived = torch.zeros_like(model.model.scalar_gpu)
        derived_cpu = torch.zeros_like(model.model.scalar)
        derived_vector = torch.zeros_like(model.model.vector, device=device)
        backend.layout.finish()
        sources = (model.model.scalar, model.model.scalar_gpu)
        backend.layout.derived_updates = plan_updates(
            [
                DerivedUpdate(
                    sources, (derived, derived_cpu), lambda values: (values[0] + values[1], values[0] + values[1])
                )
            ],
            sources,
        )
        backend.layout.derived_targets = frozenset(map(id, sources))
        backend.layout.derived.append(DerivedImage("host-vector", derived_vector, model.model.vector))
        consumers = (derived, derived_cpu, derived_vector)
        consumer_pointers = [value.data_ptr() for value in consumers]
        session = DeltaSession(identity, backend)
        base = {
            name: value.detach().reshape(-1).view(torch.uint8).cpu().pin_memory()
            for name, value in model.named_parameters()
        }
        published = {}
        worker = None
        try:
            for operation in ("A", "undo-A", "C", "undo-C", "HF"):
                undo = operation.startswith("undo-")
                label = operation.removeprefix("undo-")
                values = {name: value.clone().pin_memory() for name, value in base.items()}
                if label != "HF" and not undo:
                    values["model.layers.1.weight"][::127].bitwise_xor_(13 if label == "A" else 27)
                    if label == "A":
                        values["model.layers.0.weight"][::19].bitwise_xor_(6)
                        values["model.vector"].bitwise_xor_(43)
                    scalar = torch.tensor(3.0 if label == "A" else 7.5).reshape(-1).view(torch.uint8)
                    values["model.scalar"].copy_(scalar)
                    values["model.scalar_gpu"].copy_(scalar)
                if undo:
                    descriptor = published[label]
                else:
                    descriptor = published[label] = publish(
                        encoder, "bitcomp-zstd", root / label, "eval-snapshots", 0, 1, inventory, base, values
                    )
                request = dict(descriptor, session_id=operation, validate_payload_hash=True, participants=[identity])
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                before_derived = [value.clone() for value in consumers]
                assert session.prepare(request).result()["state"] == "PREPARED"
                prepared = session._update.prepared
                for name, value in model.named_parameters():
                    assert torch.equal(value, before[name])
                for value, old in zip(consumers, before_derived, strict=True):
                    assert torch.equal(value, old)
                if worker is None:
                    worker = backend.host_arena._worker
                assert backend.host_arena._worker is worker
                if operation != "A":
                    assert prepared.timings["apply_kernel_cache_misses"] == 0
                with patch.object(apply, "_compiled", side_effect=AssertionError("paused compilation")):
                    assert session.apply(operation, lambda: None, lambda: None, lambda: True)["state"] == "APPLIED"
                assert session.resume(operation, lambda version: None)["state"] == "RESUMED"
                session._update.release_future.result()
                for name, value in model.named_parameters():
                    assert torch.equal(value.detach().reshape(-1).view(torch.uint8).cpu(), values[name])
                    assert storage_identity(value) == identities[name]
                assert derived.item() == model.model.scalar.item() * 2
                assert derived_cpu.item() == derived.item()
                assert torch.equal(derived_vector.cpu(), model.model.vector)
                assert [value.data_ptr() for value in consumers] == consumer_pointers
                if undo or label == "HF":
                    shutil.rmtree(root / label)
            assert session.clear()["state"] == "CLEARED"
        finally:
            session._executor.shutdown(wait=True)
            if session._update.prepared is not None:
                session._update.prepared.close()
            backend.close()


@pytest.mark.parametrize("failure", ["before_apply", "during_apply", "decoder_status", "after_apply", "after_resume"])
def test_bitcomp_failed_transaction_requires_fresh_receiver_recovery(failure, monkeypatch):
    """Exercise transaction recovery around real CUDA writes, not fake payloads."""
    import torch

    from miles.backends.sglang_utils.gpu_delta.session import DeltaSession
    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    device = torch.device("cuda", torch.cuda.current_device())
    frame_bytes = 1 << 20
    encoder = GpuBatchEncoder(device, codec="bitcomp-zstd", frame_bytes=frame_bytes)
    sessions = []
    with tempfile.TemporaryDirectory(prefix="gpu-delta-ft-", dir="/dev/shm") as directory:
        root = Path(directory)

        def receiver(incarnation):
            # One 1 MiB frame per layer batch, so a failure can follow a partial apply.
            # Layer 1 and the raw vector are CPU-resident behind paused CUDA mirrors.
            model, inventory, identity, backend, _ = tiny_receiver(
                monkeypatch,
                directory,
                f"native-receiver-{incarnation}",
                [
                    torch.zeros((1, frame_bytes), dtype=torch.uint8, device=device),
                    torch.zeros((1, frame_bytes), dtype=torch.uint8),
                ],
                extra={"raw": torch.zeros(17, dtype=torch.uint8)},
            )
            sessions.append(DeltaSession(identity, backend))
            return model, inventory, sessions[-1]

        def fill_update(model, inventory, session, version):
            before = {name: value.detach().cpu().flatten().pin_memory() for name, value in model.named_parameters()}
            after = {name: torch.full_like(value, 16 + version).pin_memory() for name, value in before.items()}
            descriptor = publish(
                encoder,
                "bitcomp-zstd",
                root / str(version),
                "native-ft-stream",
                version - 1,
                version,
                inventory,
                before,
                after,
            )
            metadata = dict(
                descriptor,
                session_id=f"native-ft-update-{version}",
                validate_payload_hash=True,
                participants=[session.identity],
            )
            return metadata, before, after

        def prepare(session, metadata):
            session.prepare(metadata).result(timeout=90)
            assert session.status()["state"] == "PREPARED", session.status()

        def assert_weights(model, expected):
            for name, value in model.named_parameters():
                assert torch.equal(value.detach().cpu().flatten(), expected[name]), name

        try:
            model, inventory, session = receiver("original")
            metadata, before, after = fill_update(model, inventory, session, 1)
            prepare(session, metadata)
            assert_weights(model, before)
            prepared = session._update.prepared
            apply_batch = prepared._apply_batch
            applications, resumed = [], []

            def checked_batch(batch):
                apply_batch(batch)
                applications.append(batch)
                if failure == "during_apply":
                    raise RuntimeError("injected failure after the first CUDA batch")
                if failure == "decoder_status" and len(applications) == 2:
                    prepared.error.fill_(1)

            monkeypatch.setattr(prepared, "_apply_batch", checked_batch)

            def fence():
                if failure == "before_apply":
                    raise RuntimeError("injected reader fence failure")

            if failure in {"before_apply", "during_apply", "decoder_status"}:
                with pytest.raises(RuntimeError, match="injected|decompression failed"):
                    session.apply(metadata["session_id"], fence, lambda: None, None)
                torch.cuda.synchronize(device)
                assert session.status()["state"] == "POISONED"
                assert session.version == 0
                if failure == "before_apply":
                    assert_weights(model, before)
                else:
                    assert len(applications) == (2 if failure == "decoder_status" else 1)
                    assert torch.equal(
                        model.model.layers[0].weight.cpu().flatten(),
                        after["model.layers.0.weight"],
                    )
                    assert torch.equal(
                        model.model.layers[1].weight.cpu().flatten(),
                        before["model.layers.1.weight"],
                    )
                    assert torch.equal(model.model.raw, before["model.raw"])
                with pytest.raises(ValueError):
                    session.apply(metadata["session_id"], lambda: None, lambda: None, None)
                with pytest.raises(ValueError):
                    session.resume(metadata["session_id"], resumed.append)
                assert not resumed
            else:
                # The coordinator loses a reply after actual GPU writes. It must
                # quarantine this incarnation rather than resubmit the XOR.
                session.apply(metadata["session_id"], fence, lambda: None, None)
                assert session.status()["state"] == "APPLIED"
                assert_weights(model, after)
                if failure == "after_resume":
                    session.resume(metadata["session_id"], resumed.append)
                with pytest.raises(ValueError):
                    session.apply(metadata["session_id"], fence, lambda: None, None)
                assert len(applications) == 2
                assert_weights(model, after)
            assert resumed == ([1] if failure == "after_resume" else [])
            session._executor.shutdown(wait=True)
            if session._update.prepared is not None:
                session._update.prepared.close()
            session.backend.close()
            # Replacement starts from actual fresh model storage and a new rank
            # identity; only this incarnation may consume the base-zero payload.
            model, inventory, session = receiver("replacement")
            metadata = dict(metadata, participants=[session.identity])
            prepare(session, metadata)
            resumed = []
            session.apply(metadata["session_id"], lambda: None, lambda: None, None)
            session.resume(metadata["session_id"], resumed.append)
            assert_weights(model, after)
            metadata, before, after = fill_update(model, inventory, session, 2)
            prepare(session, metadata)
            assert_weights(model, before)
            session.apply(metadata["session_id"], lambda: None, lambda: None, None)
            session.resume(metadata["session_id"], resumed.append)
            assert session.version == 2 and resumed == [1, 2]
            assert_weights(model, after)
        finally:
            for session in sessions:
                session._executor.shutdown(wait=True)
                if session._update is not None and session._update.prepared is not None:
                    session._update.prepared.close()
                if session.backend is not None:
                    session.backend.close()


def _pinned_mapping_consumer(rank, path, expected, barrier, output):
    import torch

    from miles.utils.gpu_delta.memory import HostMapping

    torch.cuda.set_device(rank)
    stream = torch.cuda.Stream()
    with Path(path).open("r+b") as source:
        inode = (os.fstat(source.fileno()).st_dev, os.fstat(source.fileno()).st_ino)
        mapping = mmap.mmap(source.fileno(), 0)
    allocation = HostMapping(mapping)
    tensor = torch.frombuffer(allocation.view, dtype=torch.uint8)
    try:
        allocation.pin()
        assert tensor.is_pinned()
        with torch.cuda.stream(stream):
            copied = tensor.to("cuda", non_blocking=True)
        stream.synchronize()
        assert bytes(copied.cpu().numpy()) == expected
        barrier.wait(timeout=90)
        if rank == 0:
            tensor = None
            allocation.close()
        barrier.wait(timeout=90)
        if rank == 1:
            # Unregistering one rank must leave its peer's registration usable.
            with torch.cuda.stream(stream):
                copied = tensor.to("cuda", non_blocking=True)
            stream.synchronize()
            assert bytes(copied.cpu().numpy()) == expected
        output.put((rank, inode))
    finally:
        stream.synchronize()
        tensor = None
        allocation.close()
        mapping.close()


def test_shared_pinned_mapping_survives_peer_unregister():
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="gpu-delta-shared-pin-", dir="/dev/shm") as directory:
        path = Path(directory) / "prepared.bin"
        expected = random.Random(189).randbytes(1 << 20)
        path.write_bytes(expected)
        barrier, output = context.Barrier(2), context.Queue()
        children = [
            context.Process(
                target=_pinned_mapping_consumer,
                args=(rank, str(path), expected, barrier, output),
            )
            for rank in range(2)
        ]
        try:
            for child in children:
                child.start()
            rows = [output.get(timeout=120) for _ in children]
            for child in children:
                child.join(timeout=30)
                assert child.exitcode == 0
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=15)
        assert {row[0] for row in rows} == {0, 1}
        assert len({row[1] for row in rows}) == 1
        assert path.read_bytes() == expected


def _publication(directory, version, repeat, codec):
    """Real Snappy or LZ4 inner frames; the DE oracle decodes Snappy after 1 MiB outer Zstd chunks."""
    from miles.utils.gpu_delta.publication import PublicationWriter

    expected = {f"tensor-{i}": random.Random(i + version).randbytes(256 * repeat) for i in range(8)}
    writer = PublicationWriter(
        directory,
        "native-shared-stream",
        version - 1,
        version,
        "native-host-plan",
        codec,
        frame_bytes=(1 << 19) if version < 3 else (1 << 20),
    )
    try:
        for name, value in expected.items():
            if codec == "snappy-zstd":
                inner = snappy.compress(value)
            else:
                inner = lz4.block.compress(value, store_size=False)
            payload, outer_frames = bytearray(), []
            for offset in range(0, len(inner), 1 << 20) if codec == "snappy-zstd" else ():
                chunk = inner[offset : offset + (1 << 20)]
                payload.extend(bytes((-len(payload)) % 16))
                compressed = zstd.ZstdCompressor(write_checksum=True).compress(chunk)
                outer_frames.append(
                    dict(
                        encoded_offset=len(payload),
                        encoded_bytes=len(compressed),
                        decoded_offset=offset,
                        decoded_bytes=len(chunk),
                    )
                )
                payload.extend(compressed)
            payload = bytes(payload) if outer_frames else inner
            writer.add_encoded_tensor(
                name,
                [dict(encoded_offset=0, encoded_bytes=len(inner), decoded_offset=0, decoded_bytes=len(value))],
                payload,
                dict(encoded_bytes=len(payload), decoded_bytes=len(inner), frames=outer_frames),
                len(value),
                dict(dtype="U8", shape=[1, 1 << 20]),
            )
        expected["raw"] = b"unaligned-raw-target"
        writer.add_raw_xor_tensor("raw", expected["raw"], dict(dtype="U8", shape=[len(expected["raw"])]))
        descriptor = writer.finish()
    finally:
        writer.close()
    return Path(descriptor["manifest_path"]), descriptor["manifest_sha256"], expected


def _consumer(rank, engine, workers, publications, cache, barrier, output):
    import torch

    from miles.utils.gpu_delta import host as host
    from miles.utils.gpu_delta import memory as memory
    from miles.utils.gpu_delta.codec import NvcompDecoder

    os.environ["GPU_DELTA_HOST_CACHE_DIR"] = cache
    # Exercise the exact capacity-growth algorithm with small oracle tensors.
    # Production's coarser alignment is not a wire/codec requirement.
    host._CAPACITY_ALIGNMENT = 1 << 20
    torch.cuda.set_device(rank % 2)
    device = torch.device("cuda", rank % 2)
    os.environ["GPU_DELTA_CPU_WORKERS"] = str(workers)
    identity = host.host_cache_id(engine)
    participants = [
        dict(
            engine_id=engine,
            rank_id=f"{engine}-original-{local}",
            host_cache_id=identity,
            pid=100 + local,
            tp_rank=local,
            dp_rank=0,
            pp_rank=0,
        )
        for local in range(2)
    ]
    arena = host.HostArena(engine, device.index, participants[rank % 2])
    stream = torch.cuda.Stream(device=device)
    records = []
    try:
        names = {f"tensor-{i}" for i in range(4 * (rank % 2), 4 * (rank % 2 + 1))} | {"raw"}
        for version, (path, digest, expected) in enumerate(publications, 1):
            metadata = dict(
                stream_id="native-shared-stream",
                session_id=f"update-{version}",
                base_version=version - 1,
                target_version=version,
                plan_digest="native-host-plan",
                validate_payload_hash=True,
                participants=participants,
            )
            barrier.wait(timeout=90)
            metrics = {}
            publication = arena.prepare_publication(
                metadata | dict(manifest_path=str(path), manifest_sha256=digest),
                metrics,
            )
            codec = publication.header["codec"]
            if version == 1:
                ordinals = {entry["name"]: i for i, entry in enumerate(publication.read_catalog()["tensors"])}
            selected = [ordinals[name] for name in sorted(names)]
            snapshot = arena.prepare_local(publication, selected, metrics)
            assert metrics["host_rank_staging_copy_bytes"] == int(
                publication.tensors["prepared_nbytes"][selected].sum()
            )
            source = arena.tensor
            frames, offsets, size = [], {}, 0
            for name in sorted(names):
                ordinal = ordinals[name]
                if name == "raw":
                    assert bytes(snapshot.get(ordinal)) == expected[name]
                    continue
                size = (size + 15) // 16 * 16
                offsets[name] = size
                row = publication.tensors[ordinal]
                assert row["frame_count"] == 1
                frame = publication.frames[:, row["frame_start"]]
                frames.append(
                    (
                        int(snapshot.offsets[ordinal] + frame[0]),
                        int(frame[1]),
                        int(frame[2]),
                        size + int(frame[3]),
                    )
                )
                size += len(expected[name])
            decoder = NvcompDecoder(device, codec.removesuffix("-zstd"))
            table = np.asarray(frames, dtype=np.int64).T.copy()
            plan = decoder.prepare_batches(table, [len(frames)], source, stream)
            # DE writes only to hardware-decompression-capable outputs; bind_outputs does not recheck them.
            decoded = [memory.device_empty(size, dtype=torch.uint8, device=device) for _ in range(2)]
            stream.wait_stream(torch.cuda.current_stream(device))
            plans = plan.bind_outputs(decoded)
            with torch.cuda.stream(stream):
                plans[0].enqueue()
                complete = torch.cuda.Event()
                complete.record(stream)
            complete.synchronize()
            assert plans[0].statuses.tolist() == [0] * len(frames)
            assert plans[0].actual_sizes.tolist() == [frame[2] for frame in frames]
            host_copy = decoded[0].cpu()
            for name, offset in offsets.items():
                assert bytes(host_copy[offset : offset + len(expected[name])].numpy()) == expected[name]
            # Substitute the Miles all-original-engine-rank completion barrier with
            # an explicit two-process completion barrier in this isolated oracle.
            barrier.wait(timeout=90)
            snapshot.close(True)
            source = None
            assert arena.allocation is not None
            del plan, plans, decoded
            driver = arena.allocation.driver
            properties = memory._check(
                driver.cuMemGetAllocationPropertiesFromHandle(arena.allocation.handle),
                "native allocation properties",
            )
            granularity = memory._check(
                driver.cuMemGetAllocationGranularity(
                    properties,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "native allocation granularity",
            )
            records.append(
                dict(
                    version=version,
                    host_cache_id=identity,
                    arena_identity=arena.capacity["identity"],
                    local_names=sorted(names),
                    mapping_pointer=arena.tensor.data_ptr(),
                    metrics=metrics,
                    exact_host_de_bytes=True,
                    allocation_granularity=granularity,
                )
            )
        stream.synchronize()
        arena.close()
        assert arena.allocation is None and arena.mapping is None
        assert arena.shared_allocation is None
        output.put(dict(rank=rank, engine_id=engine, updates=records, final_unmapped=True))
    finally:
        stream.synchronize()
        arena.close()


@pytest.mark.parametrize("workers", [4, 8])
def test_two_engines_reuse_host_de_capacity_and_grow(workers, monkeypatch):
    context = multiprocessing.get_context("spawn")
    # The producer reads its payload hash policy when each writer is constructed.
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "0")
    with tempfile.TemporaryDirectory(prefix="gpu-delta-native-", dir="/dev/shm") as directory:
        root = Path(directory)
        publications = []
        for version, repeat in enumerate((2048, 1024, 4096), 1):
            path, digest, expected = _publication(
                root / str(version), version, repeat, ("lz4", "snappy-zstd", "lz4")[version - 1]
            )
            publications.append((str(path), digest, expected))
        barriers, output = [context.Barrier(2), context.Barrier(2)], context.Queue()
        processes = [
            context.Process(
                target=_consumer,
                args=(
                    rank,
                    f"engine-{rank // 2}",
                    workers,
                    publications,
                    str(root / "cache"),
                    barriers[rank // 2],
                    output,
                ),
            )
            for rank in range(4)
        ]
        try:
            for process in processes:
                process.start()
            records = [output.get(timeout=240) for _ in processes]
            for process in processes:
                process.join(timeout=90)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=15)
        for engine in ("engine-0", "engine-1"):
            engine_records = [record for record in records if record["engine_id"] == engine]
            assert len(engine_records) == 2
            for version in range(3):
                rows = [record["updates"][version] for record in engine_records]
                assert rows[0]["arena_identity"] != rows[1]["arena_identity"]
                assert sum(row["metrics"]["host_rank_cpu_workers"] for row in rows) == workers
                assert sum(row["metrics"]["host_prepared_cache_created"] for row in rows) == 1
                assert sum(row["metrics"]["host_prepared_cache_reused"] for row in rows) == 1
                assert all(row["metrics"]["host_rank_staging_copy_bytes"] > 0 for row in rows)
                assert set(rows[0]["local_names"]) & set(rows[1]["local_names"]) == {"raw"}
                assert sum(row["metrics"]["host_encoded_cache_created"] for row in rows) == 1
                assert sum(row["metrics"]["host_encoded_cache_hash_files"] for row in rows) == 1
                assert sum(row["metrics"]["host_rank_outer_zstd_tensors"] for row in rows) == (
                    8 if version == 1 else 0
                )
                assert sum(row["metrics"]["host_rank_encoded_copy_tensors"] for row in rows) == (
                    0 if version == 1 else 8
                )
                assert sum(row["metrics"]["host_rank_allocation_calls"] for row in rows) == (0 if version == 1 else 2)
                assert all(row["metrics"]["host_rank_mapping_reused"] == int(version == 1) for row in rows)
        assert len({record["updates"][0]["host_cache_id"] for record in records}) == 2
        assert len({record["updates"][0]["arena_identity"] for record in records}) == 4
        for record in records:
            a, b, c = record["updates"]
            for row, multiplier in ((a, 1), (c, 2)):
                alignment = max(1 << 20, row["allocation_granularity"])
                assert row["metrics"]["host_rank_capacity_bytes"] == (
                    (multiplier * row["metrics"]["host_rank_arena_bytes"] + alignment - 1) // alignment * alignment
                )
            assert a["arena_identity"] == b["arena_identity"] != c["arena_identity"]
            assert a["mapping_pointer"] == b["mapping_pointer"] != c["mapping_pointer"]
            assert [row["metrics"]["host_rank_capacity_generation"] for row in record["updates"]] == [1, 1, 2]
            assert all(
                row["metrics"]["host_rank_arena_bytes"] <= row["metrics"]["host_rank_capacity_bytes"]
                for row in record["updates"]
            )
            assert record["final_unmapped"]
        print(
            json.dumps(
                dict(
                    status="PASS",
                    cpu_workers=workers,
                    logical_engines=2,
                    ranks_per_engine=2,
                    updates=3,
                    records=records,
                ),
                sort_keys=True,
            )
        )
