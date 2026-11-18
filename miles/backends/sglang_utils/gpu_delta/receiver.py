"""CUDA-only receiver preparation and double-buffer transfer/decode/apply scheduling."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from miles.backends.sglang_utils.gpu_delta.bindings import REMOTE_EXPERT, TensorBinding, prepare_raw_xor
from miles.backends.sglang_utils.gpu_delta.byte_views import _byte_view
from miles.backends.sglang_utils.gpu_delta.checkpoint import read_canonical_checkpoint
from miles.backends.sglang_utils.gpu_delta.derived import prepare_updates
from miles.backends.sglang_utils.gpu_delta.layout import GpuDeltaLayout
from miles.utils.gpu_delta import profiling
from miles.utils.gpu_delta.codec import NvcompDecoder
from miles.utils.gpu_delta.host import HostArena
from miles.utils.gpu_delta.host_worker import copy_bytes
from miles.utils.gpu_delta.publication import CODECS, catalog_entry, plan_digest, tensor_metadata


class GpuDeltaBackend:
    """Scheduler-owned plan; large decoded masks exist only during apply."""

    def __init__(self, model_runner, identity, shadows):
        self.binding_table = None
        self.batch_plan = None
        self.raw_plan = None
        self.layers_per_batch = int(os.environ.get("GPU_DELTA_LAYERS_PER_BATCH", "1"))
        if self.layers_per_batch < 1:
            raise ValueError("GPU_DELTA_LAYERS_PER_BATCH must be positive")
        try:
            checkpoint = read_canonical_checkpoint(model_runner)
            self.layout = GpuDeltaLayout(model_runner.model, checkpoint, shadows)
            definitions = (
                {"name": name, **metadata}
                for name, metadata in self.layout.inventory.items()
                if self.layout.excluded.get(name) in (None, REMOTE_EXPERT)
            )
            self.plan_digest = plan_digest(definitions, checkpoint.content_digest)
        except ValueError as error:
            # A failed describe may reply without an identity; name the rank rejecting its checkpoint or layout.
            raise ValueError(f"tp_rank={identity['tp_rank']} dp_rank={identity['dp_rank']}: {error}") from error
        self.device = torch.device(model_runner.device, model_runner.gpu_id)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("direct GPU deltas require an explicit CUDA device")
        self.host_arena = HostArena(identity["engine_id"], self.device.index, identity)
        self.decoders = {}
        self.apply_stream = None
        self.io_stream = None

    def prepare(self, request, cancelled=None):
        # Called by the background worker. Small scalar calibration may be read
        # to prepare derived values; no live writes, collectives, loader hooks
        # or generation-stream synchronization occur here.
        with torch.cuda.device(self.device):
            prepared = PreparedDelta(self)
            try:
                prepared.prepare(request, cancelled)
                return prepared
            except BaseException:
                self.discard(prepared)
                raise

    def discard(self, prepared):
        # A failed GPU drain must retain every host mapping its readers may use.
        prepared.close()
        self.close()

    def close(self):
        self.host_arena.close()


@dataclass
class _PreparedBatch:
    decoder: object
    apply: Callable[[], None]
    zero_ranges: list[torch.Tensor]
    check_status: Callable[[], None]


class _BindingTable:
    """Rank-local geometry; only publication row values change between updates."""

    def __init__(self, layout, entries):
        self.bindings = tuple(layout.bindings)
        self.ordinals = np.asarray([entries[b.name][0] for b in self.bindings], dtype=np.int64)
        self.nbytes = np.asarray([entries[b.name][1]["nbytes"] for b in self.bindings], dtype=np.int64)
        self.raw = np.asarray([b.is_raw for b in self.bindings], dtype=bool)
        non_layers = {"embed_tokens": [], "lm_head": [], "standalone": []}
        layers, storage_rows = {}, {}
        for row, binding in enumerate(self.bindings):
            if binding.layer is None:
                # Batch by the live module: DeepSeek-V4.1's canonical embed and head name model.embed_tokens and lm_head.
                live_name = binding.name if binding.live_name is None else binding.live_name
                module = live_name.rpartition(".")[0].rpartition(".")[2]
                non_layers.get(module, non_layers["standalone"]).append(row)
            else:
                layers.setdefault(binding.layer, []).append(row)
            for pointer in binding.storage_pointers:
                storage_rows.setdefault(pointer, []).append(row)
        self.non_layers = tuple(np.asarray(rows, np.int64) for rows in non_layers.values())
        self.layers = tuple(np.asarray(rows, np.int64) for rows in layers.values())
        self.derived = tuple(layout.derived)
        self.computed = tuple(layout.computed)
        # An image refreshes when any binding sharing one of its source storages is active.
        pairs = [
            (row, index)
            for index, image in enumerate((*self.derived, *self.computed))
            for pointer in image.source_pointers
            for row in storage_rows.get(pointer, ())
        ]
        self.image_rows, self.image_indices = np.asarray(pairs, np.int64).reshape(-1, 2).T

    def select(self, tensors):
        local = tensors[self.ordinals]
        matrices = (local["frame_count"] != 0) & ~self.raw
        raw = (local["changed_bytes"] != 0) & self.raw
        active = matrices | raw
        images = np.zeros(len(self.derived) + len(self.computed), dtype=bool)
        images[self.image_indices[active[self.image_rows]]] = True
        copies, computed = images[: len(self.derived)], images[len(self.derived) :]
        return (
            np.flatnonzero(matrices),
            np.flatnonzero(raw),
            [self.derived[index] for index in np.flatnonzero(copies)],
            [self.computed[index] for index in np.flatnonzero(computed)],
        )


@profiling.annotate("gpu_delta.prepare.layer_plan")
def _plan_layers(backend, table, rows):
    """Cache membership, canonical offsets and apply geometry, never wire frames."""
    key = rows.tobytes()
    if backend.batch_plan is None or backend.batch_plan[0] != key:
        from miles.utils.gpu_delta.apply import plan_apply

        active = np.zeros(len(table.bindings), dtype=bool)
        active[rows] = True
        groups = [group[active[group]] for group in table.non_layers]
        groups = [group for group in groups if len(group)]
        model_layers = [group[active[group]] for group in table.layers]
        model_layers = [group for group in model_layers if len(group)]
        layers_per_batch = backend.layers_per_batch
        groups.extend(
            np.concatenate(model_layers[start : start + layers_per_batch])
            for start in range(0, len(model_layers), layers_per_batch)
        )
        plans = []
        for group in groups:
            outputs, size = [], 0
            for row in group:
                offset = (size + 15) // 16 * 16
                binding, nbytes = table.bindings[row], int(table.nbytes[row])
                outputs.append((binding, offset, nbytes))
                size = offset + nbytes
            # Segment views share one decoded canonical payload. Only the apply
            # plan expands them; DE membership and output offsets stay unique.
            apply_outputs = [
                (view, offset, nbytes) for binding, offset, nbytes in outputs for view in binding.apply_bindings()
            ]
            tensor_offsets = np.stack((table.ordinals[group], [offset for _, offset, _ in outputs]))
            plans.append((tensor_offsets, size, plan_apply(apply_outputs)))
        backend.batch_plan = (key, plans)
    return backend.batch_plan[1]


@profiling.annotate("gpu_delta.prepare.frame_plan")
def _plan_decode(plans, publication, offsets):
    """Rebase shared numeric frame and gap slices into this rank's layer batches."""
    batches = [publication.tensors[outputs[0]] for outputs, _, _ in plans]
    counts = [int(tensors["frame_count"].sum()) for tensors in batches]
    table = np.empty((4, sum(counts)), dtype=np.int64)
    gaps, position = [], 0
    for (outputs, _, _), tensors, count in zip(plans, batches, counts, strict=True):
        tensor_ids, output_offsets = outputs
        frame_counts = tensors["frame_count"]
        starts = tensors["frame_start"] - (np.cumsum(frame_counts) - frame_counts)
        indices = np.repeat(starts, frame_counts) + np.arange(count, dtype=np.int64)
        destination = table[:, position : position + count]
        destination[:] = publication.frames[:, indices]
        destination[0] += np.repeat(offsets[tensor_ids], frame_counts)
        destination[3] += np.repeat(output_offsets, frame_counts)
        position += count

        gap_counts = tensors["gap_count"]
        starts = tensors["gap_start"] - (np.cumsum(gap_counts) - gap_counts)
        indices = np.repeat(starts, gap_counts) + np.arange(int(gap_counts.sum()), dtype=np.int64)
        batch_gaps = publication.gaps[indices]
        batch_gaps[:, 0] += np.repeat(output_offsets, gap_counts)
        gaps.append(batch_gaps)
    return table, counts, gaps


@profiling.annotate("gpu_delta.prepare.canonical_plan")
def _qualify_canonical_plan(backend, publication):
    """Bind a canonical publication to the independently prepared local layout."""
    digest = publication.header["plan_digest"]
    if digest != backend.plan_digest:
        raise ValueError("publication does not match the receiver checkpoint bytes and canonical tensor plan")
    if backend.binding_table is not None:
        return backend.binding_table, True
    catalog = publication.read_catalog()
    if catalog["plan_digest"] != digest:
        raise ValueError("shared canonical catalog plan changed")
    entries = {entry["name"]: (ordinal, entry) for ordinal, entry in enumerate(catalog["tensors"])}
    if len(entries) != len(catalog["tensors"]):
        raise ValueError("duplicate canonical tensor")
    bound = {binding.name for binding in backend.layout.bindings}
    for name, (_, entry) in entries.items():
        # Only a tensor this rank binds, or a routed expert another rank owns, may carry a delta;
        # any other exclusion, or an unbound tensor without one, would silently drop its mask.
        exclusion = backend.layout.excluded.get(name)
        if not (exclusion == REMOTE_EXPERT or (exclusion is None and name in bound)):
            raise ValueError(f"delta publication has an unadmitted tensor: {name}")
        spec = {"name": name, **backend.layout.inventory[name]}
        if entry != catalog_entry(tensor_metadata(**spec)):
            raise ValueError(f"canonical tensor metadata mismatch: {name}")
    expected = bound | {name for name, reason in backend.layout.excluded.items() if reason == REMOTE_EXPERT}
    missing = expected - entries.keys()
    if missing:
        raise ValueError(f"publication is missing canonical tensors: {sorted(missing)}")
    # Tensor selection and physical layout are receiver-local. The publication
    # never contains TP/EP slices, backend permutations or receiver identities.
    table = _BindingTable(backend.layout, entries)
    backend.binding_table = table
    return table, False


class PreparedDelta:
    def __init__(self, backend):
        self.stream = None
        self.io_stream = None
        self.host_snapshot = None
        self._gpu_complete = False
        self.raw_apply = None
        self.raw_apply_inputs = None
        self.raw_pointer_host = None
        self.cpu_storages, self.cpu_mirrors = {}, {}
        self.cpu_target_groups = []
        self.cpu_derived = []
        self.batches = []
        self.timings = {}
        self.status_checks = []
        self.apply_bindings, self.apply_source_patches = [], []
        self.apply_host_view = None
        self.backend, self.device = backend, backend.device

    @profiling.annotate("gpu_delta.prepare.layout")
    def prepare(self, request, cancelled=None):
        backend = self.backend
        preparation_started = time.perf_counter()
        publication = backend.host_arena.prepare_publication(request, self.timings, cancelled)
        backend.layout.finish()
        self.codec = publication.header["codec"]
        inner_codec = CODECS[self.codec][0]
        if inner_codec not in backend.decoders:
            backend.decoders[inner_codec] = NvcompDecoder(self.device, inner_codec)
        self.decoder = backend.decoders[inner_codec]
        self.hardware = self.decoder.hardware
        self.target_version = publication.header["target_version"]
        plan_started = time.perf_counter()
        table, reused_plan = _qualify_canonical_plan(backend, publication)
        self.timings["host_plan_validate_s"] = time.perf_counter() - plan_started
        self.timings["host_plan_cache_reused"] = int(reused_plan)
        self.host_snapshot = backend.host_arena.prepare_local(publication, table.ordinals, self.timings, self.hardware)
        self.timings["host_rank_prepare_s"] = time.perf_counter() - preparation_started

        tensors_started = time.perf_counter()
        compressed, raw_rows, self.derived, self.computed = table.select(publication.tensors)
        previous_plan = backend.batch_plan
        self.static_plans = _plan_layers(backend, table, compressed)
        self.timings["host_batch_plan_reused"] = int(previous_plan is not None and backend.batch_plan is previous_plan)
        frame_table, frame_counts, self.gaps = _plan_decode(
            self.static_plans,
            publication,
            self.host_snapshot.offsets,
        )
        self.max_decoded = max((plan[1] for plan in self.static_plans), default=0)
        self.matrix_tensor_count = len(compressed)
        self.raw_tensor_count = len(raw_rows)
        self.timings["host_tensor_prepare_s"] = time.perf_counter() - tensors_started

        # Pack the small uncompressed XOR masks once on the host. Decoded-mask
        # storage waits for the actual serving pause; kernel warmup is below.
        raw_started = time.perf_counter()
        raw_targets, raw_h2d_bytes = [], 0
        for row in raw_rows:
            binding, size = table.bindings[row], int(table.nbytes[row])
            position = (raw_h2d_bytes + 7) // 8 * 8
            raw_targets.append((binding, position, size))
            raw_h2d_bytes = position + size
        self.raw_pinned = torch.empty(raw_h2d_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        raw_view = memoryview(self.raw_pinned.numpy())
        for ordinal, (_, position, size) in zip(table.ordinals[raw_rows], raw_targets, strict=True):
            copy_bytes(raw_view[position : position + size], self.host_snapshot.get(int(ordinal)))
        self.timings.update(
            host_raw_pack_s=time.perf_counter() - raw_started,
            raw_tensors=len(raw_rows),
            raw_bytes=int(table.nbytes[raw_rows].sum()),
            raw_h2d_bytes=raw_h2d_bytes,
            compressed_batches=len(self.static_plans),
            computed_images=len(self.computed),
            layers_per_batch=backend.layers_per_batch,
            compressed_tensors=self.matrix_tensor_count,
            de_host_input_bytes=int(frame_table[1].sum()) if self.hardware else 0,
            decoded_zero_ranges=sum(map(len, self.gaps)),
            decoded_zero_bytes=sum(int(gaps[:, 1].sum()) for gaps in self.gaps),
        )
        self._prepare_gpu_metadata(frame_table, frame_counts, raw_targets)
        self.timings["host_prepare_s"] = time.perf_counter() - preparation_started

    @profiling.annotate("gpu_delta.prepare.gpu_metadata")
    def _prepare_gpu_metadata(self, frame_table, frame_counts, raw_targets):
        """Prepare small GPU inputs; retain their leases, not wire-frame objects."""
        started = time.perf_counter()
        backend = self.backend
        if backend.apply_stream is None:
            apply_stream = torch.cuda.Stream(device=self.device)
            io_stream = torch.cuda.Stream(device=self.device)
            backend.apply_stream, backend.io_stream = apply_stream, io_stream
        self.stream, self.io_stream = backend.apply_stream, backend.io_stream
        self.decoded_ready = [torch.cuda.Event() for _ in range(2)]
        self.decoded_free = [torch.cuda.Event() for _ in range(2)]
        with torch.cuda.stream(self.stream):
            raw_updates = self._prepare_raw_masks(raw_targets)
            self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
            from miles.utils.gpu_delta.apply import kernel_cache_info

            compile_started = time.perf_counter()
            cache_before = kernel_cache_info()
            metadata_reused = self._prepare_raw_apply(raw_updates)
            for _, _, group in self.static_plans:
                metadata_reused = group.prepare_metadata(self.device.index) or metadata_reused
            cache_after = kernel_cache_info()
            self.timings.update(
                host_apply_kernel_prepare_s=time.perf_counter() - compile_started,
                apply_kernel_cache_misses=cache_after.misses - cache_before.misses,
                apply_kernel_cache_reused=int(
                    (metadata_reused or cache_after.hits > cache_before.hits)
                    and cache_after.misses == cache_before.misses
                ),
            )
            self.apply_host_metadata = torch.empty(
                sum(2 * len(group.sources) + group.metadata_size for _, _, group in self.static_plans),
                dtype=torch.int64,
                pin_memory=True,
            )
            self.apply_metadata = torch.empty(self.apply_host_metadata.numel(), dtype=torch.int64, device=self.device)
            self._prepare_apply_bindings()
        self._prepare_decode_plan(frame_table, frame_counts)
        # PREPARED includes small input transfers, never output-slot allocation,
        # decoding or synchronization with rollout. The fixed apply kernel
        # is already compiled/loaded; membership never specializes its code.
        ready = []
        for stream in (self.stream, self.io_stream):
            event = torch.cuda.Event()
            event.record(stream)
            ready.append(event)
        apply_groups = [group for _, _, group in self.static_plans]
        raw_metadata = 0
        if self.raw_apply_inputs is not None:
            apply_groups.append(backend.raw_plan[1])
            raw_metadata = sum(t.numel() * t.element_size() for t in self.raw_apply_inputs[1:])
        self.timings.update(
            apply_metadata_h2d_bytes=self.apply_metadata.numel() * 8 + raw_metadata,
            apply_descriptor_h2d_bytes=sum(group.metadata_size for group in apply_groups) * 8,
            apply_groups=len(apply_groups),
            apply_grid_ctas=sum(group.grid_ctas for group in apply_groups),
            apply_contracts=sum(len(group.contracts) for group in apply_groups),
            apply_word32_contracts=sum(group.word32_contracts for group in apply_groups),
        )
        waiting = time.perf_counter()
        for event in ready:
            event.synchronize()
        self.timings.update(
            host_metadata_prepare_s=time.perf_counter() - started,
            host_metadata_wait_s=time.perf_counter() - waiting,
            decoder_metadata_h2d_bytes=frame_table.nbytes,
            decoder_workspace_bytes=(self.workspace.temporary.numel() if self.workspace else 0),
        )

    def _prepare_raw_masks(self, raw_targets):
        """Upload the raw masks, predict derived values without live writes, and split derived copies by device.

        Runs on the apply stream; returns ``prepare_raw_xor``'s physical masks.
        """
        self.raw_device = (
            self.raw_pinned.to(self.device, non_blocking=True)
            if any(binding.storage[0].device.type == "cuda" for binding, _, _ in raw_targets)
            else None
        )
        if self.raw_device is None:
            self.timings["raw_h2d_bytes"] = 0
        raw_updates = prepare_raw_xor(
            [
                (
                    binding,
                    (self.raw_pinned if binding.storage[0].device.type == "cpu" else self.raw_device)[
                        position : position + size
                    ],
                )
                for binding, position, size in raw_targets
            ],
            self.backend.layout.derived_targets,
        )
        raw_values = {id(target): source for target, _, source in raw_updates if source is not None}
        self.derived.extend(prepare_updates(self.backend.layout.derived_updates, raw_values))
        gpu_derived = []
        for image in self.derived:
            destination = (
                self.cpu_derived
                if image.source.device.type == "cpu" or image.destination.device.type == "cpu"
                else gpu_derived
            )
            destination.append(image)
        self.derived = gpu_derived
        return raw_updates

    def _prepare_decode_plan(self, frame_table, frame_counts):
        """Upload the decoder's frame table and bind each decode batch's status check to the sticky error."""
        self.workspace = self.decode_plan = None
        if self.static_plans:
            self.decode_plan = self.decoder.prepare_batches(
                frame_table,
                frame_counts,
                self.backend.host_arena.tensor,
                self.io_stream if self.hardware else self.stream,
            )
            self.decode_plan.record_stream(self.stream)
            self.workspace = self.decode_plan.workspace
            from miles.utils.gpu_delta.apply import prepare_status_check

            with torch.cuda.stream(self.stream):
                self.status_checks = [prepare_status_check(decode, self.error) for decode in self.decode_plan.batches]

    def _prepare_raw_apply(self, updates):
        """Bind the small-mask XOR launch completely before the scheduler pause."""
        if not updates:
            return False
        from miles.utils.gpu_delta.apply import plan_apply

        key = tuple(
            (target.device, target.data_ptr(), tuple(target.shape), tuple(target.stride()), target.dtype)
            for target, _, _ in updates
        )
        cached = self.backend.raw_plan
        if cached is None or cached[0] != key:
            outputs, offset = [], 0
            for target, mask, _ in updates:
                size = mask.numel()
                binding = TensorBinding(
                    "small-mask", "U8", (size,), [[0, size]], (target,), destinations=(_byte_view(target),)
                )
                outputs.append((binding, offset, size))
                offset += size
            group = plan_apply(outputs)
            self.backend.raw_plan = key, group
        else:
            group = cached[1]
        if group.cpu_targets:
            self.timings["raw_h2d_bytes"] += sum(
                mask.numel() * mask.element_size() for _, mask, _ in updates if mask.device.type == "cpu"
            )
            masks = torch.cat(
                [
                    mask.to(self.device, non_blocking=True) if mask.device.type == "cpu" else mask
                    for _, mask, _ in updates
                ]
            )
        else:
            masks = torch.cat([mask for _, mask, _ in updates])
        reused = group.prepare_metadata(self.device.index)
        group.bind(masks)
        if group.cpu_targets:
            self.raw_pointer_host = torch.tensor(
                list(group.pointer_rows(masks.data_ptr(), allow_unbound_cpu=True)), dtype=torch.int64, pin_memory=True
            )
            pointers = self.raw_pointer_host.to(self.device, non_blocking=True)
            self.cpu_storages.update(group.cpu_storages)
            self.cpu_target_groups.append((group, self.raw_pointer_host.numpy()))
        else:
            pointers = torch.tensor(list(group.pointer_rows(masks.data_ptr())), dtype=torch.int64, device=self.device)
        metadata = torch.tensor(group.metadata_rows, dtype=torch.int64, device=self.device)
        self.raw_apply = group.bind_launch(pointers, metadata, self.error)
        self.raw_apply_inputs = masks, pointers, metadata
        return reused

    @profiling.annotate("gpu_delta.prepare.apply_bindings")
    def _prepare_apply_bindings(self):
        """Prepare fixed descriptors, target pointers and source-index patches."""
        self.apply_host_view = self.apply_host_metadata.numpy()
        indices = [[], []]
        offsets = [[], []]
        position = 0
        for index, (_, _, group) in enumerate(self.static_plans):
            count = 2 * len(group.sources)
            start, stop = position + count, position + count + group.metadata_size
            self.apply_host_view[position:start] = list(group.pointer_rows(0, allow_unbound_cpu=True))
            if group.cpu_targets:
                self.cpu_storages.update(group.cpu_storages)
                self.cpu_target_groups.append((group, self.apply_host_view[position:start]))
            self.apply_host_view[start:stop] = group.metadata_rows
            self.apply_bindings.append(
                group.bind_launch(self.apply_metadata[position:start], self.apply_metadata[start:stop], self.error)
            )
            slot = index % 2
            indices[slot].extend(position + i for i in group.source_indices)
            offsets[slot].extend(group.sources)
            position = stop
        if position != self.apply_host_metadata.numel():
            raise RuntimeError("GPU delta apply metadata slab capacity mismatch")
        self.apply_source_patches = [
            (np.asarray(i, dtype=np.int64), np.asarray(o, dtype=np.int64))
            for i, o in zip(indices, offsets, strict=True)
        ]

    @profiling.annotate("gpu_delta.pause.bind_apply_metadata")
    def _bind_apply_metadata(self):
        """Validate decoded slots, then patch only actual source bases."""
        for index, (_, _, group) in enumerate(self.static_plans):
            group.bind(self.decoded[index % 2])
        for slot, (indices, offsets) in enumerate(self.apply_source_patches):
            if indices.size:
                # Each group's constant-time bound check above proves no int64
                # overflow before this vectorized patch. Targets are untouched.
                self.apply_host_view[indices] = offsets + self.decoded[slot].data_ptr()
        return self.apply_bindings

    @profiling.annotate("gpu_delta.pause.cpu_mirrors")
    def _bind_cpu_mirrors(self):
        # Copy current physical bytes only after the reader fence. One complete
        # storage preserves aliases, affine offsets and untouched padding.
        for key, storage in self.cpu_storages.items():
            mirror = torch.empty(storage.numel(), dtype=torch.uint8, device=self.device)
            self.cpu_mirrors[key] = mirror
            mirror.copy_(storage, non_blocking=True)
        for group, pointers in self.cpu_target_groups:
            for index, address in group.bind_cpu_targets(self.cpu_mirrors):
                pointers[index] = address
        if self.raw_pointer_host is not None:
            self.raw_apply_inputs[1].copy_(self.raw_pointer_host, non_blocking=True)
        self.timings["cpu_weight_h2d_bytes"] = sum(storage.numel() for storage in self.cpu_storages.values())

    @profiling.annotate("gpu_delta.pause.cpu_commit")
    def _commit_cpu(self):
        # The GPU verdict is known before any CPU weight changes. A failure in
        # copying back still poisons the transaction and requires replacement.
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.no_grad():
            for key, mirror in self.cpu_mirrors.items():
                self.cpu_storages[key].copy_(mirror)
            self.timings["cpu_weight_d2h_bytes"] = sum(mirror.numel() for mirror in self.cpu_mirrors.values())
            pending_gpu = False
            for image in self.cpu_derived:
                pending_gpu |= image.destination.device.type == "cuda"
                image.destination.copy_(image.source, non_blocking=image.destination.device.type == "cuda")
            if pending_gpu:
                done = torch.cuda.Event()
                done.record(self.stream)
                done.synchronize()

    @profiling.annotate("gpu_delta.pause.allocate_bind")
    def _allocate_paused(self):
        """Allocate reusable decoded outputs, then bind their pointers."""
        started = time.perf_counter()
        self.decoded = (
            [self.decoder.empty(self.max_decoded, torch.uint8) for _ in range(2)] if self.static_plans else []
        )
        decoders = self.decode_plan.bind_outputs(self.decoded) if self.decode_plan else []
        apply_calls = self._bind_apply_metadata()
        self.apply_metadata.copy_(self.apply_host_metadata, non_blocking=True)
        for index, decode in enumerate(decoders):
            scratch = self.decoded[index % 2]
            self.batches.append(
                _PreparedBatch(
                    decode,
                    apply_calls[index],
                    [scratch[offset : offset + size] for offset, size in self.gaps[index]],
                    self.status_checks[index],
                )
            )
        self.timings.update(
            paused_setup_host_s=time.perf_counter() - started,
            decoder_metadata_uploads=(2 if self.hardware else 3) * int(bool(decoders)),
            compressed_input_buffers=0 if self.hardware or not decoders else 2,
            compressed_input_capacity_bytes=(self.decode_plan.input_capacity_bytes if self.decode_plan else 0),
            compressed_input_h2d_bytes=(self.decode_plan.input_bytes if self.decode_plan else 0),
            decoded_buffers=len(self.decoded),
            decoded_scratch_bytes=len(self.decoded) * self.max_decoded,
        )
        self.h2d_bytes = sum(
            self.timings[key]
            for key in (
                "raw_h2d_bytes",
                "decoder_metadata_h2d_bytes",
                "apply_metadata_h2d_bytes",
                "compressed_input_h2d_bytes",
            )
        )
        self.h2d_bytes += self.timings.get("cpu_weight_h2d_bytes", 0)

    @profiling.annotate("gpu_delta.pause.apply")
    def apply(self):
        apply_started = time.perf_counter()
        self.backend.layout.check_identity()
        self.timings["host_apply_identity_s"] = time.perf_counter() - apply_started
        with torch.cuda.device(self.device), torch.no_grad():
            self.stream.wait_stream(torch.cuda.default_stream(self.device))
            with torch.cuda.stream(self.stream):
                if self.cpu_storages:
                    self._bind_cpu_mirrors()
                # Queue small XOR masks before allocating and binding matrix buffers.
                self._apply_raw()
                self._allocate_paused()
                self._apply_matrices()
                done = torch.cuda.Event()
                done.record(self.stream)
        completion_started = time.perf_counter()
        done.synchronize()
        self.timings["host_apply_completion_wait_s"] = time.perf_counter() - completion_started
        if self.error.item() != 0:
            raise RuntimeError("direct GPU delta decompression failed; session is poisoned")
        if self.cpu_storages or self.cpu_derived:
            self._commit_cpu()
        # Release output/workspace leases before generation resumes. DE allocation
        # free/fence overhead remains included in paused host wall time.
        self._release_gpu()
        self._gpu_complete = True
        self.timings["paused_apply_host_wall_s"] = time.perf_counter() - apply_started
        return {
            "applied": True,
            "target_version": self.target_version,
            "tensors": self.matrix_tensor_count + self.raw_tensor_count,
            "timings": self.timings,
            "h2d_bytes": self.h2d_bytes,
        }

    @profiling.annotate("gpu_delta.pause.raw_apply")
    def _apply_raw(self):
        if self.raw_apply is not None:
            self.raw_apply()

    @profiling.annotate("gpu_delta.pause.paused_gpu_pipeline")
    def _apply_matrices(self):
        if self.batches:
            self.io_stream.wait_stream(self.stream)
            if self.hardware:
                self._decode_batch(self.batches[0], 0)
            else:
                self._copy_batch(self.batches[0], 0)
        matrices_started = time.perf_counter()
        if self.batches:
            if self.hardware:
                self._apply_hardware_batches()
            else:
                self._apply_cuda_batches()
        self.timings["host_matrix_enqueue_s"] = time.perf_counter() - matrices_started
        self._refresh_images()

    @profiling.annotate("gpu_delta.pause.derived_refresh")
    def _refresh_images(self):
        # After the last matrix batch: copies, prepared calibration, then computed images.
        # The sticky error gate keeps every old value when any decode or apply failed.
        images = (*self.derived, *self.computed)
        if images:
            decode_succeeded = self.error.view(()) == 0
            for image in images:
                image.refresh(decode_succeeded)

    def _apply_hardware_batches(self):
        # DE writes one decoded slot while the SMs consume the other.
        for index, batch in enumerate(self.batches):
            slot = index % 2
            self.stream.wait_event(self.decoded_ready[slot])
            self._apply_batch(batch)
            self.decoded_free[slot].record(self.stream)
            # Queue apply before nvCOMP, which may wait on its calling stream.
            if index + 1 < len(self.batches):
                self._decode_batch(self.batches[index + 1], index + 1)

    @profiling.annotate("gpu_delta.pause.h2d")
    def _copy_batch(self, batch, index):
        with torch.cuda.stream(self.io_stream):
            slot = index % 2
            if index >= 2:
                self.io_stream.wait_event(self.decoded_free[slot])
            batch.decoder.copy_input(self.io_stream)
            self.decoded_ready[slot].record(self.io_stream)

    def _apply_cuda_batches(self):
        # CUDA decode and apply share one compute stream. Only compressed H2D
        # overlaps compute; a slot becomes reusable as soon as decoding ends.
        for index, batch in enumerate(self.batches):
            slot = index % 2
            self.stream.wait_event(self.decoded_ready[slot])
            if index + 1 < len(self.batches):
                self._copy_batch(self.batches[index + 1], index + 1)
            self._enqueue_decode(batch)
            self.decoded_free[slot].record(self.stream)
            self._apply_batch(batch)

    def _decode_batch(self, batch, index):
        stream = batch.decoder.stream
        with torch.cuda.stream(stream):
            slot = index % 2
            if index >= 2:
                stream.wait_event(self.decoded_free[slot])
            self._enqueue_decode(batch)
            self.decoded_ready[slot].record(stream)

    @profiling.annotate("gpu_delta.pause.decode")
    def _enqueue_decode(self, batch):
        # Zero unwritten gaps, then decode, both on the caller's current stream.
        if batch.zero_ranges:
            torch._foreach_zero_(batch.zero_ranges)
        batch.decoder.enqueue()

    @profiling.annotate("gpu_delta.pause.layout_apply")
    def _apply_batch(self, batch):
        # Sticky status and every consumer of it are ordered on the apply
        # stream. DE never races that gate while decoding the next batch.
        batch.check_status()
        batch.apply()

    @profiling.annotate("gpu_delta.pause.release_gpu")
    def _release_gpu(self):
        self.batches.clear()
        self.status_checks.clear()
        self.raw_apply = self.raw_apply_inputs = None
        self.raw_pointer_host = None
        self.cpu_mirrors.clear()
        self.cpu_storages.clear()
        self.cpu_target_groups.clear()
        self.cpu_derived = []
        self.derived = []
        self.computed = []
        self.apply_bindings.clear()
        self.apply_source_patches.clear()
        self.apply_host_view = None
        self.apply_metadata = self.apply_host_metadata = None
        self.decoded = self.raw_device = self.error = self.workspace = self.decode_plan = None

    @profiling.annotate("gpu_delta.release_host")
    def close(self, reuse_host=False):
        # Error/cancel paths must attempt every stream before releasing any
        # storage. Resume can reuse the successful apply fence and scratch
        # release; failure cleanup always drains, even after a completed apply.
        if not reuse_host or not self._gpu_complete:
            failure = None
            for stream in (self.io_stream, self.stream):
                if stream is not None:
                    try:
                        stream.synchronize()
                    except BaseException as error:
                        if failure is None:
                            failure = error
            if failure is not None:
                raise failure
            self._release_gpu()
        if self.host_snapshot is not None:
            self.host_snapshot.close(reuse_host)
            self.host_snapshot = None
        self.raw_pinned = None
