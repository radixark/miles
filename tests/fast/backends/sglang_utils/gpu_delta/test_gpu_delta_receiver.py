"""Receiver preparation, plan reuse and decode/apply lifecycle contracts."""

import copy
import sys
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from miles.backends.sglang_utils.gpu_delta import bindings as byte_layout
from miles.backends.sglang_utils.gpu_delta import receiver
from miles.utils.gpu_delta.host_worker import TENSOR_DTYPE


def publication_view(catalog):
    return SimpleNamespace(
        header={"plan_digest": catalog["plan_digest"]},
        read_catalog=lambda: catalog,
    )


class TestCanonicalPlanCache(unittest.TestCase):
    def publication(self):
        entries = [
            {
                "name": name,
                "dtype": "U8",
                "shape": [2, 4],
                "encoding": "xor_bytes",
                "byte_order": "little",
                "nbytes": 8,
            }
            for name in ("local", "foreign")
        ]
        backend = SimpleNamespace(
            _canonical_plan_digest=None,
            binding_table=None,
            batch_plan=None,
            layers_per_batch=1,
            layout=SimpleNamespace(
                finish=Mock(),
                inventory={
                    name: {"dtype": "U8", "shape": [2, 4], "encoding": "xor_bytes"} for name in ("local", "foreign")
                },
                excluded={"foreign": "expert projection owned by another EP/TP rank"},
                bindings=[
                    SimpleNamespace(
                        name="local",
                        slices=[[0, 2], [0, 2]],
                        is_raw=False,
                        raw_ue8m0=False,
                        layer=None,
                        storage_pointers=(),
                    )
                ],
                derived=[],
            ),
        )
        definitions = [
            {key: entry[key] for key in ("name", "dtype", "shape", "encoding")}
            for entry in sorted(entries, key=lambda entry: entry["name"])
        ]
        return backend, {
            "tensors": sorted(entries, key=lambda entry: entry["name"]),
            "plan_digest": receiver.plan_digest(definitions),
        }

    def test_warm_plan_reuses_admitted_digest_with_fresh_payloads(self):
        # One canonical catalog admits every rank-local slice cold, without per-rank views.
        for slices in ([[0, 2], [2, 4]], [[0, 2], [0, 4]], [[0, 2], [0, 2]]):
            backend, publication = self.publication()
            backend.layout.bindings[0].slices = slices
            admitted, reused = receiver._qualify_canonical_plan(backend, publication_view(publication))
            self.assertFalse(reused)
            self.assertEqual(admitted.bindings[0].slices, slices)
            np.testing.assert_array_equal(admitted.ordinals, [1])
            np.testing.assert_array_equal(admitted.nbytes, [8])
            self.assertTrue(all("views" not in entry for entry in publication["tensors"]))
        updated = publication_view(publication)
        with (
            patch.object(receiver, "plan_digest", side_effect=AssertionError("cold only")),
            patch.object(updated, "read_catalog", side_effect=AssertionError("cold only")),
        ):
            entries, reused = receiver._qualify_canonical_plan(backend, updated)
        self.assertTrue(reused)
        self.assertIs(entries, admitted)
        self.assertEqual(backend._canonical_plan_digest, publication["plan_digest"])
        with self.assertRaisesRegex(ValueError, "plan changed"):
            receiver._qualify_canonical_plan(backend, publication_view(publication | {"plan_digest": "changed"}))

    def test_compact_scale_format_matches_the_checkpoint_prior(self):
        backend, publication = self.publication()
        definitions = []
        for entry in publication["tensors"]:
            entry["encoding"] = "xor_raw_bytes"
            canonical = backend.layout.inventory[entry["name"]]
            canonical.update(
                encoding="xor_raw_bytes",
                format="fp8-block128-ue8m0",
                checkpoint_dtype="F32",
            )
            definition = {key: entry[key] for key in ("name", "dtype", "shape", "encoding")}
            entry.update(format="fp8-block128-ue8m0", checkpoint_dtype="F32")
            definition.update(format="fp8-block128-ue8m0", checkpoint_dtype="F32")
            definitions.append(definition)
        publication["plan_digest"] = receiver.plan_digest(definitions)
        receiver._qualify_canonical_plan(backend, publication_view(publication))
        backend._canonical_plan_digest = None
        backend.layout.inventory["local"]["checkpoint_dtype"] = "U8"
        with self.assertRaisesRegex(ValueError, "canonical tensor metadata mismatch"):
            receiver._qualify_canonical_plan(backend, publication_view(publication))

    def test_failed_cold_qualification_does_not_admit_a_cache(self):
        for message, mutate in (
            ("canonical tensor plan", lambda _, catalog: catalog.update(plan_digest="invalid")),
            ("missing local tensors", lambda _, catalog: catalog["tensors"].pop(1)),
            ("unadmitted tensor", lambda backend, _: backend.layout.excluded.update(local="static bundled draft")),
        ):
            with self.subTest(message):
                backend, publication = self.publication()
                invalid = copy.deepcopy(publication)
                mutate(backend, invalid)
                with self.assertRaisesRegex(ValueError, message):
                    receiver._qualify_canonical_plan(backend, publication_view(invalid))
                self.assertIsNone(backend._canonical_plan_digest)
                backend.layout.excluded.pop("local", None)
                self.assertFalse(receiver._qualify_canonical_plan(backend, publication_view(publication))[1])


class TestReceiverPreparation(unittest.TestCase):
    def test_cuda_pipeline_prefetches_and_fences_two_input_slots(self):
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                events.append((self.name, "wait", event.name))

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self, stream):
                events.append((stream.name, "record", self.name))

        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared.stream, prepared.io_stream = Stream("compute"), Stream("h2d")
        prepared.decoded_ready = [Event(f"ready{i}") for i in range(2)]
        prepared.decoded_free = [Event(f"free{i}") for i in range(2)]
        prepared.batches = [
            SimpleNamespace(
                decoder=SimpleNamespace(
                    copy_input=lambda stream, i=i: events.append(("copy", i)),
                    enqueue=lambda i=i: events.append(("decode", i)),
                ),
                zero_ranges=[],
                check_status=lambda i=i: events.append(("status", i)),
                apply=lambda i=i: events.append(("apply", i)),
            )
            for i in range(4)
        ]
        with patch.object(torch.cuda, "stream", return_value=nullcontext()):
            prepared._copy_batch(prepared.batches[0], 0)
            prepared._apply_cuda_batches()
        for i in range(4):
            self.assertLess(events.index(("copy", i)), events.index(("decode", i)))
            self.assertLess(events.index(("decode", i)), events.index(("status", i)))
            self.assertLess(events.index(("status", i)), events.index(("apply", i)))
            if i < 3:
                self.assertLess(events.index(("copy", i + 1)), events.index(("decode", i)))
            if i >= 2:
                wait = events.index(("h2d", "wait", f"free{i % 2}"))
                self.assertLess(events.index(("compute", "record", f"free{i % 2}")), wait)
                self.assertLess(wait, events.index(("copy", i)))

    def test_decode_table_preserves_sparse_rows_and_natural_zero_ranges(self):
        plans = [
            (np.array([[0, 1], [0, 64]], dtype="<i8"), 80, None),
            (np.empty((2, 0), dtype="<i8"), 0, None),
            (np.array([[2], [0]], dtype="<i8"), 32, None),
            (np.array([[3], [0]], dtype="<i8"), 24, None),
        ]
        tensors = np.zeros(4, dtype=TENSOR_DTYPE)
        tensors["frame_start"] = [0, 2, 2, 3]
        tensors["frame_count"] = [2, 0, 1, 0]
        tensors["gap_start"] = [0, 3, 4, 6]
        tensors["gap_count"] = [3, 1, 2, 1]
        publication = SimpleNamespace(
            tensors=tensors,
            frames=np.array([[0, 16, 8], [3, 5, 7], [8, 8, 8], [8, 24, 16]], dtype="<i8"),
            gaps=np.array(
                [[0, 8], [16, 8], [32, 16], [0, 16], [0, 16], [24, 8], [0, 24]],
                dtype="<i8",
            ),
        )
        # DE offsets differ from the shared prepared image; both must produce
        # the same decoded geometry without changing the shared metadata.
        original_frames, original_gaps = (
            publication.frames.copy(),
            publication.gaps.copy(),
        )
        for offsets in (np.array([64, 192, 128, 256]), np.array([0, 64, 80, 112])):
            table, counts, gaps = receiver._plan_decode(plans, publication, offsets)
            self.assertEqual(table.dtype, np.int64)
            self.assertTrue(table.flags.c_contiguous)
            np.testing.assert_array_equal(
                table,
                [
                    [offsets[0], offsets[0] + 16, offsets[2] + 8],
                    [3, 5, 7],
                    [8, 8, 8],
                    [8, 24, 16],
                ],
            )
            self.assertEqual(counts, [2, 0, 1, 0])
            for actual, expected in zip(
                gaps,
                (
                    [[0, 8], [16, 8], [32, 16], [64, 16]],
                    [],
                    [[0, 16], [24, 8]],
                    [[0, 24]],
                ),
                strict=False,
            ):
                np.testing.assert_array_equal(actual, np.asarray(expected, dtype="<i8").reshape(-1, 2))
        np.testing.assert_array_equal(publication.frames, original_frames)
        np.testing.assert_array_equal(publication.gaps, original_gaps)
        empty, counts, gaps = receiver._plan_decode(plans[-1:], publication, offsets)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual(counts, [0])
        np.testing.assert_array_equal(gaps[0], [[0, 24]])
        empty, counts, gaps = receiver._plan_decode([], publication, offsets)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual((counts, gaps), ([], []))

    def test_prepare_uses_shared_publication_and_only_local_ordinals(self):
        backend, catalog = TestCanonicalPlanCache().publication()
        backend.device = torch.device("cpu")
        backend.decoders = {"lz4": SimpleNamespace(hardware=True)}
        refs = []

        class TrackedEntry(dict):
            pass

        def read_catalog():
            value = copy.deepcopy(catalog)
            foreign = TrackedEntry(value["tensors"][0])
            value["tensors"][0] = foreign
            refs.append(weakref.ref(foreign))
            return value

        publication = SimpleNamespace(
            header={
                "codec": "lz4-zstd",
                "backend": "hardware",
                "target_version": 7,
                "plan_digest": catalog["plan_digest"],
            },
            read_catalog=read_catalog,
        )

        def prepare_local(actual, tensor_ids, timings, hardware):
            self.assertIs(actual, publication)
            self.assertTrue(hardware)
            self.assertIsNone(refs[-1]())
            self.assertEqual(tuple(tensor_ids), (1,))
            raise RuntimeError("stop at local preparation boundary")

        backend.host_arena = SimpleNamespace(
            prepare_publication=lambda *args: publication, prepare_local=prepare_local
        )
        with self.assertRaisesRegex(RuntimeError, "local preparation boundary"):
            receiver.PreparedDelta(backend).prepare({})
        backend.layout.finish.assert_called_once_with()

    def test_layer_batch_cache_tracks_active_rows_and_grouping(self):
        names = [
            "model.layers.0.a",
            "model.layers.0.b",
            "model.layers.1.a",
            "model.layers.2.a",
            "model.layers.3.a",
            "standalone",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        bindings = [
            byte_layout._direct_binding(
                name,
                {"dtype": "U8", "shape": [2, 4]},
                torch.zeros(2, 4, dtype=torch.uint8),
            )
            for name in names
        ]
        entries = {b.name: (i, {"nbytes": 8}) for i, b in enumerate(bindings)}
        table = receiver._BindingTable(SimpleNamespace(bindings=bindings, derived=[]), entries)
        rows = np.arange(len(bindings))
        backend = SimpleNamespace(batch_plan=None, layers_per_batch=1)
        # This test checks grouping/cache ownership, not Triton launch plans.
        with patch.dict(
            sys.modules,
            {"miles.utils.gpu_delta.apply": SimpleNamespace(plan_apply=lambda outputs: None)},
        ):
            previous = receiver._plan_layers(backend, table, rows)
            self.assertIs(receiver._plan_layers(backend, table, rows), previous)
            reduced = receiver._plan_layers(backend, table, rows[1:])
            self.assertIsNot(reduced, previous)
            self.assertEqual(reduced[3][0][0, 0], 1)
            for count, expected in ((1, [[0], [1], [2], [3]]), (2, [[0, 1], [2, 3]])):
                backend = SimpleNamespace(batch_plan=None, layers_per_batch=count)
                grouped = receiver._plan_layers(backend, table, rows)
                self.assertEqual(
                    [list(dict.fromkeys(bindings[i].layer for i in plan[0][0])) for plan in grouped],
                    [[None], [None], [None], *expected],
                )
                self.assertIs(receiver._plan_layers(backend, table, rows), grouped)
            self.assertEqual(receiver._plan_layers(backend, table, rows[:0]), [])

    def test_binding_table_selects_active_rows_and_derived_sources(self):
        bindings = [
            byte_layout._direct_binding(
                name,
                {"dtype": "U8", "shape": [2, 4]},
                torch.zeros(2, 4, dtype=torch.uint8),
            )
            for name in ("matrix0", "matrix1")
        ]
        bindings += [
            byte_layout._direct_binding("raw", {"dtype": "F32", "shape": [1]}, torch.zeros(1)),
            byte_layout.TensorBinding(
                "scale",
                "U8",
                (1,),
                [[0, 1]],
                (torch.zeros(1),),
                encoding="xor_raw_bytes",
                format="fp8-block128-ue8m0",
            ),
        ]
        image = SimpleNamespace(source_pointer=bindings[1].storage_pointers[0])
        table = receiver._BindingTable(
            SimpleNamespace(bindings=bindings, derived=[image]),
            {
                b.name: (ordinal, {"nbytes": size})
                for b, ordinal, size in zip(bindings, [2, 0, 3, 1], [8, 8, 4, 1], strict=False)
            },
        )
        tensors = np.zeros(4, dtype=TENSOR_DTYPE)
        tensors["changed_bytes"] = [8, 1, 8, 4]
        tensors["frame_count"] = [0, 0, 1, 0]
        matrices, raw, derived = table.select(tensors)
        np.testing.assert_array_equal(matrices, [0])
        np.testing.assert_array_equal(raw, [2, 3])
        self.assertEqual(derived, [])
        tensors["frame_count"][0] = 1
        tensors["changed_bytes"][1] = 0
        matrices, raw, derived = table.select(tensors)
        np.testing.assert_array_equal(matrices, [0, 1])
        np.testing.assert_array_equal(raw, [2])
        self.assertEqual(derived, [image])

    def test_backend_reads_inventory_once_and_drains_all_streams_on_failure(self):
        fake_plan = SimpleNamespace(check_identity=lambda: None, finish=Mock(), bindings=[], excluded={})
        fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cpu"))]))
        model_runner = SimpleNamespace(model=fake_model, device="cuda", gpu_id=0)
        with (
            patch.dict(receiver.os.environ, GPU_DELTA_LAYERS_PER_BATCH="3"),
            patch.object(receiver, "GpuDeltaLayout", return_value=fake_plan) as layout_type,
            patch("miles.utils.gpu_delta.host.HostArena") as arena_type,
            patch(
                "miles.backends.sglang_utils.gpu_delta.checkpoint.read_canonical_checkpoint_inventory",
                return_value={"weight": {"shape": [1], "dtype": "U8"}},
            ) as read_inventory,
        ):
            backend = receiver.GpuDeltaBackend(model_runner, {"engine_id": "test-engine"})
            self.assertEqual(backend.device, torch.device("cuda", 0))
            self.assertEqual(backend.decoders, {})
            self.assertIsNone(backend.apply_stream)
            self.assertIsNone(backend.io_stream)
            self.assertEqual(backend.layers_per_batch, 3)
            with patch.dict(receiver.os.environ, GPU_DELTA_LAYERS_PER_BATCH="0"):
                self.assertEqual(backend.layers_per_batch, 3)
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    receiver.GpuDeltaBackend(model_runner, {"engine_id": "invalid"})
            read_inventory.assert_called_once()
            # Dtype admission rejects at construction, names the rank and creates no host resources.
            layout_type.side_effect = ValueError("mismatched dtypes")
            rejected = {"engine_id": "rejected", "tp_rank": 1, "dp_rank": 2}
            with self.assertRaisesRegex(ValueError, "^tp_rank=1 dp_rank=2: mismatched dtypes$"):
                receiver.GpuDeltaBackend(model_runner, rejected)
            arena_type.assert_called_once_with("test-engine", 0, {"engine_id": "test-engine"})
            backend.host_arena.prepare_publication.side_effect = ValueError("bad manifest")
            with patch.object(torch.cuda, "device", return_value=nullcontext()):
                with self.assertRaisesRegex(ValueError, "bad manifest"):
                    backend.prepare({})
            fake_plan.finish.assert_not_called()
            backend.host_arena.close.assert_called_once()

        drains = []
        backend = receiver.GpuDeltaBackend.__new__(receiver.GpuDeltaBackend)
        backend.host_arena = SimpleNamespace(close=lambda: drains.append("backend"))

        def decode_failure():
            drains.append("decode")
            raise RuntimeError("decode stream failed")

        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared._gpu_complete = True
        prepared.io_stream = SimpleNamespace(synchronize=decode_failure)
        prepared.stream = SimpleNamespace(synchronize=lambda: drains.append("compute"))
        prepared._release_gpu = lambda: drains.append("release-gpu")
        prepared.host_snapshot = SimpleNamespace(close=lambda reuse: drains.append(("snapshot", reuse)))
        prepared.raw_pinned = object()
        with self.assertRaisesRegex(RuntimeError, "decode stream failed"):
            backend.discard(prepared)
        self.assertEqual(drains, ["decode", "compute"])
        self.assertIsNotNone(prepared.raw_pinned)
        self.assertIsNotNone(prepared.host_snapshot)

        drains.clear()
        prepared.io_stream.synchronize = lambda: drains.append("decode")
        backend.discard(prepared)
        self.assertEqual(
            drains,
            ["decode", "compute", "release-gpu", ("snapshot", False), "backend"],
        )
        self.assertIsNone(prepared.raw_pinned)
        self.assertIsNone(prepared.host_snapshot)

        # Successful resume releases its host lease once. It skips the stream
        # drains only when apply has already fenced and released GPU scratch.
        for complete in (False, True):
            with self.subTest(gpu_complete=complete):
                drains.clear()
                prepared._gpu_complete = complete
                prepared.host_snapshot = SimpleNamespace(close=lambda reuse: drains.append(("snapshot", reuse)))
                prepared.raw_pinned = object()
                prepared.close(True)
                expected = [] if complete else ["decode", "compute", "release-gpu"]
                self.assertEqual(drains, [*expected, ("snapshot", True)])
                self.assertIsNone(prepared.raw_pinned)
                self.assertIsNone(prepared.host_snapshot)


if __name__ == "__main__":
    unittest.main()
