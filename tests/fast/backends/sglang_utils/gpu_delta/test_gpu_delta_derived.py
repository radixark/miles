"""Scalar refresh contracts: prepare is private, captured outputs stay in place."""

import ast
import importlib.util
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from miles.backends.sglang_utils.gpu_delta import derived
from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer as calibration
from miles.backends.sglang_utils.gpu_delta.kernel_backends.flashinfer import FlashInferLayout

_root = Path(importlib.util.find_spec("sglang").origin).parent / "srt"


def _production_helpers(relative_path, names):
    # Execute the actual pure helpers without loading CUDA inference backends.
    # This follows the checkpoint-admission tests in this same test directory.
    path = _root / relative_path
    tree = ast.parse(path.read_text())
    scope = {
        "torch": torch,
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "_cutedsl_logged_scalarize": set(),
        "log_info_on_rank0": lambda *args: None,
        "print_warning_once": lambda *args: None,
    }
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(  # noqa: S102 - exact checked-in source, no CUDA-only module imports.
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


def _layer(backend):
    cute = backend == "cutedsl"
    mega = backend == "megamoe"
    vector_input = cute
    return SimpleNamespace(
        num_local_experts=2,
        num_experts=4,
        moe_ep_rank=1,
        quant_config=SimpleNamespace(use_per_token_activation=False),
        moe_runner_config=SimpleNamespace(
            is_gated=True,
            activation="silu",
            gemm1_clamp_limit=None,
            swiglu_limit=None,
            gemm1_alpha=1.0,
        ),
        w13_weight_scale_2=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        w2_weight_scale_2=torch.tensor([6.0, 7.0]),
        w13_input_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        w2_input_scale=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        g1_alphas=torch.zeros(2),
        g1_alphas_up=torch.zeros(2),
        g2_alphas=torch.zeros(2),
        g1_scale_c=torch.zeros(2),
        w13_input_scale_quant=torch.ones(2) if vector_input else torch.tensor(1.0),
        w2_input_scale_quant=torch.ones(2) if cute or mega else torch.tensor(1.0),
        gemm1_clamp_limit=None,
        gemm1_beta=torch.zeros(2) if not (cute or mega) else None,
        _cutedsl_scales=(torch.zeros(2), torch.ones(1), torch.zeros(2)),
        _cutedsl_input_scale=torch.ones(1),
        _flashinfer_megamoe_input_norm_const=0.25,
        _flashinfer_megamoe_layer=SimpleNamespace(_megakernel_config=SimpleNamespace(input_norm_const=0.25)),
    )


def _update(layer, backend, quant_mode="w4a4"):
    return calibration.nvfp4_derived_update(layer, FlashInferLayout("nvfp4", backend, True, quant_mode))


def _prepare(update, overrides):
    # prepare_updates validates a batch's predicates together; validate one update here.
    copies, checks = update._prepare(overrides)
    derived._validate_checks(checks)
    return copies


def _value(copies, destination):
    return next(copy.source for copy in copies if copy.destination is destination)


def _prepare_updates(updates, raw):
    targets = [target for target, _ in raw]
    plans = derived.plan_updates(updates, targets)
    return derived.prepare_updates(plans, {id(target): value for target, value in raw})


class TestDerivedNvfp4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal = _production_helpers(
            "layers/quantization/modelopt_quant.py",
            {"_compute_gemm1_alphas", "_input_scale_to_local_experts"},
        )
        cls.cute = _production_helpers(
            "layers/moe/moe_runner/flashinfer_cutedsl.py",
            {"cutedsl_quant_scale_to_scalar", "resolve_cutedsl_standard_scales"},
        )
        cls.trt = _production_helpers("layers/moe/moe_runner/flashinfer_trtllm.py", {"_compute_g1_scale_c"})

    def setUp(self):
        self.helpers = patch.object(
            calibration,
            "_normal_scale_helpers",
            return_value=(
                self.normal._compute_gemm1_alphas,
                self.normal._input_scale_to_local_experts,
            ),
        )
        self.helpers.start()
        self.addCleanup(self.helpers.stop)

        self.cute_patch = patch.object(
            calibration,
            "_cutedsl_scale_helper",
            return_value=self.cute.resolve_cutedsl_standard_scales,
        )
        self.cute_patch.start()
        self.addCleanup(self.cute_patch.stop)
        self.trt_patch = patch.object(
            calibration,
            "_trtllm_scale_helper",
            return_value=self.trt._compute_g1_scale_c,
        )
        self.trt_patch.start()
        self.addCleanup(self.trt_patch.stop)

    def test_trtllm_clamping_rejects_block_scale_requantization(self):
        layer = _layer("trtllm")
        layer.moe_runner_config.gemm1_clamp_limit = 24.0
        layer.gemm1_clamp_limit = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "block-scale requantization"):
            _update(layer, "trtllm")
        layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        update = _update(layer, "trtllm")
        copies = _prepare(update, {})
        torch.testing.assert_close(_value(copies, layer.gemm1_clamp_limit), torch.tensor([3.0, 1.5]))
        with self.assertRaisesRegex(ValueError, "block-scale requantization"):
            _prepare(update, {id(layer.w13_weight_scale_2): torch.tensor([[2.0, 3.0], [4.0, 4.0]])})
        torch.testing.assert_close(layer.g1_alphas, torch.zeros(2))

    def test_cutedsl_local_experts_and_shared_fc2_quantization(self):
        layer = _layer("cutedsl")
        update = _update(layer, "cutedsl")
        copies = _prepare(update, {id(layer.w2_input_scale): torch.tensor([100.0, 200.0, 10.0, 20.0])})
        torch.testing.assert_close(_value(copies, layer.g2_alphas), torch.tensor([60.0, 140.0]))
        # Standard CuTe scalarizes only this rank's selected experts for FC2.
        torch.testing.assert_close(_value(copies, layer._cutedsl_scales[1]), torch.tensor([0.05]))
        torch.testing.assert_close(_value(copies, layer._cutedsl_scales[2]), torch.tensor([120.0, 140.0]))
        torch.testing.assert_close(_value(copies, layer._cutedsl_input_scale), torch.tensor([0.25]))

    def test_w4a16_retains_copy_only_semantics(self):
        layer = _layer("cutedsl")
        update = _update(layer, "cutedsl", "w4a16")
        self.assertEqual(len(update.sources), 2)
        live = [(tensor, tensor.clone(), tensor.data_ptr()) for tensor in (*update.sources, *update.destinations)]
        weight = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
        weight2 = torch.tensor([23.0, 29.0])
        with patch.object(
            calibration,
            "_normal_scale_helpers",
            side_effect=AssertionError("W4A16 needs no arithmetic helper"),
        ):
            copies = _prepare(update, {id(layer.w13_weight_scale_2): weight, id(layer.w2_weight_scale_2): weight2})
        torch.testing.assert_close(_value(copies, layer.g1_alphas), torch.tensor([11.0, 17.0]))
        torch.testing.assert_close(_value(copies, layer.g1_alphas_up), torch.tensor([13.0, 19.0]))
        torch.testing.assert_close(_value(copies, layer._cutedsl_scales[0]), torch.tensor([11.0, 17.0]))
        torch.testing.assert_close(_value(copies, layer.g2_alphas), torch.tensor([23.0, 29.0]))
        torch.testing.assert_close(_value(copies, layer._cutedsl_scales[2]), torch.tensor([23.0, 29.0]))
        self.assertTrue(all(copy.destination is not layer._cutedsl_input_scale for copy in copies))
        # Prepared values are snapshots: a later change to the predicted inputs never reaches them.
        predicted = weight.clone()
        copies = _prepare(update, {id(layer.w13_weight_scale_2): predicted})
        predicted.fill_(0)
        torch.testing.assert_close(_value(copies, layer.g1_alphas), torch.tensor([11.0, 17.0]))
        # MegaMoE W4A16 has the same weight bytes, but no captured input norm.
        layer = _layer("megamoe")
        layer._flashinfer_megamoe_input_norm_const = None
        layer._flashinfer_megamoe_layer._megakernel_config = SimpleNamespace()
        update = _update(layer, "megamoe", "w4a16")
        self.assertEqual((len(update.sources), len(update.destinations)), (2, 3))
        live += [(tensor, tensor.clone(), tensor.data_ptr()) for tensor in (*update.sources, *update.destinations)]
        tied = torch.tensor([[11.0, 11.0], [17.0, 17.0]])
        with patch.object(calibration, "_normal_scale_helpers", side_effect=AssertionError):
            copies = _prepare(update, {id(layer.w13_weight_scale_2): tied, id(layer.w2_weight_scale_2): weight2})
        torch.testing.assert_close(_value(copies, layer.g1_alphas), torch.tensor([11.0, 17.0]))
        torch.testing.assert_close(_value(copies, layer.g2_alphas), torch.tensor([23.0, 29.0]))
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            _prepare(update, {id(layer.w13_weight_scale_2): weight})
        # Preparation only returns copies: no live source or destination is written or replaced.
        for tensor, old, pointer in live:
            torch.testing.assert_close(tensor, old)
            self.assertEqual(tensor.data_ptr(), pointer)
        # TRTLLM serves NVFP4 only as W4A4. CuTe DSL refreshes the 3-tuple of scales its warmup records; this guard
        # rejects a non-tuple or wrong-length value (identify_layout already rejects an unwarmed wrapper).
        with self.assertRaisesRegex(ValueError, "^unsupported NVFP4 derived quant mode for trtllm: 'w4a16'$"):
            _update(_layer("trtllm"), "trtllm", "w4a16")
        for scales in (None, ()):
            layer = _layer("cutedsl")
            layer._cutedsl_scales = scales
            with (
                self.subTest(scales=scales),
                self.assertRaisesRegex(ValueError, "^CuTe DSL scale metadata must be warmed before admission$"),
            ):
                _update(layer, "cutedsl", "w4a16")

    def test_megamoe_rejects_captured_scalar_change_and_gate_up_mismatch(self):
        layer = _layer("megamoe")
        layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        update = _update(layer, "megamoe")
        copies = _prepare(update, {id(layer.w2_input_scale): torch.tensor([1.0, 1.0, 3.0, 5.0])})
        torch.testing.assert_close(_value(copies, layer.g2_alphas), torch.tensor([18.0, 35.0]))
        with self.assertRaisesRegex(ValueError, "input normalization changed"):
            _prepare(update, {id(layer.w13_input_scale): torch.tensor([1.0, 2.0, 3.0, 5.0])})
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            _prepare(update, {id(layer.w13_weight_scale_2): torch.tensor([[2.0, 3.0], [4.0, 4.0]])})
        torch.testing.assert_close(layer.g1_alphas, torch.zeros(2))

        layer = _layer("megamoe")
        layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        update = _update(layer, "megamoe")
        near = layer.w13_weight_scale_2.clone()
        near[:, 1] += 1e-6
        _prepare(update, {id(layer.w13_weight_scale_2): near})
        near[0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            _prepare(update, {id(layer.w13_weight_scale_2): near})
        layer._flashinfer_megamoe_input_norm_const = 0.1
        layer._flashinfer_megamoe_layer._megakernel_config.input_norm_const = 0.1
        update = _update(layer, "megamoe")
        with self.assertRaisesRegex(ValueError, "input normalization changed"):
            _prepare(update, {id(layer.w13_input_scale): torch.full((4,), 10.0)})

    def test_megamoe_batch_validates_once_before_returning_any_copies(self):
        layers = [_layer("megamoe") for _ in range(2)]
        for layer in layers:
            layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        updates = [_update(layer, "megamoe") for layer in layers]
        targets = [layer.w2_weight_scale_2[:1] for layer in layers]
        plans = derived.plan_updates(updates, targets)
        values = [torch.tensor([11.0]), torch.tensor([13.0])]
        raw = {id(target): value for target, value in zip(targets, values, strict=True)}
        cpu = torch.Tensor.cpu
        transfers = []

        def transfer(tensor, *args, **kwargs):
            transfers.append(tensor.numel())
            return cpu(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "cpu", transfer),
            patch.object(
                torch,
                "allclose",
                side_effect=AssertionError("must use device predicates"),
            ),
        ):
            copies = derived.prepare_updates(plans, raw)
        self.assertEqual(transfers, [4])
        torch.testing.assert_close(_value(copies, layers[1].g2_alphas), torch.tensor([78.0, 56.0]))
        for layer in layers:
            torch.testing.assert_close(layer.g2_alphas, torch.zeros(2))
            torch.testing.assert_close(layer.w2_weight_scale_2, torch.tensor([6.0, 7.0]))

        # Failure in a later layer must not expose or write the earlier copies.
        layers[1].w13_input_scale[-1] = 5.0
        transfers.clear()
        with (
            patch.object(torch.Tensor, "cpu", transfer),
            self.assertRaisesRegex(ValueError, "input normalization changed"),
        ):
            derived.prepare_updates(plans, raw)
        self.assertEqual(transfers, [4])
        for layer in layers:
            torch.testing.assert_close(layer.g1_alphas, torch.zeros(2))

    def test_per_token_w4a4_reuses_its_captured_neutral_constants(self):
        # Per-token CuTe DSL W4A4 copies the gate/up global scales and keeps its captured activation constants.
        layer = _layer("cutedsl")
        layer.quant_config.use_per_token_activation = True
        # Values no recomputation or fresh neutral allocation produces: only the captured tensors match.
        layer.w13_input_scale_quant, layer._cutedsl_input_scale = torch.full((2,), 0.5), torch.tensor([0.25])
        weights = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
        copies = _prepare(_update(layer, "cutedsl"), {id(layer.w13_weight_scale_2): weights})
        torch.testing.assert_close(_value(copies, layer.g1_alphas_up), weights[:, 1])
        torch.testing.assert_close(_value(copies, layer.w13_input_scale_quant), torch.full((2,), 0.5))
        torch.testing.assert_close(_value(copies, layer._cutedsl_input_scale), torch.tensor([0.25]))

    def test_prepare_updates_overlays_gate_up_scalars(self):
        layer = _layer("trtllm")
        update = _update(layer, "trtllm")
        other = _layer("trtllm")
        unchanged = _update(other, "trtllm")
        sources_before = [source.clone() for source in update.sources]
        for complete in (False, True):
            with self.subTest(complete=complete):
                if complete:
                    targets = [
                        layer.w13_weight_scale_2[:, 0],
                        layer.w13_weight_scale_2[:, 1],
                    ]
                    values = [torch.tensor([11.0, 17.0]), torch.tensor([13.0, 19.0])]
                    expected_gate, expected_up = [44.0, 68.0], [52.0, 76.0]
                else:
                    targets = [
                        layer.w13_weight_scale_2[0, 1:2],
                        layer.w13_weight_scale_2[1, 0:1],
                    ]
                    values = [torch.tensor([11.0]), torch.tensor([17.0])]
                    expected_gate, expected_up = [8.0, 68.0], [44.0, 20.0]
                copies = _prepare_updates((update, unchanged), list(zip(targets, values, strict=True)))
                torch.testing.assert_close(_value(copies, layer.g1_alphas), torch.tensor(expected_gate))
                torch.testing.assert_close(_value(copies, layer.g1_alphas_up), torch.tensor(expected_up))
                self.assertEqual(len(copies), len(update.destinations))
        self.assertEqual(_prepare_updates((update,), []), ())
        for source, before in zip(update.sources, sources_before, strict=False):
            torch.testing.assert_close(source, before)

    def test_derived_destination_and_patch_geometry_fail_before_writes(self):
        layer = _layer("trtllm")
        update = _update(layer, "trtllm")
        layer.g1_alphas.resize_(3)
        with self.assertRaisesRegex(ValueError, "destination geometry changed"):
            _prepare(update, {})
        torch.testing.assert_close(layer.g2_alphas, torch.zeros(2))


class TestPatchGeometry(unittest.TestCase):
    def test_plan_updates_raises_patch_errors(self):
        allocation = torch.arange(64, dtype=torch.float32)
        source = allocation[8:40].view(4, 8)
        update = derived.DerivedUpdate((source,), (torch.zeros(4, 8),), lambda values: values)
        cases = {
            "outside its source": [allocation[38:42]],
            "updates overlap": [source[:, :3], source[1:, 2:4]],
            "dtype mismatch": [allocation.view(torch.int32)[8:10]],
        }
        for message, targets in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                derived.plan_updates((update,), targets)
        # A stride-0 view writes one element twice, an overlap within one patch.
        with self.assertRaisesRegex(ValueError, "updates overlap"):
            derived.plan_updates((update,), [source[0, :1].expand(2)])
        transposed = derived.DerivedUpdate((source.t(),), (torch.zeros(8, 4),), lambda values: values)
        with self.assertRaisesRegex(ValueError, "source must be contiguous"):
            derived.plan_updates((transposed,), [source[:1]])

    def test_plan_updates_admits_disjoint_strided_patches_of_a_large_scale(self):
        # Two interleaved column patches and a row block of one source; a target of no source is ignored.
        scale = torch.zeros(1024, 40)
        targets = [scale[:512, 0::2], scale[:512, 1::2], scale[512:]]
        update = derived.DerivedUpdate((scale,), (torch.zeros(1024, 40),), lambda values: values)
        ((planned, sources),) = derived.plan_updates((update,), targets + [torch.zeros(3)])
        self.assertIs(planned, update)
        ((original, patches),) = sources
        self.assertIs(original, scale)
        self.assertEqual([id(patch) for patch in patches], [id(target) for target in targets])


if __name__ == "__main__":
    unittest.main()
