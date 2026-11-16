"""Canonical model mapping, native byte views and consumer admission contracts."""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

from tests.fast.fixtures.gpu_delta_fixtures import byte_view

from miles.backends.sglang_utils.gpu_delta import bindings as byte_layout
from miles.backends.sglang_utils.gpu_delta import layout
from miles.backends.sglang_utils.gpu_delta.bindings import ComputedImage, ConsumerSnapshot
from miles.backends.sglang_utils.gpu_delta.byte_views import DestinationSegment
from miles.backends.sglang_utils.gpu_delta.checkpoint import CanonicalCheckpoint
from miles.backends.sglang_utils.gpu_delta.derived import DerivedUpdate, prepare_updates
from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer as fi
from miles.backends.sglang_utils.gpu_delta.kernel_backends import identify_expert_layout
from miles.backends.sglang_utils.gpu_delta.kernel_backends.flashinfer import FlashInferLayout
from miles.backends.sglang_utils.gpu_delta.models import deepseek_v2
from miles.utils.gpu_delta import publication


def _linear(cls):
    # Skip SGLang's constructor: the mapper reads only the partition group and sharding axis.
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def _nvfp4_method():
    cls = type("ModelOptNvFp4FusedMoEMethod", (), {})
    method = cls()
    method.runner = None
    method._is_cutedsl_v2_standard = True
    method.quant_config = SimpleNamespace(is_checkpoint_nvfp4_serialized=True)
    return method


def _glm_checkpoint(inventory):
    # Admission reads only metadata and the policy of the checkpoint's own config.json.
    policy = publication.static_policy({"architectures": ["GlmMoeDsaForCausalLM"], "num_hidden_layers": 1})
    return CanonicalCheckpoint(inventory=inventory, policy=policy, locations={})


def _fixed_moe():
    return SimpleNamespace(
        enable_eplb=False,
        elastic_ep_backend=None,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        moe_a2a_backend="none",
        moe_runner_backend="flashinfer_cutedsl",
    )


def _bare_layout(bindings, mapping, consumers=()):
    # ``consumers``: the snapshots the mapping pinned while binding, as GpuDeltaLayout.__init__ records them.
    plan = layout.GpuDeltaLayout.__new__(layout.GpuDeltaLayout)
    plan.bindings, plan._mapping, plan._checkpoint = bindings, mapping, None
    plan._parameter_roots, plan._consumers, plan._tensor_checks = [], list(consumers), ()
    plan._mapping_consumers = tuple(consumers)
    return plan


def _apply_binding(binding, mask):
    for view in binding.apply_bindings():
        selected = view.selected_bytes(mask.reshape(-1))
        for destination in view.destinations:
            destination.bitwise_xor_(selected.reshape(destination.shape))


# MLA heads: 4 nope + 4 rope query rows, 4 nope + 6 value kv_b rows, KV LoRA rank 8.
_NOPE, _ROPE, _V, _LORA = 4, 4, 6, 8


def _mla_layer(attn_rank, attn_size, heads, dcp=1, storage="companion"):
    """One GLM MLA layer on an attention-TP rank, with the consumers SGLang's loader and DCP gather build.

    ``storage`` is the tree that gathered the DCP replicas: the companion keeps a separate local q_b
    and no V image; SGLang main makes the local q_b rows a view of the gathered image and gathers the
    transposed V image ``w_vc_qrep`` (``model_runner._prepare_replicated_q_proj``).
    """
    canonical = {
        "q_b_proj": torch.randn(heads * (_NOPE + _ROPE), _LORA).bfloat16(),
        "kv_b_proj": torch.randn(heads * (_NOPE + _V), _LORA).bfloat16(),
    }
    root = torch.nn.Module()
    root.config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    root.num_fused_shared_experts = 0
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    attn = root.model.layers[0].self_attn = torch.nn.Module()
    group = None if attn_size == 1 else SimpleNamespace(rank_in_group=attn_rank, world_size=attn_size)
    for name, value in canonical.items():
        module = _linear(ColumnParallelLinear)
        module.tp_group = group
        width = value.shape[0] // attn_size
        module.weight = torch.nn.Parameter(
            value[attn_rank * width : (attn_rank + 1) * width].clone(), requires_grad=False
        )
        module.weight.output_dim = 0
        setattr(attn, name, module)
    attn.qk_nope_head_dim, attn.v_head_dim, attn.qk_head_dim = _NOPE, _V, _NOPE + _ROPE
    key, value = attn.kv_b_proj.weight.unflatten(0, (-1, _NOPE + _V)).split([_NOPE, _V], dim=1)
    # DeepseekV2WeightLoaderMixin.post_load_weights for BF16 kv_b_proj.
    attn.w_kc = key.transpose(1, 2).contiguous().transpose(1, 2).detach()
    attn.w_vc = value.contiguous().transpose(1, 2).detach()
    attn.q_b_proj_qrep = attn.q_b_proj_qrep_weight = attn.w_kc_qrep = None
    if storage == "main":
        attn.w_vc_qrep = None
    if dcp > 1:
        local_heads = heads // attn_size
        first = attn_rank // dcp * dcp * local_heads
        span = slice(first, first + dcp * local_heads)
        gathered = canonical["kv_b_proj"].unflatten(0, (heads, _NOPE + _V))[span]
        attn.q_b_proj_qrep_weight = canonical["q_b_proj"].unflatten(0, (heads, -1))[span].flatten(0, 1).clone()
        attn.w_kc_qrep = gathered[:, :_NOPE].contiguous()
        if storage == "main":
            rows = attn.q_b_proj.weight.shape[0]
            rank = attn_rank % dcp
            attn.q_b_proj.weight.data = attn.q_b_proj_qrep_weight[rank * rows : (rank + 1) * rows]
            attn.w_vc_qrep = gathered[:, _NOPE:].transpose(1, 2).contiguous()
    inventory = {
        f"model.layers.0.self_attn.{name}.weight": {"dtype": "BF16", "shape": list(value.shape)}
        for name, value in canonical.items()
    }
    return root, attn, canonical, inventory


class TestFlashInferDeltaLayout(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            byte_layout,
            "_identify_moe_layout",
            side_effect=lambda layer: identify_expert_layout(layer, "flashinfer_cutedsl"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        consumer = patch.object(
            fi,
            "consumer_objects",
            side_effect=lambda layer, _: (
                layer._cutedsl_wrapper,
                layer._cutedsl_scales,
                *layer._cutedsl_scales,
            ),
        )
        consumer.start()
        self.addCleanup(consumer.stop)

    def test_tp_selection_reads_reused_decoder_scratch(self):
        for dtype, name in (
            (torch.uint8, "U8"),
            (torch.bfloat16, "BF16"),
            (torch.float32, "F32"),
        ):
            with self.subTest(dtype=name):
                size = 12 * 20 * torch.empty((), dtype=dtype).element_size()
                canonical = torch.randint(256, (size,), dtype=torch.uint8)
                target = canonical.view(dtype).reshape(12, 20)[2:10, 5:13].clone()
                binding = byte_layout._direct_binding(
                    "weight",
                    {"dtype": name, "shape": [12, 20], "encoding": "xor_bytes"},
                    target,
                    [[2, 10], [5, 13]],
                )
                decoded = torch.zeros_like(canonical)
                payload = binding.selected_bytes(decoded)
                self.assertEqual(
                    payload.untyped_storage().data_ptr(),
                    decoded.untyped_storage().data_ptr(),
                )
                self.assertFalse(payload.is_contiguous())
                # A prepared TP view observes later decoder writes without a
                # cached contiguous copy. Native tests cover the actual XOR gate.
                for _ in range(2):
                    decoded.random_(256)
                    expected = decoded.view(dtype).reshape(12, 20)[2:10, 5:13]
                    torch.testing.assert_close(byte_view(payload), byte_view(expected))

    def test_attention_and_shared_experts_follow_their_linear_partition_groups(self):
        for tp_rank, tp_size in ((0, 1), (3, 8)):
            for dcp in (False, True):
                with self.subTest(tp_rank=tp_rank, tp_size=tp_size, dcp=dcp):
                    self._check_shared_expert_partition(tp_rank, tp_size, dcp)
        # CP partners repeat attention shards while shared experts retain their
        # own TP group: TP4/CP2 attention rank1 lives on global ranks1 and3.
        # Pure CP also repeats a complete attention image on each TP2 rank.
        for tp_rank, tp_size, attn_rank, attn_size in (
            (1, 4, 1, 2),
            (3, 4, 1, 2),
            (0, 2, 0, 1),
            (1, 2, 0, 1),
        ):
            with self.subTest(tp_rank=tp_rank, tp_size=tp_size, attn_size=attn_size, prefill_cp=True):
                self._check_shared_expert_partition(tp_rank, tp_size, False, attn_rank, attn_size)

    def _check_shared_expert_partition(self, tp_rank, tp_size, dcp, attn_rank=3, attn_size=4):
        intermediate, hidden = 32, 16
        shard = intermediate // tp_size
        before = {
            key: torch.randn(shape).bfloat16()
            for key, shape in (
                ("gate", (intermediate, hidden)),
                ("up", (intermediate, hidden)),
                ("down", (hidden, intermediate)),
            )
        }
        after = {key: torch.randn_like(value) for key, value in before.items()}
        root = torch.nn.Module()
        root.config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        root.num_fused_shared_experts = 0
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.mlp = torch.nn.Module()
        shared = layer.mlp.shared_experts = torch.nn.Module()
        shared.gate_up_proj = torch.nn.Module()
        # The mapper follows the ordinary loader's input_dim for row parallel.
        shared.down_proj = _linear(RowParallelLinear)
        selection = slice(tp_rank * shard, (tp_rank + 1) * shard)
        shared.gate_up_proj.weight = torch.nn.Parameter(
            torch.cat([before["gate"][selection], before["up"][selection]]),
            requires_grad=False,
        )
        shared.down_proj.weight = torch.nn.Parameter(before["down"][:, selection].clone(), requires_grad=False)
        shared.gate_up_proj.weight.output_dim = 0
        shared.down_proj.weight.input_dim = 1
        # The current Linear API retains its partition owner, not tp_rank.
        group = None if tp_size == 1 else SimpleNamespace(rank_in_group=tp_rank, world_size=tp_size)
        for module in (shared.gate_up_proj, shared.down_proj):
            module.tp_group = group
        inventory = {
            f"model.layers.0.mlp.shared_experts.{key}_proj.weight": {
                "dtype": "BF16",
                "shape": list(value.shape),
            }
            for key, value in before.items()
        }
        # Attention and shared experts may use different TP groups. MLA's
        # projected K/V caches must refresh from the selected local kv_b shard.
        attn = layer.self_attn = torch.nn.Module()
        attn_group = SimpleNamespace(rank_in_group=attn_rank, world_size=attn_size)
        attn.qk_nope_head_dim, attn.v_head_dim = 4, 6
        attn.qk_head_dim = 8
        attn.q_b_proj_qrep = None
        attn.q_b_proj_qrep_weight = attn.w_kc_qrep = None
        attention_before, attention_after = {}, {}
        for name, shape, axis in (
            ("q_b_proj", (32, hidden), 0),
            ("kv_b_proj", (40, hidden), 0),
            ("o_proj", (hidden, 24), 1),
        ):
            module = _linear(RowParallelLinear if axis == 1 else ColumnParallelLinear)
            module.tp_group = attn_group
            value = torch.randn(shape).bfloat16()
            attention_before[name] = value
            attention_after[name] = torch.randn_like(value)
            width = shape[axis] // attn_size
            module.weight = torch.nn.Parameter(
                value.narrow(axis, attn_rank * width, width).clone(),
                requires_grad=False,
            )
            setattr(module.weight, "input_dim" if axis == 1 else "output_dim", axis)
            setattr(attn, name, module)
            inventory[f"model.layers.0.self_attn.{name}.weight"] = {
                "dtype": "BF16",
                "shape": list(shape),
            }
        key, value = attn.kv_b_proj.weight.unflatten(0, (-1, 10)).split([4, 6], dim=1)
        attn.w_kc = key.clone().detach()
        attn.w_vc = value.transpose(1, 2).clone().detach()
        if dcp:
            # TP4 / DCP2, rank3: gathered consumers cover heads2:4, while local
            # parameters cover only head3. The source origin is nonzero for both.
            attn.q_b_proj_qrep_weight = attention_before["q_b_proj"][16:].clone()
            attn.w_kc_qrep = attention_before["kv_b_proj"][20:].reshape(2, 10, hidden)[:, :4].clone()
        plan = layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)
        if dcp:
            # Discovery must pin auxiliary consumers at construction;
            # deferred finishing cannot silently accept a replacement image.
            original_qrep = attn.q_b_proj_qrep_weight
            attn.q_b_proj_qrep_weight = original_qrep.clone()
            with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
                plan.finish()
            attn.q_b_proj_qrep_weight = original_qrep
        plan.finish()
        for binding in plan.bindings:
            if ".self_attn." in binding.name:
                key = binding.name.split(".")[-2]
                mask = byte_view(attention_before[key]) ^ byte_view(attention_after[key])
            else:
                key = binding.name.split(".")[-2].removesuffix("_proj")
                mask = byte_view(before[key]) ^ byte_view(after[key])
            _apply_binding(binding, mask)
        for image in plan.derived:
            image.destination.copy_(image.source)
        plan.check_identity()
        for name, value in attention_after.items():
            axis = 1 if name == "o_proj" else 0
            width = value.shape[axis] // attn_size
            torch.testing.assert_close(
                byte_view(getattr(attn, name).weight),
                byte_view(value.narrow(axis, attn_rank * width, width)),
                rtol=0,
                atol=0,
            )
        kv_width = attention_after["kv_b_proj"].shape[0] // attn_size
        key, value = (
            attention_after["kv_b_proj"]
            .narrow(0, attn_rank * kv_width, kv_width)
            .unflatten(0, (-1, 10))
            .split([4, 6], dim=1)
        )
        torch.testing.assert_close(byte_view(attn.w_kc), byte_view(key), rtol=0, atol=0)
        torch.testing.assert_close(byte_view(attn.w_vc), byte_view(value.transpose(1, 2)), rtol=0, atol=0)
        if dcp:
            torch.testing.assert_close(
                byte_view(attn.q_b_proj_qrep_weight),
                byte_view(attention_after["q_b_proj"][16:]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                byte_view(attn.w_kc_qrep),
                byte_view(attention_after["kv_b_proj"][20:].reshape(2, 10, hidden)[:, :4]),
                rtol=0,
                atol=0,
            )
            original = attn.w_kc_qrep
            attn.w_kc_qrep = original.clone()
            with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
                plan.check_identity()
            attn.w_kc_qrep = original
        torch.testing.assert_close(
            byte_view(shared.gate_up_proj.weight),
            byte_view(torch.cat([after["gate"][selection], after["up"][selection]])),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            byte_view(shared.down_proj.weight),
            byte_view(after["down"][:, selection]),
            rtol=0,
            atol=0,
        )
        shared.down_proj.weight = torch.nn.Parameter(shared.down_proj.weight.clone(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()

    def test_derived_geometry_rejected_at_admission(self):
        for source in (torch.zeros(3), torch.zeros(2, dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError, "derived delta buffer geometry"):
                byte_layout.DerivedImage("consumer", torch.zeros(2), source)

    def test_identity_checks_share_geometry_but_preserve_each_live_binding(self):
        root = torch.nn.Module()
        root.weight = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        root.tied = root.weight
        root.view = root.weight.detach().view(3, 2)
        root.wrapper = SimpleNamespace(config="unchanged")
        plan = layout.GpuDeltaLayout.__new__(layout.GpuDeltaLayout)
        plan._parameter_roots = [
            (root, name, tensor, byte_layout._tensor_identity(tensor)) for name, tensor in root._parameters.items()
        ]
        plan._consumers = [
            byte_layout.ConsumerSnapshot(lambda: (root.weight, root.view, root.wrapper)),
            byte_layout.ConsumerSnapshot(lambda: (root.tied, root.view)),
        ]
        plan._prepare_identity_checks()
        storage = torch.Tensor.untyped_storage
        reads = []

        def read_storage(tensor):
            reads.append(id(tensor))
            return storage(tensor)

        with patch.object(torch.Tensor, "untyped_storage", read_storage):
            plan.check_identity()
            plan.check_identity()
        self.assertEqual(reads, [id(root.weight), id(root.view)] * 2)

        # The tied parameter and consumer can be rebound to objects with exactly
        # the same storage and geometry; every live field still has to match.
        root.tied = torch.nn.Parameter(root.weight.detach(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()
        root.tied = root.weight
        original = root.view
        root.view = original.view_as(original)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        root.view = original
        wrapper = root.wrapper
        root.wrapper = SimpleNamespace(config="unchanged")
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        root.wrapper = wrapper

        # A shared storage pointer cannot hide distinct view geometry. Each new
        # check must also observe in-place changes after an earlier success.
        root.view.transpose_(0, 1)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        root.view.transpose_(0, 1)
        plan.check_identity()
        root.weight.transpose_(0, 1)
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()

    def test_w4a16_calibration_is_static_but_weight_scales_remain_mutable(self):
        prefix = "model.layers.0.mlp.experts"
        layer = SimpleNamespace(
            moe_tp_size=1,
            use_presharded_weights=False,
            quant_method=_nvfp4_method(),
            moe_runner_config=SimpleNamespace(is_gated=True),
            _map_global_expert_id_to_local_expert_id=lambda expert: (expert if expert < 2 else -1),
            w13_input_scale=torch.full((4, 2), 7.0),
            w2_input_scale=torch.full((4,), 11.0),
            w13_weight_scale_2=torch.ones(2, 2),
            w2_weight_scale_2=torch.ones(2),
        )
        plan = byte_layout.ParameterBindings.__new__(byte_layout.ParameterBindings)
        plan.modules, plan.excluded = {prefix: layer}, {}
        plan.moe_layers = {prefix: (layer, FlashInferLayout("nvfp4", "cutedsl", True, "w4a16"))}
        meta = {"dtype": "F32", "shape": [], "encoding": "xor_raw_bytes"}
        for projection in ("gate", "up", "down"):
            for expert in (0, 3):
                name = f"{prefix}.{expert}.{projection}_proj.input_scale"
                self.assertIsNone(plan.bind(name, meta, name))
                self.assertEqual(plan.excluded[name], byte_layout.STATIC_CALIBRATION)
            name = f"{prefix}.0.{projection}_proj.weight_scale_2"
            binding = plan.bind(name, meta, name)
            after = torch.tensor(0.5)
            self.assertEqual(binding.encoding, "xor_raw_bytes")
            torch._foreach_copy_([binding.storage[0]], [after])
            torch.testing.assert_close(byte_view(binding.storage[0]), byte_view(after))
        self.assertTrue(torch.all(layer.w13_input_scale == 7))
        self.assertTrue(torch.all(layer.w2_input_scale == 11))

    def test_admission_reports_all_incompatible_dtypes_without_changing_weights(self):
        root = torch.nn.Module()
        root.config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        root.num_fused_shared_experts = 0
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.indexer = torch.nn.Module()
        layer.self_attn.indexer.k_norm = torch.nn.LayerNorm(3)
        for name, value in (
            ("downcast_scalar", torch.tensor(1.0, dtype=torch.bfloat16)),
            ("downcast_vector", torch.ones(3, dtype=torch.float16)),
            ("upcast_matrix", torch.ones(2, 3, dtype=torch.float32)),
            ("downcast_matrix", torch.ones(2, 3, dtype=torch.bfloat16)),
            ("unchanged_scalar", torch.tensor(0.0, dtype=torch.bfloat16)),
            ("valid_bf16_scalar", torch.tensor(2.0, dtype=torch.float32)),
            ("valid_vector", torch.ones(3, dtype=torch.float32)),
        ):
            layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
        prefix = "model.layers.0."
        mismatches = {
            prefix + "self_attn.indexer.k_norm.weight": ("F16", "F32"),
            prefix + "downcast_scalar": ("F32", "BF16"),
            prefix + "downcast_vector": ("F32", "F16"),
            prefix + "upcast_matrix": ("BF16", "F32"),
            prefix + "downcast_matrix": ("F32", "BF16"),
            # Admission checks the complete inventory, even if this tensor's
            # subsequent delta would be empty because it remains zero.
            prefix + "unchanged_scalar": ("F32", "BF16"),
        }
        compatible = {
            prefix + "self_attn.indexer.k_norm.bias": "BF16",
            prefix + "valid_bf16_scalar": "BF16",
            prefix + "valid_vector": "F32",
        }
        parameters = dict(root.named_parameters())
        inventory = {
            name: {
                "dtype": mismatches[name][0] if name in mismatches else compatible[name],
                "shape": list(value.shape),
            }
            for name, value in parameters.items()
        }
        before = {name: (byte_view(value).clone(), value.data_ptr()) for name, value in parameters.items()}

        with self.assertRaises(ValueError) as caught:
            layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)

        message = str(caught.exception)
        self.assertIn(f"Found {len(mismatches)} mismatched tensors", message)
        for name, (checkpoint_dtype, served_dtype) in mismatches.items():
            rows = [row for row in message.splitlines() if name in row]
            self.assertEqual(len(rows), 1, message)
            self.assertIn(f"checkpoint={checkpoint_dtype}", rows[0])
            self.assertIn("served=", rows[0])
            self.assertIn(served_dtype, rows[0].split("served=", 1)[1])
        for name in compatible:
            self.assertNotIn(name, message)
        for name, value in parameters.items():
            expected, pointer = before[name]
            self.assertTrue(torch.equal(byte_view(value), expected), name)
            self.assertEqual(value.data_ptr(), pointer)

    def test_glm_mapping_matches_loader_views_and_preserves_consumers(self):
        root = torch.nn.Module()
        root.config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            q_lora_rank=3,
            kv_lora_rank=2,
            qk_rope_head_dim=2,
            n_routed_experts=1,
        )
        root.num_fused_shared_experts = 1
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.indexer = torch.nn.Module()
        norm = layer.self_attn.indexer.k_norm = torch.nn.LayerNorm(128)
        inventory = {
            f"model.layers.0.self_attn.indexer.k_norm.{key}": {
                "dtype": "BF16",
                "shape": [128],
            }
            for key in ("weight", "bias")
        }
        attn = layer.self_attn
        attn.w_kc_qrep = attn.q_b_proj_qrep_weight = attn.q_b_proj_qrep = None
        attn.fused_qkv_a_proj_with_mqa = torch.nn.Linear(8, 7, bias=False).bfloat16()
        attn.indexer.wk_weights_proj = torch.nn.Linear(8, 9, bias=False).bfloat16()
        attn.kv_b_proj = torch.nn.Linear(8, 20, bias=False).bfloat16()
        attn.qk_nope_head_dim, attn.v_head_dim = 4, 6
        key, value = attn.kv_b_proj.weight.unflatten(0, (2, 10)).split([4, 6], dim=1)
        attn.w_kc = key.transpose(1, 2).contiguous().transpose(1, 2).detach()
        attn.w_vc = value.contiguous().transpose(1, 2).detach()
        expected_views = {
            "q_a_proj.weight": attn.fused_qkv_a_proj_with_mqa.weight[:3],
            "kv_a_proj_with_mqa.weight": attn.fused_qkv_a_proj_with_mqa.weight[3:],
            "indexer.wk.weight": attn.indexer.wk_weights_proj.weight[:5],
            "indexer.weights_proj.weight": attn.indexer.wk_weights_proj.weight[5:],
            "kv_b_proj.weight": attn.kv_b_proj.weight,
        }
        prefix = "model.layers.0.self_attn."
        inventory.update(
            {prefix + name: {"dtype": "BF16", "shape": list(target.shape)} for name, target in expected_views.items()}
        )
        inventory.update(
            {
                "model.layers.1.weight": {"dtype": "BF16", "shape": [2, 2]},
                prefix + "rotary_emb.inv_freq": {"dtype": "F32", "shape": [2]},
            }
        )
        layer.mlp = torch.nn.Module()
        experts = layer.mlp.experts = torch.nn.Module()
        experts.moe_tp_size, experts.use_presharded_weights = 1, False
        experts.quant_method = _nvfp4_method()
        experts.moe_runner_config = SimpleNamespace(
            is_gated=True, activation="silu", gemm1_clamp_limit=None, swiglu_limit=None
        )
        experts.num_local_experts, experts.num_experts = 1, 2
        experts.moe_ep_rank = 1
        experts.w13_input_scale_quant = experts.w2_input_scale_quant = torch.ones(1)
        experts._map_global_expert_id_to_local_expert_id = lambda expert: expert - 1
        experts.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
        experts.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        experts.g1_alphas, experts.g1_alphas_up, experts.g2_alphas = [torch.ones(1) for _ in range(3)]
        experts._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
        experts._cutedsl_scales = (torch.ones(1), torch.ones(1), torch.ones(1))
        experts._cutedsl_input_scale = torch.ones(1)
        experts.quant_config = SimpleNamespace(use_per_token_activation=False)
        shared_scale = "model.layers.0.mlp.shared_experts.gate_proj.weight_scale_2"
        inventory[shared_scale] = {
            "dtype": "F32",
            "shape": [],
        }
        moe = SimpleNamespace(
            enable_eplb=False,
            elastic_ep_backend=None,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            moe_a2a_backend="none",
            moe_runner_backend="flashinfer_cutedsl",
        )
        with (
            patch.dict(
                sys.modules,
                {"sglang.srt.runtime_context": SimpleNamespace(get_exec=lambda: SimpleNamespace(moe=moe))},
            ),
            patch.object(
                layout,
                "_require_fixed_moe_topology",
                wraps=layout._require_fixed_moe_topology,
            ) as require_topology,
        ):
            plan = layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)
            plan.finish()
        require_topology.assert_called_once_with(moe)
        self.assertEqual(len(plan.excluded), 2)
        for binding in plan.bindings:
            if binding.name == shared_scale:
                self.assertEqual(
                    binding.storage[0].data_ptr(),
                    experts.w13_weight_scale_2[0, 0].data_ptr(),
                )
            relative = binding.name.removeprefix(prefix)
            if relative in expected_views:
                target = expected_views[relative]
                self.assertEqual(binding.shape, tuple(target.shape))
                self.assertEqual(binding.slices, [[0, n] for n in target.shape])
                self.assertEqual(binding.storage[0].data_ptr(), target.data_ptr())
                self.assertEqual(binding.storage[0].stride(), target.stride())
        for image, source in zip(plan.derived[-2:], (key, value.transpose(1, 2)), strict=False):
            self.assertEqual(image.source.data_ptr(), source.data_ptr())
            self.assertEqual(image.source.stride(), source.stride())
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), 1.25, -3.5, 2.0**-133, -16],
            dtype=torch.bfloat16,
        ).repeat(16)
        with torch.no_grad():
            for binding in plan.bindings:
                if ".k_norm." not in binding.name:
                    continue
                self.assertEqual(binding.encoding, "xor_raw_bytes")
                target = getattr(norm, binding.name.rsplit(".", 1)[1])
                # A zero target still needs the old-byte XOR mask.
                source = torch.zeros_like(values) if binding.name.endswith("bias") else values
                target.fill_(7)
                pointer = target.data_ptr()
                expected = torch.empty_like(target)
                expected.copy_(source)  # The ordinary default_weight_loader.
                mask = byte_view(target.to(source.dtype)) ^ byte_view(source)
                for destination, physical_mask, _ in byte_layout.prepare_raw_xor([(binding, mask)]):
                    destination.view(torch.uint8).bitwise_xor_(physical_mask.view(destination.view(torch.uint8).shape))
                torch.testing.assert_close(byte_view(target), byte_view(expected), rtol=0, atol=0)
                self.assertEqual(target.data_ptr(), pointer)
        plan.check_identity()
        original_key = attn.w_kc
        attn.w_kc = attn.w_kc.clone()
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        attn.w_kc = original_key
        experts._cutedsl_scales = list(experts._cutedsl_scales)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()

    def test_one_head_mla_value_cache_is_kv_b_itself(self):
        # Attention TP4 of 4 heads: rank 2 holds one head, so the loader's w_vc is a view of kv_b_proj.
        root, attn, canonical, inventory = _mla_layer(attn_rank=2, attn_size=4, heads=4)
        kv_b = attn.kv_b_proj.weight
        self.assertEqual(attn.w_vc.untyped_storage().data_ptr(), kv_b.untyped_storage().data_ptr())
        plan = layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)
        plan.finish()
        # Only the separate w_kc copy refreshes; w_vc is the bound canonical storage.
        self.assertEqual([image.name for image in plan.derived], ["model.layers.0.self_attn.w_kc"])
        updated = {name: torch.randn_like(value) for name, value in canonical.items()}
        for binding in plan.bindings:
            name = binding.name.split(".")[-2]
            _apply_binding(binding, byte_view(canonical[name]) ^ byte_view(updated[name]))
        for image in plan.derived:
            image.refresh(torch.tensor(True))
        plan.check_identity()
        key, value = updated["kv_b_proj"][20:30].unflatten(0, (1, -1)).split([_NOPE, _V], dim=1)
        torch.testing.assert_close(byte_view(attn.w_kc), byte_view(key), rtol=0, atol=0)
        torch.testing.assert_close(byte_view(attn.w_vc), byte_view(value.transpose(1, 2)), rtol=0, atol=0)

    def test_dcp_replicas_match_companion_binding(self):
        # Attention TP4 / DCP2, rank 3 of 8 heads: local heads 6:8, gathered heads 4:8. The companion
        # gathers separate copies: the local segment comes first, then one affine segment per image.
        root, attn, _, inventory = _mla_layer(attn_rank=3, attn_size=4, heads=8, dcp=2)
        plan = layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)
        q, kv = (next(b for b in plan.bindings if b.name.endswith(f"{n}.weight")) for n in ("q_b_proj", "kv_b_proj"))
        self.assertEqual(q.slices, [[32, 64], [0, _LORA]])
        self.assertEqual(q.storage, (attn.q_b_proj.weight, attn.q_b_proj_qrep_weight))
        self.assertEqual(
            [(segment.slices, segment.source_shape) for segment in q.segments],
            [(((48, 64), (0, _LORA)), None), (((32, 64), (0, _LORA)), None)],
        )
        self.assertEqual(kv.slices, [[40, 80], [0, _LORA]])
        self.assertEqual(kv.storage, (attn.kv_b_proj.weight, attn.w_kc_qrep))
        # A logical head axis selects the K rows 0:4 of heads 4:8 in one view.
        self.assertEqual(
            [(segment.slices, segment.source_shape) for segment in kv.segments],
            [(((60, 80), (0, _LORA)), None), (((4, 8), (0, _NOPE), (0, _LORA)), (8, _NOPE + _V, _LORA))],
        )
        for binding, local, image in (
            (q, attn.q_b_proj.weight, attn.q_b_proj_qrep_weight),
            (kv, None, attn.w_kc_qrep),
        ):
            if local is not None:
                self.assertEqual(binding.segments[0].destinations[0].data_ptr(), local.data_ptr())
            self.assertEqual(binding.segments[-1].destinations[0].data_ptr(), image.data_ptr())
        # The companion snapshot pins w_kc, w_vc, w_kc_qrep, q_b_proj_qrep_weight and q_b_proj_qrep.
        (snapshot,) = plan._consumers
        self.assertEqual(
            snapshot.objects,
            (attn.w_kc, attn.w_vc, attn.w_kc_qrep, attn.q_b_proj_qrep_weight, attn.q_b_proj_qrep),
        )

    def test_dcp_replicas_bind_main_alias_once_and_transposed_value(self):
        heads = 8
        for attn_rank, attn_size, dcp in ((3, 4, 2), (0, 4, 2), (1, 2, 2), (5, 8, 4)):
            with self.subTest(attn_rank=attn_rank, attn_size=attn_size, dcp=dcp):
                root, attn, canonical, inventory = _mla_layer(attn_rank, attn_size, heads, dcp, storage="main")
                local_q = attn.q_b_proj.weight
                self.assertEqual(
                    local_q.untyped_storage().data_ptr(), attn.q_b_proj_qrep_weight.untyped_storage().data_ptr()
                )
                plan = layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)
                plan.finish()
                q = next(b for b in plan.bindings if b.name.endswith("q_b_proj.weight"))
                # The gathered image carries the local rows, so they take the mask once.
                self.assertEqual(q.storage, (attn.q_b_proj_qrep_weight,))
                self.assertEqual(len(q.segments), 1)
                updated = {name: torch.randn_like(value) for name, value in canonical.items()}
                for binding in plan.bindings:
                    name = binding.name.split(".")[-2]
                    _apply_binding(binding, byte_view(canonical[name]) ^ byte_view(updated[name]))
                for image in plan.derived:
                    image.refresh(torch.tensor(True))
                plan.check_identity()
                local_heads = heads // attn_size
                local = slice(attn_rank * local_heads, (attn_rank + 1) * local_heads)
                first = attn_rank // dcp * dcp * local_heads
                group = slice(first, first + dcp * local_heads)
                q_heads = updated["q_b_proj"].unflatten(0, (heads, -1))
                kv_heads = updated["kv_b_proj"].unflatten(0, (heads, -1))
                for live, expected in (
                    (local_q, q_heads[local].flatten(0, 1)),
                    (attn.q_b_proj_qrep_weight, q_heads[group].flatten(0, 1)),
                    (attn.kv_b_proj.weight, kv_heads[local].flatten(0, 1)),
                    (attn.w_kc, kv_heads[local, :_NOPE]),
                    (attn.w_vc, kv_heads[local, _NOPE:].transpose(1, 2)),
                    (attn.w_kc_qrep, kv_heads[group, :_NOPE]),
                    # all_gather(w_vc.contiguous()): [heads, kv_lora_rank, v_head_dim].
                    (attn.w_vc_qrep, kv_heads[group, _NOPE:].transpose(1, 2)),
                ):
                    torch.testing.assert_close(byte_view(live), byte_view(expected), rtol=0, atol=0)
                # SGLang main's snapshot also pins the gathered V image.
                (snapshot,) = plan._consumers
                self.assertIs(snapshot.objects[3], attn.w_vc_qrep)
                original = attn.w_vc_qrep
                attn.w_vc_qrep = original.clone()
                with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
                    plan.check_identity()
                attn.w_vc_qrep = original

    def test_dcp_replicas_reject_storage_no_tree_builds(self):
        def admit(edit):
            root, attn, _, inventory = _mla_layer(attn_rank=3, attn_size=4, heads=8, dcp=2, storage="main")
            edit(attn)
            layout.GpuDeltaLayout(root, _glm_checkpoint(inventory), None)

        def misaligned(attn):
            # A local q_b view of another rank's rows of the gathered image.
            attn.q_b_proj.weight.data = attn.q_b_proj_qrep_weight[: attn.q_b_proj.weight.shape[0]]

        def undeclared(attn):
            # Both trees declare w_kc_qrep on every MLA layer; a class declaring neither image is neither tree.
            del attn.w_kc_qrep, attn.w_vc_qrep

        for edit, message in (
            (misaligned, "unsupported DCP replicated projection alias"),
            (undeclared, "unsupported DCP replicated projection layout: Module declares no w_kc_qrep"),
            (lambda attn: setattr(attn, "w_kc_qrep", None), "a gathered V image without its K image"),
            (lambda attn: setattr(attn, "w_vc_qrep", attn.w_vc_qrep[:, :, 1:]), "replicated projection geometry"),
            (lambda attn: setattr(attn, "w_vc_qrep", attn.w_vc_qrep[:1]), "replicated projection geometry"),
            (lambda attn: setattr(attn, "q_b_proj_qrep", object()), "requires unquantized DCP Q replication"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                admit(edit)

    def test_fused_qkv_a_view_takes_explicit_loader_sizes(self):
        fused = torch.arange(7 * 2).reshape(7, 2)
        meta = {"dtype": "BF16", "shape": [3, 2]}
        sizes = {"shape": meta["shape"], "q_rows": 3, "kv_rows": 4}
        q = deepseek_v2.fused_qkv_a_view(fused, "q_a_proj", **sizes)
        kv = deepseek_v2.fused_qkv_a_view(fused, "kv_a_proj_with_mqa", **sizes)
        self.assertEqual((q.data_ptr(), q.shape), (fused.data_ptr(), (3, 2)))
        self.assertEqual((kv.data_ptr(), kv.shape), (fused[3:].data_ptr(), (4, 2)))
        with self.assertRaisesRegex(ValueError, "unclassified fused Q/KV-A projection: q_b_proj"):
            deepseek_v2.fused_qkv_a_view(fused, "q_b_proj", **sizes)
        with self.assertRaisesRegex(ValueError, "fused scalar projection metadata"):
            deepseek_v2.fused_qkv_a_view(fused, "q_a_proj", **(sizes | {"shape": []}))

    def test_checkpoint_policy_excludes_static_tensors_before_the_mapping(self):
        policy = publication.StaticPolicy(
            num_hidden_layers=1, draft=re.compile(r"^mtp\."), frozen=re.compile(r"^vision\.")
        )
        inventory = {
            "layers.0.attn.wq_a.scale": {"dtype": "U8", "shape": [2, 3]},
            "layers.0.attn.wq_a.weight": {"dtype": "U8", "shape": [64, 96]},
            "layers.0.attn_norm.weight": {"dtype": "BF16", "shape": [96]},
            "layers.1.attn_norm.weight": {"dtype": "BF16", "shape": [96]},
            "layers.0.attn.rotary_emb.inv_freq": {"dtype": "F32", "shape": [2]},
            "mtp.0.attn.wq_a.weight": {"dtype": "U8", "shape": [64, 96]},
            "vision.norm.weight": {"dtype": "BF16", "shape": [96]},
        }
        bound = {}

        def bind(name, meta):
            # A recording stand-in for a model mapping; it binds nothing.
            bound[name] = dict(meta)

        mapping = SimpleNamespace(
            parameters=SimpleNamespace(excluded={}, modules={}, moe_layers={}, shadowed={}), consumers=[], bind=bind
        )
        with patch.object(layout, "model_mapping", return_value=mapping):
            plan = layout.GpuDeltaLayout(SimpleNamespace(), CanonicalCheckpoint(inventory, policy, {}), None)
        self.assertEqual(
            plan.excluded,
            {
                "layers.1.attn_norm.weight": publication.STATIC_DRAFT,
                "layers.0.attn.rotary_emb.inv_freq": publication.COMPUTED_ROTARY,
                "mtp.0.attn.wq_a.weight": publication.STATIC_DRAFT,
                "vision.norm.weight": publication.FROZEN_MODULE,
            },
        )
        # Wire encodings come from the shared shape rule, so the plan digest agrees with the trainer's.
        expected = {
            name: inventory[name] | {"encoding": publication.canonical_encoding(inventory[name]["shape"])}
            for name in inventory
        }
        self.assertEqual(plan.inventory, expected)
        # No static policy sends a matrix raw, however small: only scalars, vectors and the plan's block-128 UE8M0
        # expert scales travel as raw masks.
        self.assertEqual(plan.inventory["layers.0.attn.wq_a.scale"]["encoding"], "xor_bytes")
        self.assertEqual(plan.inventory["layers.0.attn_norm.weight"]["encoding"], "xor_raw_bytes")
        self.assertEqual(bound, {name: expected[name] for name in inventory if name not in plan.excluded})
        self.assertEqual(inventory["layers.0.attn.wq_a.scale"], {"dtype": "U8", "shape": [2, 3]})

    def test_bindings_apply_the_encoding_the_canonical_plan_publishes(self):
        # The plan publishes vectors as raw masks: a byte binding would never select its row at apply.
        policy = publication.StaticPolicy(num_hidden_layers=1)
        inventory = {
            "layers.0.attn.wq_a.weight": {"dtype": "U8", "shape": [2, 3]},
            "layers.0.attn_norm.weight": {"dtype": "BF16", "shape": [3]},
        }
        live = {
            name: torch.zeros(meta["shape"], dtype=torch.bfloat16 if "norm" in name else torch.uint8)
            for name, meta in inventory.items()
        }

        def mapping(bind):
            parameters = byte_layout.ParameterBindings(torch.nn.Module())
            return SimpleNamespace(
                parameters=parameters,
                consumers=[],
                bind=lambda name, meta: bind(parameters, name, meta),
            )

        def bind_view(parameters, name, meta):
            return parameters.bind_view(name, meta, None, live[name], [[0, extent] for extent in meta["shape"]])

        checkpoint = CanonicalCheckpoint(inventory, policy, {})
        with patch.object(layout, "model_mapping", return_value=mapping(bind_view)):
            plan = layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, None)
        self.assertEqual(
            {binding.name: binding.encoding for binding in plan.bindings},
            {name: meta["encoding"] for name, meta in plan.inventory.items()},
        )
        # The vector XORs its typed storage; only the matrix gets byte views.
        self.assertEqual(
            {binding.name for binding in plan.bindings if binding.is_raw and not binding.destinations},
            {"layers.0.attn_norm.weight"},
        )
        self.assertEqual(
            {binding.name for binding in plan.bindings if binding.destinations}, {"layers.0.attn.wq_a.weight"}
        )

        def byte_binding(parameters, name, meta):
            # A mapping-built binding that ignores the plan's encoding.
            return byte_layout._direct_binding(name, meta | {"encoding": "xor_bytes"}, live[name])

        with (
            patch.object(layout, "model_mapping", return_value=mapping(byte_binding)),
            self.assertRaisesRegex(
                ValueError,
                "^binding encoding xor_bytes differs from the canonical plan's xor_raw_bytes: layers.0.attn_norm.weight$",
            ),
        ):
            layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, None)

    def test_derived_updates_patch_the_raw_pieces_a_merged_loader_writes(self):
        # TP2 rank0 of a [4, 4] merged bias: canonical rows [0, 2) and [4, 6) fill local [0, 2) and [2, 4).
        old = torch.arange(8, dtype=torch.float32)
        new = old * 3 + 1
        param = torch.nn.Parameter(old[[0, 1, 4, 5]].clone(), requires_grad=False)
        # One raw-mask binding whose segments are the loader's typed views of the parameter.
        binding = byte_layout.TensorBinding(
            "model.layers.0.mixer.conv1d.bias",
            "F32",
            (8,),
            [[0, 8]],
            (param,),
            encoding="xor_raw_bytes",
            segments=(DestinationSegment(((0, 2),), (param[0:2],)), DestinationSegment(((4, 6),), (param[2:4],))),
        )
        update = DerivedUpdate((param,), (torch.zeros(4),), lambda values: (values[0] * 2,))
        mapping = SimpleNamespace(
            parameters=SimpleNamespace(
                moe_layers={"experts": (object(), SimpleNamespace(expert_derived_updates=lambda layer: [update]))},
                shadowed={},
            ),
            consumers=[],
            finish=lambda: None,
            derived=[],
            aliases=lambda a, b: False,
        )
        plan = _bare_layout([binding], mapping)
        context = SimpleNamespace(get_exec=lambda: SimpleNamespace(moe=_fixed_moe()))
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
            plan.finish()
        views = binding.raw_bindings()
        self.assertEqual(plan.derived_targets, {id(target) for view in views for target in view.storage})
        # Predicted raw pieces reach the update, which sees the complete new loader image.
        mask = (old.view(torch.int32) ^ new.view(torch.int32)).view(torch.uint8)
        prepared = byte_layout.prepare_raw_xor([(view, mask) for view in views], plan.derived_targets)
        raw_values = {id(target): value for target, _, value in prepared if value is not None}
        (copy,) = prepare_updates(plan.derived_updates, raw_values)
        torch.testing.assert_close(copy.source, new[[0, 1, 4, 5]] * 2, rtol=0, atol=0)
        torch.testing.assert_close(param, old[[0, 1, 4, 5]], rtol=0, atol=0)

    def test_alias_groups_admit_shared_storage_only_for_declared_views(self):
        fused = torch.zeros(10, 4, dtype=torch.bfloat16)
        prefix = "model.layers.0.self_attn."
        bindings = [
            byte_layout._direct_binding(
                prefix + name,
                {"dtype": "BF16", "shape": [stop - start, 4], "encoding": "xor_bytes"},
                fused[start:stop],
            )
            for name, start, stop in (("k_proj.weight", 4, 6), ("q_proj.weight", 0, 4), ("v_proj.weight", 6, 10))
        ]
        groups = byte_layout.AliasGroups()
        for binding in bindings[:2]:
            groups.declare(binding.name, prefix + "qkv_proj.weight", binding, fused)
        plan = _bare_layout(bindings, None)
        with self.assertRaisesRegex(ValueError, "unclassified live weight alias.*v_proj"):
            plan._reject_overlaps(groups)
        groups.declare(bindings[2].name, prefix + "qkv_proj.weight", bindings[2], fused)
        plan._reject_overlaps(groups)

    def test_images_read_canonical_storage_and_computed_images_need_cuda_consumers(self):
        weight, bias = torch.zeros(4), torch.zeros(4)
        bindings = [
            byte_layout._direct_binding(
                f"model.norm.{name}", {"dtype": "F32", "shape": [4], "encoding": "xor_raw_bytes"}, tensor
            )
            for name, tensor in (("weight", weight), ("bias", bias))
        ]
        served, cache, orphan = torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4), torch.zeros(4)
        # What the mapping pinned while binding: every image destination except the orphan.
        pinned = ConsumerSnapshot(lambda: (served, cache))

        def round_to_served(value):
            return (value.to(torch.bfloat16),)

        def finish(*images, updates=(), late=None):
            moe = {"experts": (None, SimpleNamespace(expert_derived_updates=lambda layer: list(updates)))}
            consumers = [pinned]
            mapping = SimpleNamespace(
                parameters=SimpleNamespace(moe_layers=moe if updates else {}, shadowed={}),
                consumers=consumers,
                finish=lambda: late and consumers.append(late),
                derived=list(images),
                aliases=lambda a, b: False,
            )
            plan = _bare_layout(bindings, mapping, consumers)
            context = SimpleNamespace(get_exec=lambda: SimpleNamespace(moe=_fixed_moe()))
            with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
                plan.finish()
            return plan

        computed = byte_layout.ComputedImage("norm", (weight,), (served,), round_to_served)
        for images, updates, message in (
            # A computed destination never overlaps canonical storage, and every source is canonical.
            ((byte_layout.ComputedImage("norm", (bias,), (weight,), lambda value: (value,)),), (), "overlaps"),
            ((byte_layout.ComputedImage("norm", (served,), (torch.zeros(4),), lambda v: (v.float(),)),), (), "owns"),
            # A copy of storage no binding owns would never be selected for refresh.
            (
                (byte_layout.DerivedImage("copy", torch.zeros(4), torch.zeros(4)),),
                (),
                "^derived image copy reads storage no canonical binding owns$",
            ),
            # No image reads another image's or a prepared update's destination.
            ((computed, byte_layout.DerivedImage("copy", torch.zeros(4, dtype=torch.bfloat16), served)), (), "copy"),
            ((computed,), [DerivedUpdate((torch.zeros(1),), (weight,), lambda values: values)], "derived destination"),
            # A destination no snapshot pins could be replaced while refreshes kept reaching the old tensor.
            (
                (byte_layout.DerivedImage("copy", orphan, weight),),
                (),
                "^derived image copy writes an unpinned consumer$",
            ),
            (
                (byte_layout.ComputedImage("norm", (weight,), (orphan,), lambda value: (value,)),),
                (),
                "^computed image norm writes an unpinned consumer$",
            ),
            # CPU offload cannot feed a computed image refreshed on the apply stream.
            ((computed,), (), "computed images require CUDA-resident storage: norm"),
            ((SimpleNamespace(name="other"),), (), "unsupported derived image: SimpleNamespace"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                finish(*images, updates=updates)
        copy = byte_layout.DerivedImage("copy", cache, weight)
        plan = finish(copy)
        self.assertEqual((plan.derived, plan.computed), ([copy], []))
        # A snapshot appended in finish() was never checked at describe.
        with self.assertRaisesRegex(ValueError, "pin consumers while binding at describe, never in finish"):
            finish(copy, late=ConsumerSnapshot(lambda: (orphan,)))

    def test_shadows_fill_from_the_checkpoint_at_the_first_prepare_of_version_zero(self):
        name = "model.layers.0.mixer.A_log"
        values = torch.randn(4, 3).bfloat16()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({name: values}, str(path))
            policy = publication.static_policy({"architectures": ["GlmMoeDsaForCausalLM"], "num_hidden_layers": 1})
            inventory = {name: {"dtype": "BF16", "shape": [4, 3]}}
            checkpoint = CanonicalCheckpoint(inventory, policy, publication.read_safetensors_headers([str(path)]))
            filled_before_mapping_finish = []

            def mapping(model, shadows):
                parameters = byte_layout.ParameterBindings(torch.nn.Module(), shadows=shadows)

                def bind(canonical, meta):
                    # This rank serves rows [1, 3) only through arithmetic on them.
                    shadow = parameters.shadow(canonical, meta, (1, 3), torch.device("cpu"))
                    return parameters.bind_view(canonical, meta, None, shadow, [[1, 3], [0, 3]])

                return SimpleNamespace(
                    parameters=parameters,
                    consumers=[],
                    derived=[],
                    bind=bind,
                    finish=lambda: filled_before_mapping_finish.append(store[name].filled),
                    aliases=lambda a, b: False,
                )

            store = {}
            with (
                patch.object(layout, "model_mapping", side_effect=mapping),
                # This CPU test checks filling; the CUDA computed image that must read the shadow is admitted in
                # test_a_canonical_shadow_must_feed_a_computed_image and the manual CUDA test.
                patch.object(layout.GpuDeltaLayout, "_admit_images"),
            ):
                # Describe binds an unfilled shadow; the first prepare fills it before the mapping finishes.
                plan = layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, byte_layout.CanonicalShadows(store, True))
                (binding,) = plan.bindings
                self.assertFalse(store[name].filled)
                self.assertIs(binding.storage[0], store[name].tensor)
                plan.finish()
                self.assertEqual(filled_before_mapping_finish, [True])
                self.assertTrue(torch.equal(store[name].tensor.view(torch.int16), values[1:3].view(torch.int16)))
                self.assertIsNone(plan._checkpoint)

                # An applied update XORs the shadow; re-admission at version 1 keeps that committed value.
                store[name].tensor.view(torch.int16).bitwise_xor_(0x0F0F)
                committed = store[name].tensor.clone()
                missing = CanonicalCheckpoint(
                    inventory, policy, {name: (str(Path(directory) / "gone"), 8, 24, "BF16", [4, 3])}
                )
                later = layout.GpuDeltaLayout(SimpleNamespace(), missing, byte_layout.CanonicalShadows(store, False))
                later.finish()
                self.assertTrue(torch.equal(store[name].tensor, committed))

                # Without its shadow a later admission cannot rebuild the committed version.
                with self.assertRaisesRegex(
                    ValueError, "unavailable after version 0; restart the engine from the HF base"
                ):
                    layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, byte_layout.CanonicalShadows({}, False))
                with self.assertRaisesRegex(ValueError, "lossy loader cast needs a canonical shadow"):
                    layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, None)

    def test_a_canonical_shadow_must_feed_a_computed_image(self):
        # Updates XOR the receiver-owned shadow; only a computed image serves its value.
        name = "model.layers.0.mixer.A_log"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({name: torch.randn(4).bfloat16()}, str(path))
            policy = publication.static_policy({"architectures": ["GlmMoeDsaForCausalLM"], "num_hidden_layers": 1})
            inventory = {name: {"dtype": "BF16", "shape": [4]}}
            checkpoint = CanonicalCheckpoint(inventory, policy, publication.read_safetensors_headers([str(path)]))
            served = torch.zeros(4)

            def mapping(images):
                def build(model, shadows):
                    parameters = byte_layout.ParameterBindings(torch.nn.Module(), shadows=shadows)

                    def bind(canonical, meta):
                        shadow = parameters.shadow(canonical, meta, (0, 4), torch.device("cpu"))
                        return parameters.bind_view(canonical, meta, None, shadow, [[0, 4]])

                    def finish():
                        derived.extend(images(parameters.shadowed[name]))

                    derived = []
                    return SimpleNamespace(
                        parameters=parameters,
                        consumers=[ConsumerSnapshot(lambda: (served,))],
                        derived=derived,
                        bind=bind,
                        finish=finish,
                        aliases=lambda a, b: False,
                    )

                return build

            def finish(images):
                with patch.object(layout, "model_mapping", side_effect=mapping(images)):
                    plan = layout.GpuDeltaLayout(SimpleNamespace(), checkpoint, byte_layout.CanonicalShadows({}, True))
                    plan.finish()

            with self.assertRaisesRegex(ValueError, f"^canonical shadow {name} feeds no computed image$"):
                finish(lambda shadow: [])
            # A shadow some computed image reads is held to that image's CUDA residency.
            with self.assertRaisesRegex(ValueError, "computed images require CUDA-resident storage: mixer.A"):
                finish(
                    lambda shadow: [ComputedImage("mixer.A", (shadow,), (served,), lambda log: (-log.float().exp(),))]
                )

    def test_movable_or_reordered_experts_rejected_before_plan(self):
        defaults = dict(
            enable_eplb=False,
            elastic_ep_backend=None,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            moe_a2a_backend="none",
            moe_runner_backend="flashinfer_cutedsl",
        )
        layout._require_fixed_moe_topology(SimpleNamespace(**defaults))
        for key, value in (
            ("enable_eplb", True),
            ("elastic_ep_backend", "nixl"),
            ("init_expert_location", "custom.json"),
            ("ep_num_redundant_experts", 2),
            ("moe_a2a_backend", "deepep"),
            ("moe_runner_backend", "triton"),
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "fixed trivial EP"),
            ):
                layout._require_fixed_moe_topology(SimpleNamespace(**(defaults | {key: value})))

    def test_invalid_layout_geometry_rejected(self):
        prefix = "model.layers.0.mlp.experts"
        name = prefix + ".0.gate_proj.input_scale"
        for override in (
            {"use_presharded_weights": True},
            {"quant_method": type("UnadmittedMoE", (), {"runner": None})()},
        ):
            layer = SimpleNamespace(
                **(
                    dict(
                        moe_tp_size=1,
                        use_presharded_weights=False,
                        quant_method=_nvfp4_method(),
                        moe_runner_config=SimpleNamespace(is_gated=True),
                    )
                    | override
                )
            )
            plan = byte_layout.ParameterBindings.__new__(byte_layout.ParameterBindings)
            plan.modules, plan.moe_layers, plan.excluded = {prefix: layer}, {}, {}
            with (
                self.subTest(override=override),
                self.assertRaises(ValueError),
            ):
                plan.bind(name, {"dtype": "F32", "shape": []}, name)
            self.assertFalse(plan.moe_layers)


if __name__ == "__main__":
    unittest.main()
