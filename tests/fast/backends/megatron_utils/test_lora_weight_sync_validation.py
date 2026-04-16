"""Tests for LoRA weight-sync validation logic.

Verifies that silent failures are caught:
- Engine returning success=False raises RuntimeError
- Empty LoRA weights after filtering raises RuntimeError
- Zero weight chunks from iterator raises RuntimeError
- FlattenedTensorBucket round-trip preserves tensor values
"""

from argparse import Namespace
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
    ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(8, 2)),
    ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(2, 8)),
]

SAMPLE_BASE_ONLY_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
    ("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 4)),
]


@dataclass
class _FakeEngineResult:
    """Mimics sglang's LoRAUpdateOutput / weight-sync result."""

    success: bool
    error_message: str | None = None


def _make_args(**overrides):
    defaults = dict(
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["linear_qkv", "linear_proj"],
        megatron_to_hf_mode="bridge",
        rollout_num_gpus_per_engine=1,
        hf_checkpoint="/fake/path",
        update_weight_buffer_size=1 << 30,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        pause_generation_mode="retract",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# _check_weight_sync_results
# ---------------------------------------------------------------------------


class TestCheckWeightSyncResults:
    """Validate that _check_weight_sync_results raises on engine failures."""

    def test_success_results_pass(self):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _check_weight_sync_results

        results = [_FakeEngineResult(success=True)]
        _check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_lora(self):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _check_weight_sync_results

        results = [_FakeEngineResult(success=False, error_message="incompatible format")]
        with pytest.raises(RuntimeError, match="LoRA weight sync failed"):
            _check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_base(self):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _check_weight_sync_results

        results = [_FakeEngineResult(success=False, error_message="bad version")]
        with pytest.raises(RuntimeError, match="Base model weight sync failed"):
            _check_weight_sync_results(results, is_lora=False)

    def test_plain_tuple_results_pass(self):
        """Non-dataclass results (e.g. (True, 'Success') tuples) should not raise."""
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _check_weight_sync_results

        results = [(True, "Success")]
        _check_weight_sync_results(results, is_lora=False)

    def test_mixed_results_raises_on_first_failure(self):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _check_weight_sync_results

        results = [
            _FakeEngineResult(success=True),
            _FakeEngineResult(success=False, error_message="oops"),
        ]
        with pytest.raises(RuntimeError, match="oops"):
            _check_weight_sync_results(results, is_lora=True)


# ---------------------------------------------------------------------------
# _send_hf_params: empty LoRA weight detection
# ---------------------------------------------------------------------------


class TestSendHfParamsEmptyLoraDetection:
    """When is_lora=True but HF chunk has no lora_A/lora_B names, raise immediately."""

    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_raises_when_no_lora_weights_in_chunk(self, mock_iter_base, mock_dist):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        args = _make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        updater._ipc_engine = MagicMock()
        updater._ipc_gather_src = 0
        updater._ipc_gather_group = MagicMock()
        updater.use_distribute = False

        with pytest.raises(RuntimeError, match="no LoRA weights"):
            updater._send_hf_params(SAMPLE_BASE_ONLY_WEIGHTS)

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_passes_when_lora_weights_present(self, mock_iter_base, mock_dist, mock_send):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = MagicMock()

        args = _make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        updater._ipc_engine = MagicMock()
        updater._ipc_gather_src = 0
        updater._ipc_gather_group = MagicMock()
        updater.use_distribute = False

        refs, _ = updater._send_hf_params(SAMPLE_LORA_WEIGHTS)
        # Should not raise; mock_send was called with the LoRA tensors
        assert mock_send.called


# ---------------------------------------------------------------------------
# update_weights: zero-chunk detection
# ---------------------------------------------------------------------------


class TestUpdateWeightsZeroChunks:
    """When the weight iterator yields nothing for LoRA, raise instead of silently succeeding."""

    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_raises_on_zero_lora_chunks(self, mock_iter_base, mock_dist, mock_ray, mock_gloo):
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()

        empty_iterator = MagicMock()
        empty_iterator.get_hf_weight_chunks.return_value = iter([])
        mock_iter_base.create.return_value = empty_iterator

        args = _make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        updater.rollout_engines = [MagicMock()]
        updater.use_distribute = False

        with pytest.raises(RuntimeError, match="zero chunks"):
            updater.update_weights()

    @patch("miles.backends.megatron_utils.update_weight.common.ray")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_no_raise_for_base_model_zero_chunks(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_common_ray
    ):
        """Base model weight sync with zero chunks is valid (e.g. empty model state)."""
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()

        empty_iterator = MagicMock()
        empty_iterator.get_hf_weight_chunks.return_value = iter([])
        mock_iter_base.create.return_value = empty_iterator

        args = _make_args()
        updater = UpdateWeightFromTensor(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            is_lora=False,
        )
        updater.rollout_engines = [MagicMock()]
        updater.use_distribute = False

        updater.update_weights()


# ---------------------------------------------------------------------------
# FlattenedTensorBucket round-trip correctness
# ---------------------------------------------------------------------------


class TestFlattenedTensorBucketRoundTrip:
    """Verify serialize -> reconstruct preserves tensor values exactly."""

    def _get_bucket_class(self):
        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            pytest.skip("sglang FlattenedTensorBucket not available")
        return FlattenedTensorBucket

    def test_roundtrip_single_dtype(self):
        FlattenedTensorBucket = self._get_bucket_class()
        tensors = [
            ("a", torch.randn(4, 4, dtype=torch.bfloat16)),
            ("b", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        bucket = FlattenedTensorBucket(named_tensors=tensors)
        reconstructed = bucket.reconstruct_tensors()

        assert len(reconstructed) == len(tensors)
        for (orig_name, orig_t), (rec_name, rec_t) in zip(tensors, reconstructed, strict=True):
            assert orig_name == rec_name
            assert orig_t.shape == rec_t.shape
            assert orig_t.dtype == rec_t.dtype
            assert torch.equal(orig_t, rec_t), f"Tensor {orig_name} values differ after round-trip"

    @pytest.mark.xfail(
        reason="SGLang FlattenedTensorBucket.reconstruct_tensors() fails with mixed dtypes "
        "due to PyTorch view() alignment requirements (storage_offset not divisible by "
        "element size). In practice LoRA weights are typically uniform dtype so this is safe.",
        raises=RuntimeError,
        strict=False,
    )
    def test_roundtrip_mixed_dtypes(self):
        FlattenedTensorBucket = self._get_bucket_class()

        if not getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            pytest.skip("FlattenedTensorBucket does not support multi-dtypes")

        tensors = [
            ("a_bf16", torch.randn(3, 3, dtype=torch.bfloat16)),
            ("b_fp32", torch.randn(2, 2, dtype=torch.float32)),
            ("c_fp16", torch.randn(5, dtype=torch.float16)),
        ]

        bucket = FlattenedTensorBucket(named_tensors=tensors)
        reconstructed = bucket.reconstruct_tensors()

        assert len(reconstructed) == len(tensors)
        for (orig_name, orig_t), (rec_name, rec_t) in zip(tensors, reconstructed, strict=True):
            assert orig_name == rec_name
            assert orig_t.dtype == rec_t.dtype
            assert torch.equal(orig_t, rec_t), f"Tensor {orig_name} values differ after round-trip"

    def test_roundtrip_from_flattened_data(self):
        """Simulate the receiver side: reconstruct from flattened_tensor + metadata."""
        FlattenedTensorBucket = self._get_bucket_class()

        original = [
            ("lora_A", torch.randn(8, 2, dtype=torch.bfloat16)),
            ("lora_B", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        sender_bucket = FlattenedTensorBucket(named_tensors=original)
        flat_tensor = sender_bucket.get_flattened_tensor()
        metadata = sender_bucket.get_metadata()

        receiver_bucket = FlattenedTensorBucket(flattened_tensor=flat_tensor, metadata=metadata)
        reconstructed = receiver_bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(original, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)

    def test_lora_only_tensors_filtered_correctly(self):
        """Verify that after filtering, only LoRA tensors survive and round-trip intact."""
        FlattenedTensorBucket = self._get_bucket_class()

        mixed = [
            ("model.layers.0.q_proj.weight", torch.randn(4, 4)),
            ("model.layers.0.q_proj.lora_A.weight", torch.randn(4, 2)),
            ("model.layers.0.q_proj.lora_B.weight", torch.randn(2, 4)),
        ]

        lora_only = [(n, t) for n, t in mixed if is_lora_weight_name(n)]
        assert len(lora_only) == 2

        bucket = FlattenedTensorBucket(named_tensors=lora_only)
        reconstructed = bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(lora_only, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)
