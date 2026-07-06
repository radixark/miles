"""Tests for LoRA weight-sync validation logic.

Verifies that silent failures are caught:
- Engine returning success=False raises RuntimeError
- Empty LoRA weights after filtering raises RuntimeError
- Zero weight chunks from iterator raises RuntimeError
- FlattenedTensorBucket round-trip preserves tensor values
- Distributed (disaggregate) sync broadcasts the adapter over NCCL (no CUDA IPC)
"""

from argparse import Namespace
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_weight_name
from miles.backends.megatron_utils.update_weight.common import _check_weight_sync_results
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
    DistBucketedWeightUpdateMixin,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"
_MIXIN_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"
_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"

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
        results = [_FakeEngineResult(success=True)]
        _check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_lora(self):
        results = [_FakeEngineResult(success=False, error_message="incompatible format")]
        with pytest.raises(RuntimeError, match="LoRA weight sync failed"):
            _check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_base(self):
        results = [_FakeEngineResult(success=False, error_message="bad version")]
        with pytest.raises(RuntimeError, match="Base model weight sync failed"):
            _check_weight_sync_results(results, is_lora=False)

    def test_plain_tuple_results_pass(self):
        """Non-dataclass results (e.g. (True, 'Success') tuples) should not raise."""
        results = [(True, "Success")]
        _check_weight_sync_results(results, is_lora=False)

    def test_mixed_results_raises_on_first_failure(self):
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
            updater._send_lora_params(SAMPLE_BASE_ONLY_WEIGHTS)

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_passes_when_lora_weights_present(self, mock_iter_base, mock_dist, mock_send):
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

        refs, _ = updater._send_lora_params(SAMPLE_LORA_WEIGHTS)
        # Should not raise; mock_send was called with the LoRA tensors
        assert mock_send.called


# ---------------------------------------------------------------------------
# update_weights: zero-chunk detection
# ---------------------------------------------------------------------------


class TestUpdateWeightsZeroChunks:
    """When the weight iterator yields nothing for LoRA, raise instead of silently succeeding."""

    @patch("miles.backends.megatron_utils.update_weight.common.ray")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_raises_on_zero_lora_chunks(self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_common_ray):
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

    def test_roundtrip_mixed_dtypes(self):
        """FIXME(sglang upstream contract): SGLang exposes
        ``FlattenedTensorBucket.supports_multi_dtypes = True`` but
        ``reconstruct_tensors()`` actually raises ``RuntimeError`` on mixed
        dtypes, because PyTorch ``view()`` requires ``storage_offset`` to be
        divisible by the target element size and concatenated flat buffers do
        not align across heterogeneous element sizes.

        This is a latent production landmine: ``_send_to_colocated_engine`` in
        ``miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py``
        reads the flag and packs mixed dtypes into a single bucket. In practice
        LoRA weights are uniform dtype, but FP8 / INT4 mixed-precision base
        weight sync would crash on sglang's receiver.

        Fix path (either side):
          - miles side: stop trusting ``supports_multi_dtypes`` in
            ``_send_to_colocated_engine`` and always group by dtype (matches
            the FSDP path's existing implementation in
            ``experimental/fsdp_utils/update_weight_utils.py``).
          - sglang side: actually align ``storage_offset`` in reconstruction.

        Until one side is fixed, this test asserts the current observed
        failure so we notice when either side changes.
        """
        FlattenedTensorBucket = self._get_bucket_class()

        tensors = [
            ("a_bf16", torch.randn(3, 3, dtype=torch.bfloat16)),
            ("b_fp32", torch.randn(2, 2, dtype=torch.float32)),
            ("c_fp16", torch.randn(5, dtype=torch.float16)),
        ]
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        with pytest.raises(RuntimeError, match=r"storage_offset"):
            bucket.reconstruct_tensors()

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


# ---------------------------------------------------------------------------
# Distributed (disaggregate) LoRA sync. The base-weight split is mirrored:
#   - DistBucketedWeightUpdateMixin._update_lora_weights  → shared orchestration
#       (bridge iteration, guards, source gating, engine lock, unload-on-reload)
#   - <subclass>._update_lora_weight_implementation       → transport (NCCL / p2p)
# ---------------------------------------------------------------------------


class _FakeRemote:
    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _FakeEngine:
    def __init__(self, load_result=None):
        self.load_lora_adapter_from_distributed = _FakeRemote(result=load_result)
        self.unload_lora_adapter = _FakeRemote()


class TestDistLoraUpdateOrchestration:
    """Shared ``_update_lora_weights``: transport-agnostic orchestration.

    It must enforce the silent-failure guards (zero chunks, no LoRA names), gate
    on the source rank, unload a stale adapter before reload, and delegate the
    actual transmit to ``_update_lora_weight_implementation`` (mocked here).
    """

    @staticmethod
    def _make_self(*, engines, chunks=None, is_source=True, lora_loaded=False):
        if chunks is None:
            chunks = [SAMPLE_LORA_WEIGHTS]
        return SimpleNamespace(
            _hf_weight_iterator=SimpleNamespace(get_hf_weight_chunks=lambda *a, **k: iter(chunks)),
            _is_lora_source=is_source,
            _lora_loaded=lora_loaded,
            rollout_engines=engines,
            _update_lora_weight_implementation=MagicMock(name="impl"),
        )

    @staticmethod
    def _run(fake_self):
        with patch(f"{_MIXIN_MODULE}.ray") as ray_mock:
            ray_mock.get.side_effect = lambda refs: refs
            DistBucketedWeightUpdateMixin._update_lora_weights(fake_self)

    def test_delegates_accumulated_tensors_to_implementation(self):
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines)
        self._run(fake_self)
        fake_self._update_lora_weight_implementation.assert_called_once()
        (sent,) = fake_self._update_lora_weight_implementation.call_args.args
        assert sent == SAMPLE_LORA_WEIGHTS
        assert fake_self._lora_loaded is True

    def test_non_source_rank_does_not_transmit(self):
        # Non-source ranks still iterate the bridge (TP collectives) but must not
        # transmit. They also must not short-circuit the zero-chunk guard.
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines, is_source=False)
        self._run(fake_self)
        fake_self._update_lora_weight_implementation.assert_not_called()
        assert engines[0].unload_lora_adapter.calls == []

    def test_raises_on_zero_chunks(self):
        # Mirror of TestUpdateWeightsZeroChunks: empty iterator must not silently succeed.
        fake_self = self._make_self(engines=[_FakeEngine()], chunks=[])
        with pytest.raises(RuntimeError, match="zero chunks"):
            self._run(fake_self)
        fake_self._update_lora_weight_implementation.assert_not_called()

    def test_raises_when_chunk_has_no_lora_weights(self):
        # Mirror of TestSendHfParamsEmptyLoraDetection: base-only names => raise.
        fake_self = self._make_self(engines=[_FakeEngine()], chunks=[SAMPLE_BASE_ONLY_WEIGHTS])
        with pytest.raises(RuntimeError, match="no LoRA weights"):
            self._run(fake_self)
        fake_self._update_lora_weight_implementation.assert_not_called()

    def test_reload_unloads_existing_adapter_first(self):
        # When an adapter is already loaded, the stale one must be unloaded before
        # the new weights are pushed, else SGLang rejects the duplicate name.
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines, lora_loaded=True)
        self._run(fake_self)
        assert engines[0].unload_lora_adapter.calls == [{"lora_name": LORA_ADAPTER_NAME}]
        fake_self._update_lora_weight_implementation.assert_called_once()

    def test_first_load_does_not_unload(self):
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines, lora_loaded=False)
        self._run(fake_self)
        assert engines[0].unload_lora_adapter.calls == []

    def test_lora_loaded_stays_false_when_implementation_raises(self):
        fake_self = self._make_self(engines=[_FakeEngine()])
        fake_self._update_lora_weight_implementation.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            self._run(fake_self)
        assert fake_self._lora_loaded is False


class TestBroadcastLoraImplementation:
    """Broadcast transport ``UpdateWeightFromDistributed._update_lora_weight_implementation``:
    send metadata over Ray, then ``dist.broadcast`` each adapter tensor over the
    reused base group (src=0) — no CUDA IPC, valid across nodes.
    """

    @staticmethod
    def _make_self(*, engines):
        return SimpleNamespace(
            rollout_engines=engines,
            _lora_config={"peft_type": "LORA", "r": 32, "lora_alpha": 32},
            _group_name="miles-pp_0",
            _model_update_groups=MagicMock(name="base_nccl_group"),
        )

    @staticmethod
    def _run(fake_self, named_tensors):
        # NB: the real _check_weight_sync_results runs (not patched), so an engine
        # returning success=False propagates as RuntimeError exactly as in prod.
        with (
            patch(f"{_BROADCAST_MODULE}.dist") as dist_mock,
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
        ):
            ray_mock.get.side_effect = lambda refs: refs
            UpdateWeightFromDistributed._update_lora_weight_implementation(fake_self, named_tensors)
        return dist_mock

    def test_sends_metadata_rpc_and_broadcasts_each_tensor(self):
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines)
        dist_mock = self._run(fake_self, SAMPLE_LORA_WEIGHTS)

        kwargs = engines[0].load_lora_adapter_from_distributed.calls[0]
        assert kwargs["lora_name"] == LORA_ADAPTER_NAME
        assert kwargs["config_dict"] == fake_self._lora_config
        assert kwargs["group_name"] == "miles-pp_0"
        # Metadata describes every adapter tensor, no IPC payload.
        assert kwargs["names"] == [n for n, _ in SAMPLE_LORA_WEIGHTS]
        assert kwargs["dtypes"] == [t.dtype for _, t in SAMPLE_LORA_WEIGHTS]
        assert kwargs["shapes"] == [list(t.shape) for _, t in SAMPLE_LORA_WEIGHTS]
        # One NCCL broadcast (src=0, shared base group) per tensor.
        assert dist_mock.broadcast.call_count == len(SAMPLE_LORA_WEIGHTS)
        for call in dist_mock.broadcast.call_args_list:
            assert call.args[1] == 0
            assert call.kwargs["group"] is fake_self._model_update_groups

    def test_each_engine_gets_one_rpc(self):
        engines = [_FakeEngine(), _FakeEngine()]
        fake_self = self._make_self(engines=engines)
        self._run(fake_self, SAMPLE_LORA_WEIGHTS)
        assert all(len(e.load_lora_adapter_from_distributed.calls) == 1 for e in engines)

    def test_raises_when_engine_reports_failure(self):
        # Mirror of TestCheckWeightSyncResults: a success=False result propagates.
        engines = [_FakeEngine(load_result=_FakeEngineResult(success=False, error_message="incompatible format"))]
        fake_self = self._make_self(engines=engines)
        with pytest.raises(RuntimeError, match="LoRA weight sync failed"):
            self._run(fake_self, SAMPLE_LORA_WEIGHTS)
