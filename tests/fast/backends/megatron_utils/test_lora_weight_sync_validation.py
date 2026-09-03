"""Tests for LoRA weight-sync validation logic.

Verifies that silent failures are caught:
- Engine returning success=False raises RuntimeError
- Iterator validation errors (empty export / no lora names) propagate through
  the orchestration (the guards themselves are covered in
  tests/fast/backends/training_utils/test_hf_weight_iterator.py)
- FlattenedTensorBucket round-trip preserves tensor values
- Distributed (disaggregate) sync broadcasts the adapter over NCCL (no CUDA IPC)
"""

from argparse import Namespace
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_weight_name

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"
_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
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
# check_weight_sync_results
# ---------------------------------------------------------------------------


class TestCheckWeightSyncResults:
    """Validate that check_weight_sync_results raises on engine failures."""

    def test_success_results_pass(self):
        results = [_FakeEngineResult(success=True)]
        check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_lora(self):
        results = [_FakeEngineResult(success=False, error_message="incompatible format")]
        with pytest.raises(RuntimeError, match="LoRA weight sync failed"):
            check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_base(self):
        results = [_FakeEngineResult(success=False, error_message="bad version")]
        with pytest.raises(RuntimeError, match="Base model weight sync failed"):
            check_weight_sync_results(results, is_lora=False)

    def test_plain_tuple_results_pass(self):
        """Non-dataclass results (e.g. (True, 'Success') tuples) should not raise."""
        results = [(True, "Success")]
        check_weight_sync_results(results, is_lora=False)

    def test_mixed_results_raises_on_first_failure(self):
        results = [
            _FakeEngineResult(success=True),
            _FakeEngineResult(success=False, error_message="oops"),
        ]
        with pytest.raises(RuntimeError, match="oops"):
            check_weight_sync_results(results, is_lora=True)


# ---------------------------------------------------------------------------
# Colocated send_adapter: transport pass-through (export guards are covered in
# tests/fast/backends/training_utils/test_hf_weight_iterator.py)
# ---------------------------------------------------------------------------


class TestColocatedSendAdapter:
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.ipc_collect")
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    @patch(f"{_UW_MODULE}.dist")
    def test_passes_when_lora_weights_present(self, mock_dist, mock_send, mock_ray, _ipc, _cache):
        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_ray.get.side_effect = lambda refs: refs

        protocol = UpdateWeightFromTensor(_make_args(check_lora_weight_equal=False, offload_train=False))
        protocol._ipc_engine = MagicMock()
        protocol._ipc_gather_src = 0
        protocol._ipc_gather_group = MagicMock()
        protocol.use_distribute = False

        protocol.send_adapter(
            SAMPLE_LORA_WEIGHTS,
            lora_name=LORA_ADAPTER_NAME,
            lora_config={"peft_type": "LORA", "r": 32, "lora_alpha": 32},
            upsert=False,
        )
        assert mock_send.called

    @patch(f"{_UW_MODULE}.dist")
    def test_multi_lora_upsert_rejected(self, mock_dist):
        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()

        protocol = UpdateWeightFromTensor(_make_args())
        with pytest.raises(NotImplementedError, match="multi-LoRA"):
            protocol.send_adapter(SAMPLE_LORA_WEIGHTS, lora_name="slot_0", lora_config={}, upsert=True)


# ---------------------------------------------------------------------------
# update_weights: empty base iteration
# ---------------------------------------------------------------------------


class TestUpdateWeightsEmptyBaseIteration:
    def test_no_raise_for_base_model_zero_buckets(self):
        """Base model weight sync with zero buckets is valid (e.g. empty model state)."""
        empty_iterator = MagicMock()
        empty_iterator.iter_hf_base_weights.return_value = iter([])

        protocol = MagicMock()
        protocol.uses_session_frame = True
        protocol.needs_base_resync_for_lora = False
        protocol.is_sender = True
        protocol.group_name = "test"
        protocol.begin_sync.return_value = True
        protocol.rollout_engines = [MagicMock()]

        args = _make_args(custom_model_provider_path=None)
        with (
            patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol),
            patch(f"{_UPDATER_MODULE}.dist") as mock_dist,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_UPDATER_MODULE}.pause_engines"),
            patch(f"{_UPDATER_MODULE}.begin_weight_update"),
            patch(f"{_UPDATER_MODULE}.set_weight_version"),
            patch(f"{_UPDATER_MODULE}.end_weight_update"),
            patch(f"{_UPDATER_MODULE}.resume_engines"),
        ):
            mock_dist.get_rank.return_value = 0
            updater = WeightUpdater(
                args,
                [MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen",
                quantization_config=None,
                iterator_factory=lambda *a, **k: empty_iterator,
                parallel_state=MagicMock(),
                is_lora=False,
            )
            updater.update_weights()

        protocol.send_bucket.assert_not_called()
        protocol.after_base_weights.assert_called_once()
        assert updater.weight_version == 1


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
            ``fsdp_utils/update_weight_utils.py``).
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
#   - WeightUpdater._send_lora_adapter  → shared orchestration
#       (bridge iteration, guards, source gating, unload-on-reload)
#   - <protocol>.send_adapter           → transport (NCCL / p2p)
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
    """Shared ``WeightUpdater._send_lora_adapter``: transport-agnostic orchestration.

    It must enforce the silent-failure guards (zero chunks, no LoRA names), gate
    on the source rank, unload a stale adapter before reload, and delegate the
    actual transmit to ``protocol.send_adapter`` (mocked here).
    """

    @staticmethod
    def _make_self(*, named_tensors=None, is_source=True):
        if named_tensors is None:
            named_tensors = SAMPLE_LORA_WEIGHTS
        return SimpleNamespace(
            _hf_weight_iterator=SimpleNamespace(get_hf_lora_weights=lambda *a, **k: named_tensors),
            protocol=SimpleNamespace(is_lora_sender=is_source, send_adapter=MagicMock(name="send_adapter")),
            _lora_sync_config={"peft_type": "LORA", "r": 32, "lora_alpha": 32},
        )

    def test_delegates_accumulated_tensors_to_protocol(self):
        fake_self = self._make_self()
        WeightUpdater._send_lora_adapter(fake_self)
        fake_self.protocol.send_adapter.assert_called_once()
        (sent,) = fake_self.protocol.send_adapter.call_args.args
        assert sent == SAMPLE_LORA_WEIGHTS
        kwargs = fake_self.protocol.send_adapter.call_args.kwargs
        assert kwargs["lora_name"] == LORA_ADAPTER_NAME
        assert kwargs["lora_config"] == fake_self._lora_sync_config
        assert kwargs["upsert"] is False

    def test_non_source_rank_does_not_transmit(self):
        # Non-source ranks still call the iterator (TP collectives) but must not
        # transmit.
        fake_self = self._make_self(is_source=False)
        WeightUpdater._send_lora_adapter(fake_self)
        fake_self.protocol.send_adapter.assert_not_called()

    def test_iterator_validation_errors_propagate(self):
        # The export guards live in get_hf_lora_weights; orchestration must not swallow them.
        def _raise(*_a, **_k):
            raise RuntimeError("LoRA weight sync failed: the weight iterator produced zero chunks.")

        fake_self = self._make_self()
        fake_self._hf_weight_iterator = SimpleNamespace(get_hf_lora_weights=_raise)
        with pytest.raises(RuntimeError, match="zero chunks"):
            WeightUpdater._send_lora_adapter(fake_self)
        fake_self.protocol.send_adapter.assert_not_called()


class TestBroadcastLoraImplementation:
    """Broadcast transport ``UpdateWeightFromDistributed.send_adapter``:
    send metadata over Ray, then ``dist.broadcast`` each adapter tensor over the
    reused base group (src=0) — no CUDA IPC, valid across nodes.
    """

    LORA_CONFIG = {"peft_type": "LORA", "r": 32, "lora_alpha": 32}

    @staticmethod
    def _make_self(*, engines, lora_loaded=False):
        return SimpleNamespace(
            rollout_engines=engines,
            group_name="miles-pp_0",
            _model_update_groups=MagicMock(name="base_nccl_group"),
            _lora_loaded=lora_loaded,
        )

    def _run(self, fake_self, named_tensors, upsert=False):
        # NB: the real check_weight_sync_results runs (not patched), so an engine
        # returning success=False propagates as RuntimeError exactly as in prod.
        with (
            patch(f"{_BROADCAST_MODULE}.dist") as dist_mock,
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.unload_lora_adapter") as unload_mock,
        ):
            ray_mock.get.side_effect = lambda refs: refs
            UpdateWeightFromDistributed.send_adapter(
                fake_self,
                named_tensors,
                lora_name=LORA_ADAPTER_NAME,
                lora_config=self.LORA_CONFIG,
                upsert=upsert,
            )
        self._unload_mock = unload_mock
        return dist_mock

    def test_sends_metadata_rpc_and_broadcasts_each_tensor(self):
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines)
        dist_mock = self._run(fake_self, SAMPLE_LORA_WEIGHTS)

        kwargs = engines[0].load_lora_adapter_from_distributed.calls[0]
        assert kwargs["lora_name"] == LORA_ADAPTER_NAME
        assert kwargs["config_dict"] == self.LORA_CONFIG
        assert "upsert" not in kwargs
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

    def test_upsert_flag_reaches_engine_rpc(self):
        engines = [_FakeEngine()]
        fake_self = self._make_self(engines=engines)
        self._run(fake_self, SAMPLE_LORA_WEIGHTS, upsert=True)
        assert engines[0].load_lora_adapter_from_distributed.calls[0]["upsert"] is True

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
        assert fake_self._lora_loaded is False

    def test_reload_unloads_stale_adapter_first(self):
        # When an adapter is already loaded, the stale one must be unloaded before
        # the new weights are pushed, else SGLang rejects the duplicate name.
        fake_self = self._make_self(engines=[_FakeEngine()], lora_loaded=True)
        self._run(fake_self, SAMPLE_LORA_WEIGHTS)
        self._unload_mock.assert_called_once_with(fake_self.rollout_engines, LORA_ADAPTER_NAME)
        assert fake_self._lora_loaded is True

    def test_first_load_does_not_unload(self):
        fake_self = self._make_self(engines=[_FakeEngine()], lora_loaded=False)
        self._run(fake_self, SAMPLE_LORA_WEIGHTS)
        self._unload_mock.assert_not_called()
        assert fake_self._lora_loaded is True

    def test_upsert_does_not_unload_or_mark_loaded(self):
        fake_self = self._make_self(engines=[_FakeEngine()], lora_loaded=True)
        self._run(fake_self, SAMPLE_LORA_WEIGHTS, upsert=True)
        self._unload_mock.assert_not_called()
        assert fake_self._lora_loaded is True
