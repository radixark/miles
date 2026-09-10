from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest
from tests.fast.fixtures.driver_fakes import FakeObjectStore
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.data import RolloutDataPack, process_rollout_data, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.dp_schedule import TrainParallelConfig
from miles.utils.object_store import (
    BaseObjectStore,
    ObjectStoreGetResult,
    StoreObjectRef,
    ValueSpec,
    _MooncakeStoreObjectRef,
)


def _ref(payload: Any) -> StoreObjectRef:
    return _MooncakeStoreObjectRef(payload=payload)


class _RecordingStore(BaseObjectStore):
    def __init__(self) -> None:
        self.removed: list[StoreObjectRef] = []

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        return _ref(value)

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        raise NotImplementedError

    def remove(self, ref: StoreObjectRef) -> None:
        self.removed.append(ref)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    instance = _RecordingStore()
    monkeypatch.setattr(object_store, "_INSTANCE", instance)
    return instance


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(object_store, "_INSTANCE", FakeObjectStore())


class TestRemoveTrainOutputRefs:
    def test_every_shipped_ref_is_released(self, store: _RecordingStore):
        """Nothing else frees these objects, so a missed ref leaks for the whole run under mooncake."""
        refs = [_ref("a"), _ref("b")]

        remove_train_output_refs([TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref) for ref in refs])

        assert store.removed == refs

    def test_a_worker_that_shipped_nothing_is_skipped(self, store: _RecordingStore):
        """Only pp-last-stage critic workers ship values, so the rest carry None and must not reach the store."""
        ref = _ref("a")

        remove_train_output_refs(
            [
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
            ]
        )

        assert store.removed == [ref]


class TestRemoveRolloutDataRefs:
    def test_the_reference_the_rollout_shipped_is_released(self, store: _RecordingStore):
        """The driver is the only process that frees a rollout, once per step, or the store fills up."""
        ref = _ref("rollout")

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=ref))

        assert store.removed == [ref]

    def test_every_shard_of_a_split_rollout_is_released(self, store: _RecordingStore):
        """Without --delay-split-train-data-by-dp there is one object per dp rank, and each one pins memory."""
        refs = [_ref("a"), _ref("b")]

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=refs))

        assert store.removed == refs

    def test_a_pack_that_carries_no_data_never_reaches_the_store(self, store: _RecordingStore):
        """An empty-batch timeout ships no object, and asking the store to free None would raise."""
        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(empty_batch_timeout=True))

        assert store.removed == []


_NUM_SAMPLES = 256
_DP_SIZE = 3
_TRAIN_PARALLEL_CONFIG = make_train_parallel_config(
    dp_size=_DP_SIZE,
    cp_size=2,
    vpp_size=1,
    microbatch_group_size_per_vp_stage=None,
    independent_dp=True,
    supports_precomputed_schedule=True,
)


def _delay_split_args(**overrides: Any) -> Namespace:
    fields: dict[str, Any] = dict(
        delay_split_train_data_by_dp=True,
        global_batch_size=_NUM_SAMPLES,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=64,
        micro_batch_size=None,
        balance_data=False,
        balance_by_flops=False,
        allow_partial_train_step=False,
        multi_lora=False,
    )
    fields.update(overrides)
    return Namespace(**fields)


def _raw_batch(*, num_samples: int = _NUM_SAMPLES, **extra: Any) -> dict[str, Any]:
    lengths = [(i * 37) % 41 + 1 for i in range(num_samples)]
    return {
        "tokens": [list(range(length)) for length in lengths],
        "loss_masks": [[1] * length for length in lengths],
        "response_lengths": lengths,
        "rewards": [float(i) for i in range(num_samples)],
        "sample_indices": list(range(num_samples)),
        "rollout_ids": list(range(num_samples)),
        **extra,
    }


def _singleton_batch() -> dict[str, Any]:
    raw = _raw_batch()
    raw["tokens"] = [list(range(65)) for _ in range(_NUM_SAMPLES)]
    raw["loss_masks"] = [[1] * 65 for _ in range(_NUM_SAMPLES)]
    raw["response_lengths"] = [65] * _NUM_SAMPLES
    return raw


def _multimodal_batch() -> dict[str, Any]:
    return _raw_batch(multimodal_train_inputs=[None] * _NUM_SAMPLES)


def _batch_without_rollout_ids() -> dict[str, Any]:
    raw = _raw_batch()
    del raw["rollout_ids"]
    return raw


def _shards_of_every_rank(
    *,
    args: Namespace | None = None,
    raw: dict[str, Any] | None = None,
    witness_info: WitnessInfo | None = None,
    train_parallel_config: TrainParallelConfig | None = None,
) -> list[dict]:
    args = args or _delay_split_args()
    raw = raw or _raw_batch()
    if train_parallel_config is None:
        train_parallel_config = _TRAIN_PARALLEL_CONFIG
    ref = object_store.get_instance().put(value=raw)
    return [
        process_rollout_data(
            args,
            ref,
            dp_rank=dp_rank,
            train_parallel_config=train_parallel_config,
            witness_info=witness_info,
        )[0]
        for dp_rank in range(train_parallel_config.dp_size)
    ]


@pytest.mark.usefixtures("fake_store")
class TestProcessRolloutData:
    def test_dp_size_only_backend_keeps_the_strided_split_with_delayed_split(self) -> None:
        """Backends advertising only dp_size keep the legacy split without a precomputed schedule."""
        shards = _shards_of_every_rank(
            args=_delay_split_args(allow_partial_train_step=True),
            raw=_raw_batch(),
            train_parallel_config=make_train_parallel_config(dp_size=3),
        )

        for dp_rank, shard in enumerate(shards):
            assert "micro_batch_indices" not in shard
            assert "num_rollouts" not in shard
            assert shard["sample_indices"] == list(range(dp_rank, 256, 3))

    @pytest.mark.parametrize("balance_data", [False, True])
    def test_three_live_cells_partition_a_batch_none_of_them_divides(self, balance_data: bool) -> None:
        """256 packable samples over 3 cells must land in exactly one shard each, balanced or not."""
        shards = _shards_of_every_rank(args=_delay_split_args(balance_data=balance_data), raw=_raw_batch())

        covered = [i for shard in shards for i in shard["sample_indices"]]
        assert sorted(covered) == list(range(_NUM_SAMPLES))

    def test_singleton_batch_raises_when_equal_counts_cannot_be_aligned(self) -> None:
        """256 indivisible micro-batches cannot be distributed equally over 3 cells."""
        with pytest.raises(AssertionError, match="maximal splitting"):
            _shards_of_every_rank(raw=_singleton_batch())

    def test_every_cell_denominates_by_the_global_rollout_count(self):
        """Loss, LR and metrics divide by num_rollouts, which must not shrink with the cell count."""
        shards = _shards_of_every_rank()

        assert [shard["num_rollouts"] for shard in shards] == [[_NUM_SAMPLES]] * _DP_SIZE

    def test_cells_hold_equal_micro_batch_counts_and_different_sample_counts(self) -> None:
        """Cells share micro-batch counts even when their sample counts differ."""
        shards = _shards_of_every_rank()

        assert [shard["num_microbatches"] for shard in shards] == [shards[0]["num_microbatches"]] * _DP_SIZE
        for shard in shards:
            assert len(shard["micro_batch_indices"]) == sum(shard["num_microbatches"])
            assert sorted(i for mb in shard["micro_batch_indices"] for i in mb) == list(range(len(shard["tokens"])))
        assert len({len(shard["tokens"]) for shard in shards}) > 1

    def test_every_cell_derives_the_same_schedule_from_the_same_inputs(self):
        """Cells never exchange the schedule, so a second derivation must reproduce the first bit for bit."""
        first = _shards_of_every_rank()
        second = _shards_of_every_rank()

        for a, b in zip(first, second, strict=True):
            assert (a["sample_indices"], a["micro_batch_indices"]) == (b["sample_indices"], b["micro_batch_indices"])

    def test_total_lengths_follow_the_shard_row_order(self):
        """A scheduled shard reorders rows, and total_lengths must be reordered with them."""
        shards = _shards_of_every_rank()

        for shard in shards:
            assert shard["total_lengths"] == [len(tokens) for tokens in shard["tokens"]]

    def test_witness_ids_stay_attached_to_their_sequences(self):
        """Witness ids are merged before the split, so each shard row keeps the id of its own sequence."""
        witness_info = WitnessInfo(witness_ids=[1000 + i for i in range(_NUM_SAMPLES)], stale_ids=[])

        shards = _shards_of_every_rank(witness_info=witness_info)

        for shard in shards:
            assert shard["seq_witness_ids"] == [1000 + i for i in shard["sample_indices"]]

    def test_a_trailing_rollout_with_witness_ids_raises_when_the_schedule_drops_it(self) -> None:
        """Dropping a trailing rollout must fail when its row already has a witness id."""
        raw = _raw_batch(num_samples=257)
        witness_info = WitnessInfo(witness_ids=[1000 + i for i in range(257)], stale_ids=[])

        with pytest.raises(AssertionError, match="the schedule dropped 1 rows"):
            _shards_of_every_rank(raw=raw, witness_info=witness_info)

    def test_two_full_steps_keep_every_row_and_its_witness_id(self) -> None:
        """Two full steps preserve all rows and keep witness ids aligned with their sequences."""
        raw = _raw_batch(num_samples=512)
        witness_info = WitnessInfo(witness_ids=[1000 + i for i in range(512)], stale_ids=[])

        shards = _shards_of_every_rank(raw=raw, witness_info=witness_info)

        assert sorted(i for shard in shards for i in shard["sample_indices"]) == list(range(512))
        for shard in shards:
            assert shard["seq_witness_ids"] == [1000 + i for i in shard["sample_indices"]]

    @pytest.mark.parametrize("raw", [_multimodal_batch(), _batch_without_rollout_ids()])
    def test_unschedulable_batches_fall_back_to_the_strided_split(self, raw: dict[str, Any]):
        """Multimodal batches and batches without rollout_ids keep the legacy training-side split."""
        shards = _shards_of_every_rank(raw=raw)

        for dp_rank, shard in enumerate(shards):
            assert "micro_batch_indices" not in shard
            assert "num_rollouts" not in shard
            assert shard["sample_indices"] == list(range(dp_rank, _NUM_SAMPLES, _DP_SIZE))

    def test_a_pre_split_rollout_must_carry_one_shard_per_cell(self):
        """Without the delayed split the rollout side already fixed the shard count, which must match dp_size."""
        args = _delay_split_args(delay_split_train_data_by_dp=False)

        with pytest.raises(AssertionError):
            process_rollout_data(
                args,
                [_ref(_raw_batch()), _ref(_raw_batch())],
                dp_rank=0,
                train_parallel_config=_TRAIN_PARALLEL_CONFIG,
                witness_info=None,
            )
