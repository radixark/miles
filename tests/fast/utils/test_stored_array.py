from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from tests.fast.fixtures.object_store_fixtures import InMemoryStore

from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.stored_array import (
    StoredArray,
    StoredArrayHandle,
    store_replay_arrays,
    stored_array_checkpoint_reads,
    stored_array_checkpoint_writes,
)
from miles.utils.types import Sample


def routed_experts(num_tokens: int = 5) -> np.ndarray:
    return np.arange(num_tokens * 3 * 2, dtype=np.int32).reshape(num_tokens, 3, 2)


class TestStoredArray:
    def test_the_trainer_reads_back_the_exact_array(self, in_memory_store: InMemoryStore):
        array = routed_experts()

        stored = StoredArray.put(array)
        read = StoredArrayHandle.model_validate_json(stored.handle.model_dump_json()).read()

        assert read.dtype == array.dtype
        np.testing.assert_array_equal(read, array)
        assert len(stored) == 5 and stored.shape == (5, 3, 2)

    def test_dropping_a_sample_removes_its_object(self, in_memory_store: InMemoryStore):
        """A filtered, stale or aborted group is dropped anywhere in the rollout; none of those sites remove by hand."""
        sample = Sample(rollout_routed_experts=StoredArray.put(routed_experts()))

        del sample

        assert in_memory_store.objects == {} and in_memory_store.removed == [0]

    def test_a_handed_off_object_outlives_the_sample(self, in_memory_store: InMemoryStore):
        """After the hand-off the driver removes it, once the trainer has read it."""
        sample = Sample(rollout_routed_experts=StoredArray.put(routed_experts()))

        ref = sample.rollout_routed_experts.hand_off()
        del sample

        assert in_memory_store.removed == []
        assert ref.payload == {"key": 0}

    def test_a_second_hand_off_fails(self, in_memory_store: InMemoryStore):
        stored = StoredArray.put(routed_experts())
        stored.hand_off()

        with pytest.raises(AssertionError, match="handed off at most once"):
            stored.hand_off()

    def test_a_deep_copy_shares_the_owner(self, in_memory_store: InMemoryStore):
        """The output snapshotter deep-copies batches; dropping the copy must not remove the original's object."""
        sample = Sample(rollout_routed_experts=StoredArray.put(routed_experts()))

        snapshot = copy.deepcopy(sample)
        del snapshot
        assert in_memory_store.removed == []

        del sample
        assert in_memory_store.removed == [0]

    def test_numpy_conversion_fails_instead_of_wrapping_the_reference(self, in_memory_store: InMemoryStore):
        with pytest.raises(TypeError, match="holds no bytes"):
            np.asarray(StoredArray.put(routed_experts()))

    def test_pickling_outside_a_checkpoint_fails(self, in_memory_store: InMemoryStore):
        with pytest.raises(AssertionError, match="stored_array_checkpoint_writes"):
            pickle.dumps(StoredArray.put(routed_experts()))


class TestCheckpoint:
    def test_a_checkpoint_restores_the_array_into_a_new_object(self, in_memory_store: InMemoryStore, tmp_path):
        array = routed_experts()
        state = [Sample(rollout_routed_experts=StoredArray.put(array))]

        with stored_array_checkpoint_writes(tmp_path / "stored_arrays"):
            save_simple_checkpoint(directory=tmp_path / "state", data=state)
        with stored_array_checkpoint_reads(tmp_path / "stored_arrays"):
            restored = load_simple_checkpoint(directory=tmp_path / "state")

        restored_stored = restored[0].rollout_routed_experts
        assert isinstance(restored_stored, StoredArray)
        assert restored_stored.handle.ref != state[0].rollout_routed_experts.handle.ref
        np.testing.assert_array_equal(restored_stored.handle.read(), array)
        assert len(list((tmp_path / "stored_arrays").iterdir())) == 1

    def test_the_saved_state_keeps_its_object(self, in_memory_store: InMemoryStore, tmp_path):
        """Saving reads the bytes; the buffer still owns the object it saved."""
        stored = StoredArray.put(routed_experts())

        with stored_array_checkpoint_writes(tmp_path / "stored_arrays"):
            save_simple_checkpoint(directory=tmp_path / "state", data=[stored])

        assert in_memory_store.removed == [] and list(in_memory_store.objects) == [0]


class TestStoreReplayArrays:
    def test_replay_arrays_are_stored_and_other_fields_stay(self, in_memory_store: InMemoryStore):
        indexer_topk = np.ones((5, 2, 4), dtype=np.int32)
        sample = Sample(tokens=[1, 2, 3], rollout_routed_experts=routed_experts(), rollout_indexer_topk=indexer_topk)

        store_replay_arrays([sample])

        assert isinstance(sample.rollout_routed_experts, StoredArray)
        assert isinstance(sample.rollout_indexer_topk, StoredArray)
        np.testing.assert_array_equal(sample.rollout_indexer_topk.handle.read(), indexer_topk)
        assert sample.tokens == [1, 2, 3]

    def test_a_sample_stored_in_an_earlier_attempt_is_not_stored_again(self, in_memory_store: InMemoryStore):
        sample = Sample(rollout_routed_experts=routed_experts())
        store_replay_arrays([sample])
        stored = sample.rollout_routed_experts

        store_replay_arrays([sample])

        assert sample.rollout_routed_experts is stored
        assert len(in_memory_store.objects) == 1
